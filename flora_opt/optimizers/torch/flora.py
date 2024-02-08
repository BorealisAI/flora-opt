# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# The implementation of `_approx_sq_grad` is adapted from https://github.com/huggingface/transformers/blob/8395f14de6068012787d83989c3627c3df6a252b/src/transformers/optimization.py#L505 by Huggingface  # noqa
#####################################################################################


import contextlib
import logging
import math
from functools import partial
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Sequence, Union

import bitsandbytes.functional as bnbf
import torch
from accelerate import Accelerator
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)
quant_fn = bnbf.quantize_blockwise
dequant_fn = bnbf.dequantize_blockwise


def stable_randn(
    shape: Union[int, Sequence[int]],
    seed: int,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(seed)
    rn = torch.randn(shape, generator=generator, device=generator.device, dtype=dtype)
    return rn


def next_seed(seed: int, adv: int = 0xF) -> int:
    """
    This is a naive helper function to generate a new seed from the given seed.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, torch.iinfo(torch.int64).max, (adv,), generator=generator, device=generator.device
    ).tolist()[-1]


def split_seed(seed: int) -> tuple[int, int]:
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randint(0, torch.iinfo(torch.int64).max, (2,), generator=generator, device=generator.device).tolist()
    )


def with_flora_accelerator(
    step_function: Callable[[Optimizer, dict, torch.Tensor, torch.Tensor], Optional[float]],
) -> Callable[[Accelerator, Optimizer, dict, torch.Tensor, Optional[torch.Tensor]], None]:
    def wrapper(accelerator, optimizer, group, p, *args, **kwargs):
        with torch.no_grad():
            # Compress and clear grad
            if not accelerator.is_factorized(p):
                if p in accelerator.packs:
                    accelerator.packs[p].add_(p.grad)
                else:
                    accelerator.packs[p] = p.grad
            else:
                seed = accelerator.seeds[p]
                accelerator.packs[p] = accelerator._down_project(seed, p.grad) + accelerator.packs.get(p, 0.0)

            p.grad = None

            if not accelerator.sync_gradients:
                return

            # Decompress and update grad
            if not accelerator.is_factorized(p):
                p.grad = accelerator.packs[p]
            else:
                p.grad = accelerator._up_project(accelerator.seeds[p], p, accelerator.packs[p])
                accelerator.seeds[p] = next_seed(seed)

            del accelerator.packs[p]

            step_function(optimizer, group, p, p.grad, *args, **kwargs)

            del p.grad

    return wrapper


def flora_step_(optimizer: Optimizer, group: Dict, p: torch.Tensor, grad: Optional[torch.Tensor] = None):
    """
    Performs a single optimization step

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """

    if grad is None:
        grad = p.grad
    p.grad = None

    if grad.is_sparse:
        raise RuntimeError("Adafactor does not support sparse gradients.")

    if grad.dtype in {torch.float16, torch.bfloat16}:
        grad = grad.float()

    p_data_fp32 = p
    if p.dtype in {torch.float16, torch.bfloat16}:
        p_data_fp32 = p_data_fp32.float()

    state = optimizer.state[p]
    grad_shape = grad.shape

    factored, use_first_moment, factored_momentum = optimizer._get_options(group, grad_shape)

    # State Initialization
    if "step" not in state:
        state["step"] = 0

        if use_first_moment:
            # Exponential moving average of gradient values
            if not factored_momentum:
                state["exp_avg"] = torch.zeros_like(grad)
            else:
                if grad_shape[0] < grad_shape[-1]:
                    cshape = (group["rank"], grad_shape[-1])
                else:
                    cshape = (grad_shape[0], group["rank"])
                state["exp_avg"] = torch.zeros(cshape).to(grad)
            if group["quantization"]:
                state["exp_avg"] = quant_fn(state["exp_avg"])
        if factored:
            state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
            state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
        else:
            state["exp_avg_sq"] = torch.zeros_like(grad)
            if group["quantization"]:
                state["exp_avg_sq"] = quant_fn(state["exp_avg_sq"])

        state["RMS"] = 0

    state["step"] += 1
    state["RMS"] = optimizer._rms(p_data_fp32)
    lr = optimizer._get_lr(group, state)
    b1 = group["beta1"]
    beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])

    update = (grad**2) + group["eps"][0]
    if factored:
        exp_avg_sq_row = state["exp_avg_sq_row"]
        exp_avg_sq_col = state["exp_avg_sq_col"]

        exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
        exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))

        # Approximation of exponential moving average of square of gradient
        update = optimizer._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
        update.mul_(grad)
    else:
        if group["quantization"]:
            exp_avg_sq = dequant_fn(*state["exp_avg_sq"])
        else:
            exp_avg_sq = state["exp_avg_sq"]
        exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
        update = exp_avg_sq.rsqrt().mul_(grad)
        if group["quantization"]:
            state["exp_avg_sq"] = quant_fn(exp_avg_sq)

    if use_first_moment:
        if group["quantization"]:
            state["exp_avg"] = dequant_fn(*state["exp_avg"])
        if not factored_momentum:
            update = update * (1 - b1) + state["exp_avg"] * b1
            state["exp_avg"].copy_(update)
        else:
            # Factorized update
            def _down_proj(seed, rank, tensor):
                lseed, rseed = split_seed(seed)
                if tensor.shape[0] < tensor.shape[-1]:
                    left_projection = stable_randn(
                        (rank, tensor.shape[0]),
                        seed=lseed,
                        device=tensor.device,
                        dtype=tensor.dtype,
                    ) / math.sqrt(rank)

                    return left_projection @ tensor
                else:
                    right_projection = stable_randn(
                        (tensor.shape[-1], rank),
                        seed=rseed,
                        device=tensor.device,
                        dtype=tensor.dtype,
                    ) / math.sqrt(rank)
                return tensor @ right_projection

            def _up_proj(seed, rank, shape, ctensor):
                lseed, rseed = split_seed(seed)
                if shape[0] < shape[-1]:
                    left_projection = stable_randn(
                        (rank, shape[0]),
                        seed=lseed,
                        device=ctensor.device,
                        dtype=ctensor.dtype,
                    ) / math.sqrt(rank)
                    return left_projection.t() @ ctensor
                else:
                    right_projection = stable_randn(
                        (shape[-1], rank),
                        seed=rseed,
                        device=ctensor.device,
                        dtype=ctensor.dtype,
                    ) / math.sqrt(rank)
                    return ctensor @ right_projection.t()

            _current_seed = state["seed"]

            raw_update = update.clone()

            update.copy_(
                update * (1 - b1)
                + _up_proj(seed=_current_seed, rank=group["rank"], shape=update.shape, ctensor=state["exp_avg"]) * b1
            )

            if state["step"] % group["kappa"] == 0:
                _next_seed = next_seed(state["seed"])
                state["exp_avg"].copy_(
                    _down_proj(
                        seed=_next_seed,
                        rank=group["rank"],
                        tensor=_up_proj(
                            seed=_current_seed, rank=group["rank"], shape=grad.shape, ctensor=state["exp_avg"]
                        ),
                    )
                )
                state["seed"] = _next_seed
                _current_seed = _next_seed

            state["exp_avg"].copy_(
                state["exp_avg"] * b1 + _down_proj(seed=_current_seed, rank=group["rank"], tensor=raw_update) * (1 - b1)
            )
        if group["quantization"]:
            state["exp_avg"] = quant_fn(state["exp_avg"])

    update.div_((optimizer._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
    update.mul_(lr)
    if group["weight_decay"] != 0:
        p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

    p_data_fp32.add_(-update)

    if p.dtype in {torch.float16, torch.bfloat16}:
        p.copy_(p_data_fp32)

    del grad, p_data_fp32, update


class Flora(Optimizer):
    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: float = None,
        rank: int = None,
        kappa: int = 1000,
        eps: tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = False,
        warmup_init: bool = False,
        factorize_second_moment: bool = True,
        seed: int = 0,
        quantization: bool = False,
    ) -> None:
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual `lr` and `relative_step=True` options")
        if warmup_init and not relative_step:
            raise ValueError("`warmup_init=True` requires `relative_step=True`")

        defaults = {
            "lr": lr,
            "rank": rank,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "beta1": beta1,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
            "kappa": kappa,
            "factorize_second_moment": factorize_second_moment,
            "quantization": quantization,
        }
        super().__init__(params, defaults)

        params_idx = seed
        for group in self.param_groups:
            for p in group["params"]:
                params_idx += 1
                if p.requires_grad:
                    self.state[p]["seed"] = params_idx

    @staticmethod
    def _get_lr(param_group: Dict, param_state: Dict) -> float:
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    @staticmethod
    def _get_options(param_group: Dict, param_shape: tuple[int, ...]) -> tuple[bool, bool, bool]:
        factored = len(param_shape) == 2 and param_group["factorize_second_moment"]
        use_first_moment = param_group["beta1"] is not None and param_group["beta1"] > 0
        factored_momentum = (
            use_first_moment
            and param_group["rank"] is not None
            and param_group["rank"] > 0
            and factored
            and min(param_shape) >= param_group["rank"]
            and max(param_shape) / min(param_shape) <= 4  # rule out embeddings
        )
        return factored, use_first_moment, factored_momentum

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row: torch.Tensor, exp_avg_sq_col: torch.Tensor) -> torch.Tensor:
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @torch.no_grad()
    def step(self, closure: bool = None) -> Optional[float]:
        """
        Performs a single optimization step

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                flora_step_(self, group, p, p.grad)

        return loss


class FloraAccelerator(Accelerator):
    def __init__(self, accumulation_compression_rank: bool = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._accumulation_compression = (
            accumulation_compression_rank is not None and self.gradient_accumulation_steps > 1
        )
        self.packs = {}
        self.seeds = {}
        self.proj_rank = accumulation_compression_rank

    def is_factorized(self, p: torch.Tensor) -> bool:
        return p.dim() >= 2 and self._accumulation_compression

    def _down_project(self, seed: int, grad: torch.Tensor) -> torch.Tensor:
        if grad.shape[0] < grad.shape[-1]:
            proj = stable_randn(
                (self.proj_rank, grad.shape[0]), seed=seed, device=grad.device, dtype=grad.dtype
            ) / math.sqrt(self.proj_rank)
            return proj @ grad
        else:
            proj = stable_randn(
                (grad.shape[-1], self.proj_rank), seed=seed, device=grad.device, dtype=grad.dtype
            ) / math.sqrt(self.proj_rank)
            return grad @ proj

    def _up_project(self, seed: int, param: torch.Tensor, compressed_grad: torch.Tensor) -> torch.Tensor:
        if param.shape[0] < param.shape[-1]:
            proj = stable_randn(
                (self.proj_rank, param.shape[0]), seed=seed, device=param.device, dtype=compressed_grad.dtype
            ) / math.sqrt(self.proj_rank)
            return proj.t() @ compressed_grad
        else:
            proj = stable_randn(
                (param.shape[-1], self.proj_rank), seed=seed, device=param.device, dtype=compressed_grad.dtype
            ) / math.sqrt(self.proj_rank)
            return compressed_grad @ proj.t()

    def prepare_optimizer(self, optimizer: Optimizer, device_placement: bool = None) -> Optimizer:
        for group in optimizer.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                if self.is_factorized(p):
                    self.seeds[p] = len(self.seeds)

                p.register_post_accumulate_grad_hook(
                    partial(with_flora_accelerator(flora_step_), self, optimizer, group)
                )
        return super().prepare_optimizer(optimizer, device_placement)

    @contextlib.contextmanager
    def accumulate(self, *models: torch.nn.Module) -> Generator[None, None, None]:
        self._do_sync()
        with contextlib.ExitStack() as cm_stack:
            for m in models:
                cm_stack.enter_context(contextlib.nullcontext())
            yield
