# 💐 Flora: Low-Rank Adapters Are Secretly Gradient Compressors

This is the official repository for the paper [Flora: Low-Rank Adapters Are Secretly Gradient Compressors](https://arxiv.org/abs/2402.03293). This repository contains the code for the experiments in the paper.

## Installation

You can install the library using pip:

```bash
pip install 'flora-opt[torch]' # for PyTorch
```

or

```bash
pip install 'flora-opt[optax]' # for JAX
```

## Usage

The library is designed to be compatible with huggingface libraries. You can use it as a drop-in replacement for huggingface libraries. Here is an example for PyTorch:

```diff
- optimizer = transformers.AdamW(model.parameters(), lr=1e-5)
+ optimizer = flora_opt.Flora(model.parameters(), lr=1e-5)
- accelerator = accelerate.Accelerator(**kwargs)
+ accelerator = flora_opt.FloraAccelerator(**kwargs)
```

Everything else remains the same. You can find more examples in the `examples` folder.

## Paper Replications (with JAX)

To replicate major experiments in the paper, run the following commands:

```bash
sh replicate.sh
```

You can also run individual experiments by selecting the corresponding script in the file `replicate.sh`.

## Explanation

The arguments for the `Flora` optimizer (for PyTorch) are explained below:

```python
flora_opt.Flora(
    params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],  # model parameters
    lr: float = None,  # learning rate
    rank: int = None,  # rank of the low-rank random projections
    kappa: int = 1000,  # the interval for updating the low-rank random projections
    eps: tuple[float, float] = (1e-30, 1e-3),  # Adafactor parameter
    clip_threshold: float = 1.0,  # Adafactor parameter
    decay_rate: float = -0.8,  # decay rate in Adafactor
    beta1: Optional[float] = None,  # decay rate for the first moment
    weight_decay: float = 0.0,  # weight decay coefficient
    scale_parameter: bool = True,  # Adafactor parameter
    relative_step: bool = False,  # Adafactor parameter
    warmup_init: bool = False,  # Adafactor parameter
    factorize_second_moment: bool = True,  # use Adafactor or Adam
    seed: int = 0,  # random seed to generate the low-rank random projections
    quantization: bool = False,  # whether to quantize the states
)
```

For JAX, the arguments are similar. The translation can be found in `flora_opt/optimizers/torch/__init__.py`.

### Citation

```bibtex
@article{hao2024flora,
  title={Flora: Low-Rank Adapters Are Secretly Gradient Compressors},
  author={Hao, Yongchang and Cao, Yanshuai and Mou, Lili},
  journal={arXiv preprint arXiv:2402.03293},
  year={2024}
}
```
