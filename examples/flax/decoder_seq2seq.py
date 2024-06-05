# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2021, The HuggingFace Team
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on Huggingface's example from https://github.com/huggingface/transformers/blob/main/examples/flax/summarization/run_summarization_flax.py by HuggingFace  # noqa
# The implementation of the loss function is adapted from https://github.com/google/flax/blob/87a211135c6a377c8f29048a1cac3840e38b9da4/examples/wmt/train.py#L104 by Google  # noqa
####################################################################################


import copy
import logging
import math
import os
import time
from functools import partial
from typing import Callable

import datasets
import evaluate
import hydra
import jax
import jax.numpy as jnp
import jax.sharding as shd
import nltk
import numpy as np
import optax
import transformers
from datasets import load_dataset
from filelock import FileLock
from flax import traverse_util
from flax.training import train_state
from flax.training.common_utils import onehot
from jax.lax import with_sharding_constraint
from mlorax import LoRASpec, lora_init
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForCausalLM,
    is_wandb_available,
    set_seed,
)
from transformers.utils import is_offline_mode

import wandb
from examples.flax.utils import data_loader
from flora_opt.optimizers.optax import compressed_acc, get_optimizer
from examples.flax.sharding import (
    get_batch_sharding,
    get_current_sharding,
    get_params_sharding,
)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


logger = logging.getLogger(__name__)
datasets.disable_caching()


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


seq2seq_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "e2e_nlg": ("meaning_representation", "human_reference"),
    "e2e_nlg_cleaned": ("meaning_representation", "human_reference"),
}


def format_input(data, input_str, prefix):
    if data in ["e2e_nlg", "e2e_nlg_cleaned"]:
        """
        original: name[The Mill], eatType[coffee shop]
        formatted: name: The Mill | eatType: coffee shop
        """
        input_str = input_str.replace(", ", " | ")
        input_str = input_str.replace("[", ": ")
        input_str = input_str.replace("]", "")

    return prefix + input_str


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray


def estimate_grad(loss_fn, state, impl="reverse"):
    if impl == "reverse":
        return jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    else:
        raise NotImplementedError


def accumulator_fn(steps, impl="naive", tau=None):
    if impl == "naive":
        return (
            lambda s: jax.tree_map(jnp.zeros_like, s.params),
            lambda acc, x, ministep: jax.tree_map(lambda ga, g: ga + (g - ga) / (ministep + 1), acc, x),
            lambda acc, p: jax.tree_map(lambda x: x / steps, acc),
        )
    elif impl == "compressed":
        return compressed_acc(steps=steps, factored=True, tau=tau)
    else:
        raise NotImplementedError


def state_update(grads, state, weight_decay, **kwargs):
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_params = jax.tree_map(lambda x, y: x - weight_decay * y, new_params, state.params)
    return (
        state.replace(
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        ),
        updates,
    )


def write_train_metric(train_metrics, train_time, step):
    length = len(train_metrics)
    for i, metric in enumerate(train_metrics):
        for key, val in metric.items():
            tag = f"train/{key}"
            wandb.log({tag: val}, step=step - length + i + 1)
    wandb.log({"train/time": train_time}, step=step)


def write_eval_metric(eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        tag = f"eval/{metric_name}"
        wandb.log({tag: value}, step=step)


def create_learning_rate_fn(
    train_ds_size: int,
    train_batch_size: int,
    num_train_epochs: int,
    num_warmup_steps: int,
    learning_rate: float,
    lr_decay: bool = True,
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=num_warmup_steps,
    )
    decay_fn = optax.linear_schedule(
        init_value=learning_rate,
        end_value=0.0 if lr_decay else learning_rate,
        transition_steps=num_train_steps - num_warmup_steps,
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def main(cfg):
    if (
        os.path.exists(cfg.training.output_dir)
        and os.listdir(cfg.training.output_dir)
        and cfg.training.do_train
        and not cfg.training.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({cfg.training.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    logger.info(f"Training/evaluation parameters {cfg.training}")

    # Set seed before initializing model.
    set_seed(cfg.training.seed)

    if cfg.data.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            cfg.data.dataset_name,
            cfg.data.dataset_config_name,
            cache_dir=cfg.model.cache_dir,
            keep_in_memory=False,
            use_auth_token=True if cfg.model.use_auth_token else None,
        )
    else:
        raise ValueError("Need a dataset name")
    # Load pretrained model and tokenizer

    if cfg.model.config_name:
        config = AutoConfig.from_pretrained(
            cfg.model.config_name,
            cache_dir=cfg.model.cache_dir,
        )
    elif cfg.model.model_name_or_path:
        config = AutoConfig.from_pretrained(
            cfg.model.model_name_or_path,
            cache_dir=cfg.model.cache_dir,
        )
    else:
        config = CONFIG_MAPPING[cfg.model.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if cfg.model.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.tokenizer_name,
            cache_dir=cfg.model.cache_dir,
            use_fast=cfg.model.use_fast_tokenizer,
            use_auth_token=True if cfg.model.use_auth_token else None,
            padding_side="left",
        )
    elif cfg.model.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.model_name_or_path,
            cache_dir=cfg.model.cache_dir,
            use_fast=cfg.model.use_fast_tokenizer,
            use_auth_token=True if cfg.model.use_auth_token else None,
            padding_side="left",
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    with jax.default_device(jax.devices("cpu")[0]):
        if cfg.model.model_name_or_path and cfg.model.pretrained:
            model = FlaxAutoModelForCausalLM.from_pretrained(
                cfg.model.model_name_or_path,
                seed=cfg.training.seed,
                cache_dir=cfg.model.cache_dir,
                dtype=getattr(jnp, cfg.model.dtype),
                use_auth_token=cfg.model.use_auth_token,
                from_pt=cfg.model.from_pt,
                revision=cfg.model.revision,
            )
        else:
            model = FlaxAutoModelForCausalLM.from_config(
                config,
                seed=cfg.training.seed,
                dtype=getattr(jnp, cfg.model.dtype),
            )
        model.params = jax.tree_map(lambda x: x.astype(getattr(jnp, cfg.model.dtype)), model.params)

    rng = jax.random.PRNGKey(cfg.training.seed)
    rng, dropout_rng = jax.random.split(rng)

    prefix = cfg.data.source_prefix if cfg.data.source_prefix is not None else ""
    suffix = cfg.data.source_suffix if cfg.data.source_suffix is not None else ""
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if cfg.training.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        column_names = dataset["train"].column_names
    elif cfg.training.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = dataset["validation"].column_names
    elif cfg.training.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        column_names = dataset["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = seq2seq_name_mapping.get(cfg.data.dataset_name, None)
    if cfg.data.source_column is None:
        source_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        source_column = cfg.data.source_column

    if cfg.data.target_column is None:
        target_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        target_column = cfg.data.target_column

    # Temporarily set max_target_length for training.
    max_target_length = cfg.data.max_target_length

    # Setting padding="max_length" as we need fixed length inputs for jitted functions
    def preprocess_function(examples):
        def get_sample(column_name):
            columns = column_name.split(".")
            _examples = examples[columns[0]]
            for column in columns[1:]:
                _examples = [ex[column] for ex in _examples]
            return _examples

        inputs = get_sample(source_column)
        targets = get_sample(target_column)
        inputs = [prefix + inp for inp in inputs]

        model_inputs = {}
        sep_ids = tokenizer(
            suffix,
            add_special_tokens=False,
        )["input_ids"]

        contexts = [format_input(cfg.data.dataset_name, inp, prefix) for inp in inputs]
        context_ids = tokenizer(
            contexts,
            max_length=cfg.data.max_source_length - len(sep_ids),
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]
        context_ids = [ids + sep_ids for ids in context_ids]

        target_ids = tokenizer(
            targets,
            max_length=max_target_length - 1,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]

        model_inputs["labels"] = np.full(
            shape=(len(contexts), cfg.data.max_source_length + max_target_length),
            fill_value=-100,
            dtype=np.int32,
        )
        model_inputs["input_ids"] = np.full_like(
            model_inputs["labels"], dtype=np.int32, fill_value=tokenizer.eos_token_id
        )
        model_inputs["attention_mask"] = np.zeros_like(model_inputs["labels"], dtype=np.int32)

        for i in range(len(contexts)):
            model_inputs["input_ids"][i][0] = tokenizer.bos_token_id
            model_inputs["input_ids"][i][1 : 1 + len(context_ids[i])] = context_ids[i]
            model_inputs["input_ids"][i][1 + len(context_ids[i]) : 1 + len(context_ids[i]) + len(target_ids[i])] = (
                target_ids[i]
            )

            model_inputs["labels"][i][len(context_ids[i]) : len(context_ids[i]) + len(target_ids[i])] = target_ids[i]
            model_inputs["labels"][i][len(context_ids[i]) + len(target_ids[i])] = tokenizer.eos_token_id

            model_inputs["attention_mask"][i][: len(context_ids[i]) + len(target_ids[i]) + 1] = 1

        return model_inputs

    def test_preprocess_function(examples):
        def get_sample(column_name):
            columns = column_name.split(".")
            _examples = examples[columns[0]]
            for column in columns[1:]:
                _examples = [ex[column] for ex in _examples]
            return _examples

        contexts = get_sample(source_column)
        targets = get_sample(target_column)

        model_inputs = {}

        sep_ids = tokenizer(
            suffix,
            add_special_tokens=False,
        )["input_ids"]

        contexts = [format_input(cfg.data.dataset_name, inp, prefix) for inp in contexts]
        context_ids = tokenizer(
            contexts,
            max_length=cfg.data.max_source_length - len(sep_ids),
            truncation=True,
            padding="max_length",
            add_special_tokens=False,
        )
        model_inputs["input_ids"] = np.asarray([ids + sep_ids for ids in context_ids["input_ids"]])
        model_inputs["attention_mask"] = np.asarray(
            [attn + [1] * len(sep_ids) for attn in context_ids["attention_mask"]]
        )
        model_inputs["labels"] = targets

        return model_inputs

    if cfg.training.do_train:
        train_dataset = dataset["train"]
        if cfg.data.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), cfg.data.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if cfg.training.do_eval:
        max_target_length = cfg.data.val_max_target_length
        eval_dataset = dataset["validation"]
        if cfg.data.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), cfg.data.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if cfg.training.do_predict:
        max_target_length = cfg.data.val_max_target_length
        predict_dataset = dataset["test"]
        if cfg.data.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), cfg.data.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Metric
    evaluators = [evaluate.load(metric) for metric in cfg.data.metrics]

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(preds, decoded_labels):
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        results = {}
        for evaluator in evaluators:
            eval_kwargs = {}
            if evaluator.name == "rouge":
                eval_kwargs["use_stemmer"] = True
            result = evaluator.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                **eval_kwargs,
            )
            result = {k: v for k, v in result.items() if not isinstance(v, list)}
            results.update(result)
        prediction_lens = [np.count_nonzero(pred != tokenizer.eos_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return results

    # Enable tensorboard only on the master node
    has_wandb = is_wandb_available()
    if has_wandb and jax.process_index() == 0:
        lr_str = f"-lr({cfg.optimizer.learning_rate})"
        tau_str = f"-tau({cfg.optimizer.tau})" if hasattr(cfg.optimizer, "tau") else ""

        b1_candidates = ["b1", "beta1", "momentum"]
        b1_str = ""
        for b1_name in b1_candidates:
            if hasattr(cfg.optimizer, b1_name) and getattr(cfg.optimizer, b1_name) is not None:
                b1_str = f"-b1({getattr(cfg.optimizer, b1_name)})"
                break
        b2_candidates = ["b2", "beta2", "decay"]
        b2_str = ""
        for b2_name in b2_candidates:
            if hasattr(cfg.optimizer, b2_name) and getattr(cfg.optimizer, b2_name) is not None:
                b2_str = f"-b2({getattr(cfg.optimizer, b2_name)})"
                break
        weight_decay_str = f"-wd({cfg.training.weight_decay})" if cfg.training.weight_decay > 0 else ""
        name = f"{cfg.optimizer.name}{lr_str}{tau_str}{b1_str}{b2_str}{weight_decay_str}"

        try:
            wandb.init(
                project="decoder-seq2seq",
                config=dict(OmegaConf.to_container(cfg, resolve=True)),
                dir=f"{cfg.training.output_dir}/",
                name=name,
            )
            wandb.define_metric("eval/loss", summary="min")
            wandb.define_metric("train/loss", summary="min")

            name = name + os.environ.get("SLURM_JOB_ID", wandb.run.id)
        except ImportError as ie:
            has_wandb = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )

    # Initialize our training

    # Store some constant
    num_epochs = int(cfg.training.num_train_epochs)
    train_batch_size = (
        int(cfg.training.per_device_train_batch_size)
        * jax.device_count()
        * cfg.grad_acc.steps
        // cfg.training.num_tp_devices
    )
    per_device_eval_batch_size = int(
        cfg.training.per_device_eval_batch_size or cfg.training.per_device_train_batch_size
    )
    eval_batch_size = per_device_eval_batch_size * jax.device_count() // cfg.training.num_tp_devices
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    # Create learning rate schedule

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        # find out all LayerNorm parameters
        layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
        layer_norm_named_params = {
            layer[-2:]
            for layer_norm_name in layer_norm_candidates
            for layer in flat_params.keys()
            if layer_norm_name in "".join(layer).lower()
        }
        flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    optimizer_cfg = OmegaConf.to_object(cfg.optimizer)
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        cfg.training.num_train_epochs,
        cfg.training.warmup_steps,
        optimizer_cfg.pop("learning_rate"),
        lr_decay=cfg.training.lr_decay,
    )
    optimizer_name = optimizer_cfg.pop("name")
    optimizer = get_optimizer(optimizer_name, learning_rate=linear_decay_lr_schedule_fn, **optimizer_cfg)

    devices = np.asarray(jax.devices()).reshape((-1, cfg.training.num_tp_devices))
    mesh = shd.Mesh(devices, ("dp", "tp"))

    params_sharding = get_params_sharding(mesh, model.params, model_name=cfg.model.model_name_or_path)

    dummy_batch = next(
        data_loader(
            rng,
            train_dataset,
            train_batch_size,
            shuffle=True,
            transform_fn=preprocess_function,
        )
    )
    batch_sharding = get_batch_sharding(mesh, dummy_batch)
    dummy_batch = jax.device_put(dummy_batch, batch_sharding)

    model.params = jax.device_put(model.params, params_sharding)

    # Setup train state
    lora_spec = LoRASpec(
        rank=cfg.lora.rank,
        rules=cfg.lora.rules,
        alpha=cfg.lora.alpha,
        tune_vectors=cfg.lora.tune_vectors,
        tune_others=cfg.lora.tune_others,
        seed=cfg.training.seed,
        disabled=cfg.lora.disabled,
        dropout=cfg.lora.dropout,
        logger_level=cfg.lora.logger_level,
    )
    trainable, apply_fn, merge_fn = lora_init(
        lora_spec,
        model,
    )

    params_sharding = get_current_sharding(mesh, trainable)
    state = TrainState.create(
        apply_fn=apply_fn,
        params=trainable,
        tx=optimizer,
        dropout_rng=dropout_rng,
    )

    state_sharding = get_current_sharding(mesh, state)
    state = jax.device_put(state, state_sharding)

    # label smoothed cross entropy
    def loss_fn(logits, labels, padding_mask, label_smoothing_factor=0.0):
        padding_mask = jnp.where(labels < 0, jnp.zeros_like(padding_mask), padding_mask)
        vocab_size = logits.shape[-1]
        confidence = 1.0 - label_smoothing_factor
        low_confidence = (1.0 - confidence) / (vocab_size - 1)
        normalizing_constant = -(
            confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
        )
        soft_labels = onehot(labels, vocab_size, on_value=confidence, off_value=low_confidence)

        loss = optax.softmax_cross_entropy(logits, soft_labels)
        loss = loss - normalizing_constant

        # ignore padded tokens from loss
        loss = loss * padding_mask
        loss = loss.sum()
        num_labels = padding_mask.sum()
        return loss / num_labels, num_labels

    # Define gradient update step fn

    @partial(
        jax.jit,
        in_shardings=(
            state_sharding,
            batch_sharding,
        ),
        out_shardings=(state_sharding, None),
        donate_argnums=(0,),
    )
    def train_step(state, batch):
        # batch.shape = (dev * ga * bsz, seqlen)

        def compute_loss(params, inputs, rng):
            inputs = copy.deepcopy(inputs)
            labels = inputs.pop("labels")
            logits = state.apply_fn(**inputs, params=params, dropout_rng=rng, train=True)[0]
            loss, num_labels = loss_fn(
                logits,
                labels,
                batch["attention_mask"],
                cfg.training.label_smoothing_factor,
            )
            return loss, num_labels

        if cfg.grad_acc.steps > 1:

            @jax.remat
            def for_body(idx, carry):
                gl_loss, gl_grads, gl_rng = carry
                step_batch = jax.tree_map(lambda x: x[idx], batch)
                step_batch = with_sharding_constraint(step_batch, batch_sharding)
                step_loss_fn = partial(compute_loss, inputs=step_batch, rng=gl_rng)
                (step_loss, aux), grads = estimate_grad(step_loss_fn, state)

                # Use Welford algorithm for numerically stable aggregation of mean.
                gl_loss = gl_loss + (step_loss - gl_loss) / (idx + 1)
                gl_grads = accumulate_update(gl_grads, grads, idx)
                gl_rng = jax.random.split(gl_rng)[-1]
                return (gl_loss, gl_grads, gl_rng)

            batch = jax.tree_map(
                lambda x: x.reshape((cfg.grad_acc.steps, -1) + x.shape[1:]),
                batch,
            )

            accumulate_init, accumulate_update, accumulate_build = accumulator_fn(
                steps=cfg.grad_acc.steps, impl=cfg.grad_acc.impl, tau=cfg.grad_acc.tau
            )

            (loss, acc_grads, dropout_rng) = jax.lax.fori_loop(
                0,
                cfg.grad_acc.steps,
                for_body,
                (
                    0.0,
                    accumulate_init(state),
                    state.dropout_rng,
                ),
            )
            grads = accumulate_build(acc_grads, state.params)
            updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
            new_params = optax.apply_updates(state.params, updates)
            new_params = jax.tree_map(lambda x, y: x - cfg.training.weight_decay * y, new_params, state.params)
            new_state = state.replace(
                params=new_params,
                opt_state=new_opt_state,
                dropout_rng=dropout_rng,
                step=state.step + 1,
            )

        else:
            batch = with_sharding_constraint(batch, batch_sharding)
            step_loss_fn = partial(compute_loss, inputs=batch, rng=state.dropout_rng)
            (loss, aux), grads = estimate_grad(step_loss_fn, state)
            dropout_rng = jax.random.split(state.dropout_rng)[-1]

            new_state, updates = state_update(
                grads=grads,
                state=state,
                weight_decay=cfg.training.weight_decay,
                dropout_rng=dropout_rng,
                step=state.step + 1,
            )

        metrics = {
            "loss": loss,
            "learning_rate": linear_decay_lr_schedule_fn(new_state.step),
            "updates_norm": optax.global_norm(updates),
        }

        return new_state, metrics

    # Define eval fn
    @jax.jit
    def eval_step(state, batch, label_smoothing_factor=0.0):
        labels = batch.pop("labels")
        logits = state.apply(**batch, params=merge_fn(state.params), train=False)[0]
        loss, num_labels = loss_fn(
            logits,
            labels,
            batch["attention_mask"],
            label_smoothing_factor,
        )

        metrics = {"loss": loss}
        return metrics

    # Define generation function
    max_length = cfg.data.max_source_length + cfg.data.val_max_target_length

    num_beams = cfg.data.num_beams if cfg.data.num_beams is not None else model.config.num_beams
    length_penalty = cfg.data.length_penalty if cfg.data.length_penalty is not None else model.config.length_penalty
    gen_kwargs = {
        "max_length": max_length,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
    }

    @jax.jit
    def generate_step(state, batch):
        output_ids = model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            params=merge_fn(state.params),
            **gen_kwargs,
        ).sequences[:, batch["input_ids"].shape[1] :]
        return output_ids

    # dummy run
    tick = time.time()
    logger.info("***** Compiling *****")
    compiled_update = train_step.lower(state, dummy_batch).compile()
    tock = time.time()
    logger.info(f"Compilation took {tock - tick} seconds")
    state, _ = compiled_update(state, dummy_batch)
    try:
        memory = sum(jax.devices()[i].memory_stats()["peak_bytes_in_use"] for i in range(jax.device_count())) / 1024**3
    except Exception:
        memory = jnp.nan
    logger.info(f"Peak memory usage: {memory} GiB")
    wandb.run.summary["memory"] = memory
    wandb.run.summary["compilation"] = tock - tick
    # Replicate the train state on each device

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.training.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")
    wandb.run.summary.update(
        {
            "num_examples": len(train_dataset),
            "num_epochs": num_epochs,
            "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
            "total_train_batch_size": train_batch_size,
            "total_optimization_steps": total_train_steps,
        }
    )

    train_start = time.time()
    train_metrics = []

    epochs = tqdm(range(num_epochs), desc="Epoch ... ", position=0)
    for epoch in epochs:
        # ======================== Training ================================

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)
        steps_per_epoch = len(train_dataset) // train_batch_size
        train_loader = data_loader(
            input_rng,
            train_dataset,
            train_batch_size,
            shuffle=True,
            transform_fn=preprocess_function,
        )

        # train
        for step in tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False):
            batch = next(train_loader)  # (#dev * #gacc * bsz, seqlen)
            batch = jax.device_put(batch, batch_sharding)
            state, train_metric = compiled_update(state, batch)
            train_metric = jax.device_get(train_metric)
            train_metrics.append(train_metric)
            cur_step = epoch * (len(train_dataset) // train_batch_size) + step
            if cur_step % cfg.training.logging_steps == 0 and cur_step > 0:
                train_time = time.time() - train_start
                if has_wandb and jax.process_index() == 0:
                    write_train_metric(train_metrics, train_time, cur_step)
                    train_metrics = []

            if (cur_step % cfg.training.eval_steps == 0 or cur_step + 1 == total_train_steps) and cur_step > 0:
                # ======================== Evaluating ==============================
                eval_metrics = []
                eval_preds = []
                eval_labels = []
                eval_loader = data_loader(
                    input_rng,
                    eval_dataset,
                    eval_batch_size,
                    drop_last=False,
                    transform_fn=preprocess_function,
                )
                eval_steps = math.ceil(len(eval_dataset) / eval_batch_size)
                for _ in tqdm(
                    range(eval_steps),
                    desc="Evaluating...",
                    position=2,
                    leave=False,
                ):
                    # Model forward
                    batch = next(eval_loader)
                    labels = batch["labels"]

                    metrics = eval_step(state, batch, cfg.training.label_smoothing_factor)
                    eval_metrics.append(metrics)

                eval_keys = eval_metrics[0].keys()
                eval_metrics = {k: jnp.stack([metrics[k] for metrics in eval_metrics]).mean() for k in eval_keys}

                if cfg.data.predict_with_generate:
                    eval_preds = []
                    eval_labels = []
                    predict_loader = data_loader(
                        input_rng,
                        predict_dataset,
                        eval_batch_size,
                        drop_last=False,
                        transform_fn=test_preprocess_function,
                    )
                    predict_steps = math.ceil(len(predict_dataset) / eval_batch_size)
                    for _ in tqdm(
                        range(predict_steps),
                        desc="Predicting...",
                        position=2,
                        leave=False,
                    ):
                        batch = next(predict_loader)
                        labels = batch.pop("labels")

                        generated_ids = generate_step(state, batch)
                        generated_ids = jax.device_get(generated_ids)

                        eval_preds.extend(generated_ids)
                        eval_labels.extend(labels)

                    gen_metrics = compute_metrics(eval_preds, eval_labels)
                    eval_metrics.update(gen_metrics)

                # Print metrics and update progress bar
                desc = f"Step... ({cur_step}"
                for key, val in eval_metrics.items():
                    desc += f" | {key}: {val}"
                epochs.write(desc)
                epochs.desc = desc

                # Save metrics
                if has_wandb and jax.process_index() == 0:
                    write_eval_metric(eval_metrics, cur_step)


@hydra.main(version_base=None, config_path="configs", config_name="dec_s2s")
def launch(cfg: OmegaConf) -> None:
    jax.config.update("jax_default_prng_impl", "rbg")
    if cfg.training.output_dir is not None:
        cfg.training.output_dir = os.path.expanduser(cfg.training.output_dir)

    if cfg.data.dataset_name is None and cfg.data.train_file is None and cfg.data.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if cfg.data.train_file is not None:
            extension = cfg.data.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("train_file` should be a csv, json or text file.")
        if cfg.data.validation_file is not None:
            extension = cfg.data.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or text file.")
    if cfg.data.val_max_target_length is None:
        cfg.data.val_max_target_length = cfg.data.max_target_length

    main(cfg)


if __name__ == "__main__":
    launch()
