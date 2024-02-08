# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Any

import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np
from flax import traverse_util

logger = logging.getLogger(__name__)


def is_ndarray(x):
    return isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)


def get_batch_sharding(mesh, inputs):
    return_dict = {}
    for k, v in inputs.items():
        if not is_ndarray(v) or v.ndim == 0:
            _spec_tuple = []
        else:
            _spec_tuple = ["dp"]
        p = shd.PartitionSpec(*_spec_tuple)
        return_dict[k] = shd.NamedSharding(mesh, p)
    return return_dict


def _to_t5_sharding(mesh, k, v):
    if not is_ndarray(v) or v.ndim == 0:
        # usually scalars
        return shd.NamedSharding(mesh, shd.PartitionSpec())
    if "Attention" in k and ("kernel" in k or "embedding" in k):
        if "relative_attention" in k:
            # relative attention
            _spec_tuple = (None, None)
        elif "Attention.o" in k:
            # o project
            _spec_tuple = ("tp", None)
        else:
            # q, k, v projects
            _spec_tuple = (None, "tp")
    elif "DenseReluDense" in k:
        if "DenseReluDense.wo" in k:
            # FFN o project
            _spec_tuple = ("tp", None)
        else:
            # FFN in projects
            _spec_tuple = (None, "tp")
    elif "shared" in k:
        # shared embeddings
        _spec_tuple = ("tp", None)
    elif "lm_head" in k:
        # lm head embeddings
        _spec_tuple = (None, "tp")
    else:
        # usually layer norm or bias
        if not v.ndim == 1:
            raise ValueError(f"Unexpected 1D tensor {k}")
        _spec_tuple = (None,)
    assert len(_spec_tuple) == v.ndim
    p = shd.PartitionSpec(*_spec_tuple)
    return shd.NamedSharding(mesh, p)


def _to_bart_sharding(mesh, k, v):
    if not is_ndarray(v) or v.ndim == 0:
        # usually scalars
        return shd.NamedSharding(mesh, shd.PartitionSpec())
    if "_attn" in k and "kernel" in k:
        if "out_proj" in k:
            _spec_tuple = ("tp", None)
        else:
            # q, k, v projects
            _spec_tuple = (None, "tp")
    elif "fc" in k and "kernel" in k:
        if "fc2" in k:
            # FFN o project
            _spec_tuple = ("tp", None)
        else:
            # FFN in projects
            _spec_tuple = (None, "tp")
    elif "shared" in k or "embed_positions" in k:
        # shared embeddings
        _spec_tuple = ("tp", None)
    elif "final_logits_bias" in k:
        # lm head embeddings
        _spec_tuple = (None, "tp")
    else:
        # usually layer norm or bias
        if not v.ndim == 1:
            raise ValueError(f"Unexpected 1D tensor {k}")
        _spec_tuple = (None,)
    assert len(_spec_tuple) == v.ndim
    p = shd.PartitionSpec(*_spec_tuple)
    return shd.NamedSharding(mesh, p)


def _to_gpt2_sharding(mesh, k, v):
    if not is_ndarray(v) or v.ndim == 0:
        # usually scalars
        return shd.NamedSharding(mesh, shd.PartitionSpec())
    if "c_proj" in k and "kernel" in k:
        _spec_tuple = ("tp", None)
    elif "c_attn" in k and "kernel" in k:
        # q, k, v projects
        _spec_tuple = (None, "tp")
    elif "mlp" in k and "kernel" in k:
        if "c_proj" in k:
            # FFN o project
            _spec_tuple = ("tp", None)
        else:
            # FFN in projects
            _spec_tuple = (None, "tp")
    elif "wpe" in k or "wte" in k:
        # shared embeddings
        _spec_tuple = ("tp", None)
    else:
        # usually layer norm or bias
        if not v.ndim == 1:
            raise ValueError(f"Unexpected 1D tensor {k}")
        _spec_tuple = (None,)
    assert len(_spec_tuple) == v.ndim
    p = shd.PartitionSpec(*_spec_tuple[::-1])
    return shd.NamedSharding(mesh, p)


def _to_gpt_neo_sharding(mesh, k, v):
    if not is_ndarray(v) or v.ndim == 0:
        # usually scalars
        return shd.NamedSharding(mesh, shd.PartitionSpec())
    if "c_proj" in k and "kernel" in k:
        _spec_tuple = ("tp", None)
    elif "attention" in k and "kernel" in k:
        # q, k, v projects
        if "out_proj" in k:
            _spec_tuple = ("tp", None)
        else:
            _spec_tuple = (None, "tp")
    elif "mlp" in k and "kernel" in k:
        if "c_proj" in k:
            # FFN o project
            _spec_tuple = ("tp", None)
        else:
            # FFN in projects
            _spec_tuple = (None, "tp")
    elif "wpe" in k or "wte" in k:
        # shared embeddings
        _spec_tuple = ("tp", None)
    else:
        # usually layer norm or bias
        if not v.ndim == 1:
            raise ValueError(f"Unexpected 1D tensor {k}")
        _spec_tuple = (None,)
    assert len(_spec_tuple) == v.ndim
    p = shd.PartitionSpec(*_spec_tuple[::-1])
    return shd.NamedSharding(mesh, p)


def get_params_sharding(mesh, pytree, model_name="t5"):
    if "t5" in model_name:
        _to_sharding = _to_t5_sharding
    elif "bart" in model_name:
        _to_sharding = _to_bart_sharding
    elif "gpt2" in model_name:
        _to_sharding = _to_gpt2_sharding
    elif "gpt-neo" in model_name:
        _to_sharding = _to_gpt_neo_sharding
    else:
        logger.warning(f"Sharding for {model_name} is not implemented yet.")

        def _to_sharding(mesh, k, v):
            return shd.NamedSharding(mesh, shd.PartitionSpec())

    flat = traverse_util.flatten_dict(pytree, sep=".")
    for k, v in flat.items():
        flat[k] = _to_sharding(mesh, k, v)
    return traverse_util.unflatten_dict(flat, sep=".")


def get_current_sharding(mesh, tree: Any) -> Any:
    """Extracts a PartitionSpec tree from a PyTree containing ``Partitioned`` values."""

    def f(x):
        if hasattr(x, "sharding") and isinstance(x.sharding, shd.NamedSharding):
            return shd.NamedSharding(mesh, x.sharding.spec)
        elif hasattr(x, "shape"):
            return shd.NamedSharding(mesh, shd.PartitionSpec())
        else:
            return None

    return jax.tree_map(f, tree)
