# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax import adafactor, adamw, lion, sgd
from optax._src import utils as optax_utils

from flora_opt.optimizers.optax.flora import flora
from flora_opt.optimizers.optax.utils import (
    NaiveDecomposition,
    TwoSideRandomDecomposition,
    random_split_like_tree,
)


def random_generate(key: chex.PRNGKey, shape: chex.Shape, dtype: chex.ArrayDType = None):
    return jax.random.normal(key, shape, dtype=dtype) / jnp.sqrt(min(shape))


class GradAccState(NamedTuple):
    decomposition: chex.ArrayTree
    rng: chex.PRNGKey


def compressed_acc(
    steps: int = 1,
    factored: bool = True,
    tau: int = 128,
    side: str = "auto",
    min_dim_size_to_factor: int = 128,
):
    def should_factorize(params):
        if factored is False:
            return False
        if params.ndim != 2:
            return False
        if max(params.shape) > min(params.shape) * 16:
            # filter out embeddings
            return False
        return min(params.shape) >= min_dim_size_to_factor
        return True

    mu_dtype = optax_utils.canonicalize_dtype(jnp.float32)

    def accumulate_init(state):
        prngkey_tree = random_split_like_tree(state.dropout_rng, state.params)

        def _fn(p, k):
            if should_factorize(p):
                shape = np.array(p.shape)
                ind, outd = shape
                l_key, r_key = jax.random.split(k, 2)
                otau = itau = tau
                if side == "auto":
                    _side = "right" if outd > ind else "left"
                else:
                    _side = side
                return TwoSideRandomDecomposition(
                    r_data=jnp.zeros((outd, otau), dtype=mu_dtype) if _side != "left" else None,
                    r_proj=r_key if _side != "left" else None,
                    l_data=jnp.zeros((itau, ind), dtype=mu_dtype) if _side != "right" else None,
                    l_proj=l_key if _side != "right" else None,
                )
            else:
                return NaiveDecomposition(
                    data=jnp.zeros_like(p),
                )

        return GradAccState(
            decomposition=jax.tree_map(_fn, state.params, prngkey_tree),
            rng=None,
        )

    def accumulate_update(acc_grads, grads, ministep):
        grads = jax.tree_map(lambda x: x, grads)

        def _naive_fn(grad, dcomp):
            return NaiveDecomposition(data=dcomp.data + (grad - dcomp.data) / (ministep + 1))

        def _full_layer_fn(grad, dcomp):
            if not should_factorize(grad):
                return _naive_fn(grad, dcomp)
            shape = np.array(grad.shape)
            ind, outd = shape
            otau = itau = tau

            if side == "auto":
                _side = "right" if outd > ind else "left"
            else:
                _side = side

            def l_proj_fn(m):
                if _side == "right":
                    return None
                return random_generate(dcomp.l_proj, (outd, itau), mu_dtype).T @ m.T

            def r_proj_fn(m):
                if _side == "left":
                    return None
                return m.T @ random_generate(dcomp.r_proj, (otau, ind), mu_dtype).T

            l_data = dcomp.l_data if _side != "right" else None
            r_data = dcomp.r_data if _side != "left" else None

            return TwoSideRandomDecomposition(
                l_data=(
                    (l_data + (l_proj_fn(grad) - l_data) / (ministep + 1)).astype(mu_dtype)
                    if _side != "right"
                    else None
                ),
                l_proj=dcomp.l_proj if _side != "right" else None,
                r_data=(
                    (r_data + (r_proj_fn(grad) - r_data) / (ministep + 1)).astype(mu_dtype) if _side != "left" else None
                ),
                r_proj=dcomp.r_proj if _side != "left" else None,
            )

        decomposition = jax.tree_map(_full_layer_fn, grads, acc_grads.decomposition)

        return GradAccState(
            decomposition=decomposition,
            rng=acc_grads.rng,
        )

    def accumulate_build(acc_grads, params):
        def _naive_fn(dcomp, p):
            return dcomp.data

        def _layer_fn(dcomp, p):
            if isinstance(dcomp, NaiveDecomposition):
                return _naive_fn(dcomp, p)

            shape = np.array(p.shape)
            ind, outd = shape
            otau = itau = tau

            if side == "auto":
                _side = "right" if outd > ind else "left"
            else:
                _side = side

            r_proj = random_generate(dcomp.r_proj, (otau, ind), mu_dtype) if _side != "left" else None
            l_proj = random_generate(dcomp.l_proj, (outd, itau), mu_dtype) if _side != "right" else None

            if _side == "left":
                return (l_proj @ dcomp.l_data).T
            if _side == "right":
                return (dcomp.r_data @ r_proj).T

            return (dcomp.r_data @ r_proj + l_proj @ dcomp.l_data).T / 2

        return jax.tree_map(
            _layer_fn,
            acc_grads.decomposition,
            params,
            is_leaf=lambda x: isinstance(x, (NaiveDecomposition, TwoSideRandomDecomposition)),
        )

    return accumulate_init, accumulate_update, accumulate_build


def loss_fn_arg(optimizer_cls):
    def wrapper(*args, **kwargs):
        optimizer = optimizer_cls(*args, **kwargs)

        def init_fn(params):
            return optimizer.init(params)

        def update_fn(loss_fn, state, params):
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_state = optimizer.update(grads, state, params)
            return (loss, aux), updates, new_state

        return optax.GradientTransformation(init_fn, update_fn)

    return wrapper


__optimizers__ = {
    "adamw": adamw,
    "adafactor": adafactor,
    "sgd": sgd,
    "lion": lion,
    "flora": flora,
}

__ALL__ = __optimizers__.keys()


def get_optimizer(name, **kwargs):
    optimizer_cls = __optimizers__.get(name)

    return optimizer_cls(**kwargs)
