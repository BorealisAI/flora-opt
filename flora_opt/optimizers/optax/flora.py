# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
from typing import Any, NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
import optax
from optax._src import base, clipping, transform, utils

from flora_opt.optimizers.optax.utils import (
    NaiveDecomposition,
    RandomDecomposition,
    next_rng_key,
    random_orthogonal,
    random_split_like_tree,
    scale_by_learning_rate,
)


def random_generate(key: chex.PRNGKey, shape: chex.Shape, dtype: chex.ArrayDType = jnp.float32) -> chex.Array:
    orth = False
    if orth:
        return random_orthogonal(key, shape, dtype)
    else:
        return jax.random.normal(key, shape, dtype=dtype) / jnp.sqrt(min(shape))


@partial(jax.remat, policy=jax.checkpoint_policies.nothing_saveable, static_argnums=(2,))
def down_project(rng: chex.PRNGKey, x: chex.Array, tau: int) -> chex.Array:
    if x.shape[0] < x.shape[-1]:
        proj = random_generate(rng, (tau, x.shape[0]), x.dtype)
        return jnp.dot(proj, x)
    else:
        proj = random_generate(rng, (x.shape[-1], tau), x.dtype)
        return jnp.dot(x, proj)


@partial(jax.remat, policy=jax.checkpoint_policies.nothing_saveable, static_argnums=(2, 3))
def up_project(rng: chex.PRNGKey, x: chex.Array, tau: int, shape: chex.Shape) -> chex.Array:
    if shape[0] < shape[-1]:
        proj = random_generate(rng, (tau, shape[0]), x.dtype)
        return jnp.dot(proj.T, x)
    else:
        proj = random_generate(rng, (shape[-1], tau), x.dtype)
        return jnp.dot(x, proj.T)


def flora(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    tau: int = 4,
    seed: int = 0,
    kappa: int = 1000,
    clipping_threshold: Optional[float] = 1.0,
    multiply_by_parameter_scale: bool = True,
    weight_decay: Optional[optax.ScalarOrSchedule] = None,
    eps: float = 1e-30,
    min_dim_size_to_factor: int = 128,
    factorized_second_moment: bool = True,
) -> optax.GradientTransformation:
    tx = [
        scale_by_flora(
            factored=tau is not None,
            beta=b1,
            tau=tau,
            seed=seed,
            kappa=kappa,
            min_dim_size_to_factor=min_dim_size_to_factor,
        )
    ]
    import optax._src.factorized

    tx.append(
        optax._src.factorized.scale_by_factored_rms(
            factored=factorized_second_moment,
            decay_rate=b2,
            epsilon=eps,
            min_dim_size_to_factor=min_dim_size_to_factor,
        )
    )
    tx = tx[::-1]

    if clipping_threshold is not None:
        tx.append(clipping.clip_by_block_rms(clipping_threshold))
    tx.append(scale_by_learning_rate(learning_rate, flip_sign=False))
    if multiply_by_parameter_scale:
        tx.append(transform.scale_by_param_block_rms())
    if weight_decay is not None:
        tx.append(transform.add_decayed_weights(weight_decay))
    tx.append(transform.scale(-1.0))

    init_fns = [t.init for t in tx]
    update_fns = [t.update for t in tx]

    def init_fn(params):
        return tuple(fn(params) for fn in init_fns)

    def update_fn(updates, state, params=None, **extra_args):
        if len(update_fns) != len(state):
            raise ValueError(
                "The number of updates and states has to be the same in " "chain! Make sure you have called init first!"
            )

        new_state = []

        updates, new_s = update_fns[0](updates, state[0], params, **extra_args)
        new_state.append(new_s)

        updates, new_s = update_fns[1](updates, state[1], params, **extra_args)
        new_state.append(new_s)

        for s, fn in zip(state[2:], update_fns[2:]):
            updates, new_s = fn(updates, s, params, **extra_args)
            new_state.append(new_s)
        return updates, tuple(new_state)

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


class ScaleByFloraState(NamedTuple):
    """State for the Flora algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    decomposition: chex.ArrayTree
    rng: chex.PRNGKey


def scale_by_flora(
    factored: bool = True,
    beta: Optional[float] = None,
    tau: int = 4,
    seed: int = 0,
    kappa: int = 1000,
    min_dim_size_to_factor: int = 128,
) -> optax.GradientTransformation:
    def should_factorize(params):
        if factored is False:
            return False
        if params.ndim != 2:
            return False
        if max(params.shape) > min(params.shape) * 4:
            # do not factorize embeddings
            return False
        return min(params.shape) >= min_dim_size_to_factor

    mu_dtype = utils.canonicalize_dtype(jnp.bfloat16)

    def init_fn(params):
        rng = jax.random.PRNGKey(seed)
        prngkey_tree = random_split_like_tree(rng, params)

        def _init_layer(params, key):
            if should_factorize(params):
                if params.shape[0] < params.shape[-1]:
                    data = jnp.zeros((tau, params.shape[-1]), dtype=mu_dtype)
                else:
                    data = jnp.zeros((params.shape[0], tau), dtype=mu_dtype)
                return RandomDecomposition(data=data, proj=key)
            else:
                return NaiveDecomposition(
                    data=jnp.zeros_like(params, dtype=mu_dtype),
                )

        return ScaleByFloraState(
            count=jnp.zeros([], jnp.int32),
            decomposition=jax.tree_map(_init_layer, params, prngkey_tree),
            rng=rng,
        )

    def update_state(grads, state, params=None):
        grads = jax.tree_map(lambda x: x.astype(mu_dtype), grads)

        @partial(jax.remat, policy=jax.checkpoint_policies.nothing_saveable, static_argnums=(0,))
        def _maybe_switch_proj_fn(shape, dcomp):
            flag = jnp.mod(state.count, kappa) == 0
            return RandomDecomposition(
                data=jax.lax.cond(
                    flag,
                    lambda: down_project(dcomp.proj, up_project(dcomp.proj, dcomp.data, tau, shape), tau),
                    lambda: dcomp.data,
                ),
                proj=jax.lax.cond(
                    flag,
                    lambda: next_rng_key(dcomp.proj),
                    lambda: dcomp.proj,
                ),
            )

        def _update_layer_fn(grad, dcomp):
            if not should_factorize(grad):
                return NaiveDecomposition(data=beta * dcomp.data + (1 - beta) * grad)

            dcomp = _maybe_switch_proj_fn(grad.shape, dcomp)
            return RandomDecomposition(
                data=dcomp.data * beta + (1 - beta) * down_project(dcomp.proj, grad, tau),
                proj=dcomp.proj,
            )

        decomposition = jax.tree_map(_update_layer_fn, grads, state.decomposition)

        return ScaleByFloraState(
            count=state.count + 1,
            decomposition=decomposition,
            rng=state.rng,
        )

    def query(grads: Any, state: ScaleByFloraState, params: Any = None):
        def _layer_fn(grad, dcomp):
            if not should_factorize(grad):
                return dcomp.data
            else:
                return up_project(dcomp.proj, dcomp.data, tau, grad.shape)

        return jax.tree_map(_layer_fn, grads, state.decomposition)

    def update_fn(grads: Any, state: ScaleByFloraState, params: Any = None):
        del params

        state = update_state(grads, state)
        updates = query(grads, state)
        updates = jax.tree_map(lambda m, g: m * beta + g * (1 - beta), updates, grads)

        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_interp(
    weight: float = 0.9,
    transform_fn: Optional[callable] = None,
) -> optax.GradientTransformation:
    def init_fn(params):
        return base.EmptyState()

    def update_fn(grads, state, params=None, updates=None):
        updates = jax.tree_map(lambda g, u: (1 - weight) * g + weight * u, grads, updates)
        if transform_fn is not None:
            updates = jax.tree_map(transform_fn, updates)
        return updates, state

    return base.GradientTransformation(init_fn, update_fn)
