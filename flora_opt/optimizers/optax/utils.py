# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import base, numerics, transform, utils
from optax._src.factorized import (
    FactoredState,
    _decay_rate_pow,
    _factored_dims,
    _UpdateResult,
)


class Decomposition:
    pass


ScalarOrSchedule = Union[float, jax.Array, base.Schedule]
MaskOrFn = Optional[Union[Any, Callable[[base.Params], Any]]]


def scale_by_learning_rate(learning_rate: ScalarOrSchedule, flip_sign=True):
    m = -1 if flip_sign else 1
    if callable(learning_rate):
        return transform.scale_by_schedule(lambda count: m * learning_rate(count))
    return transform.scale(m * learning_rate)


@partial(jax.jit, static_argnums=(1, 2))
def random_normal(key, shape, dtype):
    return jax.random.normal(key, shape, dtype=dtype)


@partial(jax.jit, static_argnums=(1, 2))
def random_orthogonal(key, shape, dtype):
    reversed = shape[0] < shape[1]
    z = jax.random.normal(key, shape, dtype)
    orth = jnp.linalg.qr(z.T if reversed else z)[0]
    return orth.T if reversed else orth


@jax.jit
def next_rng_key(key):
    return jax.random.split(key)[0]


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jax.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_unflatten(treedef, keys)


def diag_pinv(a):
    max_rows_cols = a.shape[-1]
    rcond = 10.0 * max_rows_cols * jnp.array(jnp.finfo(a.dtype).eps)
    cutoff = rcond * jnp.abs(a).max()
    a = jnp.where(a > cutoff, a, jnp.inf).astype(a.dtype)
    a = jnp.reciprocal(a)

    return a


def bias_correction(decay, count, vector):
    return jax.tree_map(lambda e: (e / (1 - decay**count)).astype(e.dtype), vector)


class EigenDecomposition(NamedTuple):
    basis: chex.Array  # shape=(d, tau), dtype=jnp.floating.
    sigma: chex.Array  # shape=(tau,), dtype=jnp.floating.


class NaiveDecomposition(NamedTuple):
    data: chex.Array


class RandomDecomposition(NamedTuple):
    """
    Randomized factorization.
    The original matrix is decomposed as:
        data @ proj

    :param data: (d2, tau)
    :param proj: (tau, d1)
    """

    data: chex.Array
    proj: Union[chex.PRNGKey, chex.Array]


class TwoSideRandomDecomposition(NamedTuple):
    l_data: chex.Array
    l_proj: Union[chex.PRNGKey, chex.Array]
    r_data: chex.Array
    r_proj: Union[chex.PRNGKey, chex.Array]


def scale_by_sign():
    def init_fn(params):
        return base.EmptyState()

    def update_fn(grads, state, params=None, *args, **kwargs):
        return jax.tree_map(jnp.sign, grads), state

    return base.GradientTransformation(init_fn, update_fn)


def scale_by_factored_rms(
    factored: bool = True,
    decay_rate: float = 0.8,
    step_offset: int = 0,
    min_dim_size_to_factor: int = 128,
    epsilon: float = 1e-30,
    decay_rate_fn: Callable[[int, float], chex.Array] = _decay_rate_pow,
):
    def _to_state(count: chex.Array, result_tree):
        """Maps from a tree of (factored) values to separate trees of values."""
        return FactoredState(
            count=count,
            v_row=jax.tree_util.tree_map(lambda o: o.v_row, result_tree),
            v_col=jax.tree_util.tree_map(lambda o: o.v_col, result_tree),
            v=jax.tree_util.tree_map(lambda o: o.v, result_tree),
        )

    def init_fn(params):
        """Initialise the optimiser's state."""

        def _init(param):
            shape = param.shape
            factored_dims = _factored_dims(shape, factored, min_dim_size_to_factor)
            if factored_dims is not None:
                d1, d0 = factored_dims
                vr_shape = np.delete(shape, d0)
                vc_shape = np.delete(shape, d1)
                return _UpdateResult(
                    update=jnp.zeros((1,)), v_row=jnp.zeros(vr_shape), v_col=jnp.zeros(vc_shape), v=jnp.zeros((1,))
                )
            else:
                return _UpdateResult(
                    update=jnp.zeros((1,)), v_row=jnp.zeros((1,)), v_col=jnp.zeros((1,)), v=jnp.zeros(param.shape)
                )

        return _to_state(jnp.zeros([], jnp.int32), jax.tree_util.tree_map(_init, params))

    def update_fn(grads, state, params, updates, query_only=False, **kwargs):
        """Apply gradient transformation."""
        if params is None:
            raise ValueError(base.NO_PARAMS_MSG)

        def _update(grad, update, v_row, v_col, v, param, step):
            shape = param.shape
            decay_rate_t = decay_rate_fn(step - step_offset, decay_rate)

            # Scaled by factorized second moment statistics.
            new_v_row = jnp.zeros((1,))
            new_v_col = jnp.zeros((1,))
            new_v = jnp.zeros((1,))

            factored_dims = _factored_dims(shape, factored, min_dim_size_to_factor)
            if factored_dims is not None:
                d1, d0 = factored_dims
                if query_only:
                    new_v_row = v_row
                    new_v_col = v_col
                else:
                    grad_sqr = numerics.abs_sq(grad) + epsilon
                    new_v_row = decay_rate_t * v_row + (1.0 - decay_rate_t) * jnp.mean(grad_sqr, axis=d0)
                    new_v_col = decay_rate_t * v_col + (1.0 - decay_rate_t) * jnp.mean(grad_sqr, axis=d1)
                reduced_d1 = d1 - 1 if d1 > d0 else d1
                row_col_mean = jnp.mean(new_v_row, axis=reduced_d1, keepdims=True)
                row_factor = (new_v_row / row_col_mean) ** -0.5
                col_factor = (new_v_col) ** -0.5
                update = update * jnp.expand_dims(row_factor, axis=d0) * jnp.expand_dims(col_factor, axis=d1)
            else:
                if query_only:
                    new_v = v
                else:
                    grad_sqr = numerics.abs_sq(grad) + epsilon
                    new_v = decay_rate_t * v + (1.0 - decay_rate_t) * grad_sqr
                update = update * (new_v) ** -0.5

            return _UpdateResult(update, new_v_row, new_v_col, new_v)

        # Transform grad and compute new per-parameter stats.
        output = jax.tree_util.tree_map(
            lambda *args: _update(*args, state.count), grads, updates, state.v_row, state.v_col, state.v, params
        )

        # Unpack updates / stats and return.
        updates = jax.tree_util.tree_map(lambda o: o.update, output)
        return updates, _to_state(utils.safe_int32_increment(state.count), output)

    return base.GradientTransformation(init_fn, update_fn)
