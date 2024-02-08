# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import jax
import jax.numpy as jnp

from flora_opt.optimizers.optax.flora import random_generate


def test_stable_randn_consistent():
    with jax.default_device(jax.devices("cpu")[0]):
        results = {}

        for seed in range(10):
            rng = jax.random.PRNGKey(seed)
            shape = [2] * (seed + 1)

            results[seed] = random_generate(rng, shape, jnp.float32)

        for seed in range(10):
            shape = [2] * (seed + 1)

            assert jnp.allclose(results[seed], random_generate(jax.random.PRNGKey(seed), shape, jnp.float32))

        for seed in reversed(range(10)):
            shape = [2] * (seed + 1)

            assert jnp.allclose(results[seed], random_generate(jax.random.PRNGKey(seed), shape, jnp.float32))


def test_stable_randn_randomness():
    with jax.default_device(jax.devices("cpu")[0]):
        results = {}
        shape = (512, 512)
        for seed in range(10):
            rng = jax.random.PRNGKey(seed)
            results[seed] = random_generate(rng, shape, jnp.float32)
            for prev_seed in range(seed):
                assert not jnp.allclose(
                    results[seed], random_generate(jax.random.PRNGKey(prev_seed), shape, jnp.float32)
                )
                assert jnp.allclose(
                    random_generate(jax.random.PRNGKey(prev_seed), shape, jnp.float32), results[prev_seed]
                )
