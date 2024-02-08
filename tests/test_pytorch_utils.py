# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch

from flora_opt.optimizers.torch.flora import next_seed, split_seed, stable_randn


def test_stable_randn_consistent():
    results = {}

    for seed in range(10):
        shape = [2] * (seed + 1)

        results[seed] = stable_randn(shape, seed)

    for seed in range(10):
        shape = [2] * (seed + 1)

        assert torch.allclose(results[seed], stable_randn(shape, seed), atol=1e-6)

    for seed in reversed(range(10)):
        shape = [2] * (seed + 1)

        assert torch.allclose(results[seed], stable_randn(shape, seed), atol=1e-6)


def test_stable_randn_randomness():
    results = {}
    shape = (512, 512)
    for seed in range(10):
        results[seed] = stable_randn(shape, seed)
        for prev_seed in range(seed):
            assert not torch.allclose(results[seed], stable_randn(shape, prev_seed), atol=1e-6)
            assert torch.allclose(stable_randn(shape, prev_seed), results[prev_seed], atol=1e-6)


def test_next_seed_spread():
    seed = 0
    hist = [seed]
    for i in range(10):
        seed = next_seed(seed)
        hist.append(seed)

    assert len(set(hist)) == len(hist)


def test_next_seed_consistency():
    results = {}
    for seed in range(100):
        results[seed] = next_seed(seed)
        assert next_seed(seed) == next_seed(seed)
        assert next_seed(seed) == results[seed]

    for seed in range(100):
        assert results[seed] == next_seed(seed)


def test_split_seed():
    for seed in range(100):
        seeds = split_seed(seed)
        assert len(seeds) == 2
        assert not seeds[0] == seed
        assert not seeds[1] == seed
        assert not seeds[0] == seeds[1]
