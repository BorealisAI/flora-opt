# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


try:
    from flora_opt.optimizers.optax.flora import flora as optax_flora
except ImportError:
    optax_flora = None

try:
    from flora_opt.optimizers.torch.flora import Flora, FloraAccelerator
except ImportError:
    Flora = None
    FloraAccelerator = None


__all__ = {
    "optax_flora": optax_flora,
    "Flora": Flora,
    "FloraAccelerator": FloraAccelerator,
}
