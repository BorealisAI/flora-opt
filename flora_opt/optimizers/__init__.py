# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging

logger = logging.getLogger(__name__.split(".")[0])

try:
    from flora_opt.optimizers.optax import flora as optax_flora
except ImportError as e:
    logger.info(f"JAX dependencies are not installed. optax_flora is disabled. Details: {e}")
    optax_flora = None

try:
    from flora_opt.optimizers.torch.flora import Flora, FloraAccelerator
except ImportError as e:
    logger.info(f"PyTorch dependencies are not installed. Flora and FloraAccelerator are disabled. Details: {e}")
    Flora = None
    FloraAccelerator = None


__all__ = {
    "optax_flora": optax_flora,
    "Flora": Flora,
    "FloraAccelerator": FloraAccelerator,
}
