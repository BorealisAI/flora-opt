# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from flora_opt.optimizers import optax_flora, Flora, FloraAccelerator


def set_logging_level(level: int):
    logger = logging.getLogger(__name__.split(".")[0])
    logger.setLevel(level)


if Flora is None and optax_flora is None and FloraAccelerator is None:
    raise ImportError("Both PyTorch and JAX dependencies are not installed. Please install one of them.")


__all__ = {
    "optax_flora": optax_flora,
    "Flora": Flora,
    "FloraAccelerator": FloraAccelerator,
}
