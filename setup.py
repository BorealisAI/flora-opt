# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="flora-opt",
    version="0.0.3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yongchang Hao",
    packages=find_packages(include=["flora_opt", "flora_opt.*"]),
    extras_require={
        "torch": ["torch", "bitsandbytes", "accelerate"],
        "jax": ["optax", "flax", "jax"],
        "testing": ["pytest", "pytest-cov"],
    },
)
