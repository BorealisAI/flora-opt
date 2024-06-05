# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2021, The HuggingFace Team
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on Huggingface's example from https://github.com/huggingface/transformers/blob/main/examples/flax/summarization/run_summarization_flax.py by HuggingFace  # noqa
####################################################################################


import math
from typing import Callable, Generator, Optional

import jax
import numpy as np
from datasets import Dataset


def data_loader(
    rng: jax.random.PRNGKey,
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    transform_fn: Optional[Callable] = None,
) -> Generator[dict, None, None]:
    """
    Returns batches of size `batch_size` from `dataset`. If `drop_last` is set to `False`, the final batch may be incomplete, and range in size from 1 to `batch_size`. Shuffle batches if `shuffle` is `True`.
    """  # noqa
    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
        batch_idx = np.asarray(batch_idx)
    else:
        batch_idx = np.arange(len(dataset))

    if drop_last:
        steps_per_epoch = len(dataset) // batch_size
        batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
        batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))
    else:
        steps_per_epoch = math.ceil(len(dataset) / batch_size)
        batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        if transform_fn is not None:
            batch = transform_fn(batch)
        batch = {k: np.array(v) for k, v in batch.items()}

        yield batch
