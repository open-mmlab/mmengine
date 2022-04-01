# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Any, Sequence, Tuple

import numpy as np
import torch

from .base_data_element import BaseDataElement

DATA_BATCH = Sequence[Tuple[Any, BaseDataElement]]


def worker_init_fn(worker_id: int, num_workers: int, rank: int,
                   seed: int) -> None:
    """This function will be called on each worker subprocess after seeding and
    before data loading.

    Args:
        worker_id (int): Worker id in [0, num_workers - 1].
        num_workers (int): How many subprocesses to use for data loading.
        rank (int): Rank of process in distributed environment. If in
            non-distributed environment, it is a constant number `0`.
        seed (int): Random seed.
    """
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def pseudo_collate(data_batch: DATA_BATCH) -> DATA_BATCH:
    """The default behavior of dataloader is to merge a list of samples to form
    a mini-batch of Tensor(s). However, in MMEngine, ``pseudo_collate`` does
    nothing just returns ``data_batch``.

    Args:
        data_batch (Sequence[Tuple[Any, BaseDataElement]]): Batch of data from
            dataloader.

    Returns:
        Sequence[Tuple[Any, BaseDataElement]]: Return input ``data_batch``.
    """
    return data_batch
