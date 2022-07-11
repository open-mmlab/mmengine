# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate as _default_collate

from mmengine.registry import Registry

DATA_BATCH = Sequence[dict]
COLLATE_FUNCTIONS = Registry('Collate Functions')


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


@COLLATE_FUNCTIONS.register_module()
def pseudo_collate(data_batch: DATA_BATCH) -> dict:
    """The default behavior of dataloader is to merge a list of samples to form
    a mini-batch of Tensor(s). However, in MMEngine, ``pseudo_collate`` does
    nothing just returns ``data_batch``.

    Args:
        data_batch (Sequence[dict]): Batch of data from
            dataloader.

    Returns:
        Sequence[dict]: Return input ``data_batch``.
    """
    collated_data: Dict[str, Optional[list]] = {
        key: [d[key] for d in data_batch]
        for key in data_batch[0]
    }
    collated_data['data_sample'] = collated_data.get('data_sample', None)
    return collated_data


@COLLATE_FUNCTIONS.register_module()
def default_collate(data_batch: DATA_BATCH) -> dict:
    """Stack ``inputs`` in ``data_batch`` into a batched tensor with the first
    dimension batch size, and then move input tensor to the target device.

    Different from ``default_collate`` in pytorch, ``default_collate`` will
    only stack ``inputs`` tensor in ``data_batch``.

    Note:
        ``default_collate`` only accept input tensor with the same shape.

    Args:
        data_batch (Sequence[dict]): Data sampled from dataset.

    Returns:
        Tuple[torch.Tensor, Optional[list]]: Data in the same format as the
        model input.
    """
    collated_data: Dict[str, Optional[list]] = {
        key: [d[key] for d in data_batch]
        for key in data_batch[0]
    }
    collated_data['inputs'] = _default_collate(collated_data['inputs'])
    collated_data['data_sample'] = collated_data.get('data_sample', None)
    return collated_data
