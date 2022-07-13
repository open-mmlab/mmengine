# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data._utils.collate import \
    default_collate as torch_default_collate

from mmengine.registry import Registry
from ..data import BaseDataElement

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
def pseudo_collate(data_batch: Sequence) -> Any:
    """Convert list of data sampled from dataset into a batch of data, of which
    type consistent with the type of each element in ``data_batch``.

    The default behavior of dataloader is to merge a list of samples to form
    a mini-batch of Tensor(s). However, in MMEngine, ``pseudo_collate``
    will not stack tensors to batch tensors, and convert int, float, ndarray to
    tensors.

    This code is referenced from:
    `Pytorch default_collate <https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py>`_.  # noqa: E501
    Args:
        data_batch (Sequence[T]): Batch of data from dataloader.

    Returns:
        Sequence[T]: Transversed Data in the same format as the element of
        ``data_batch``.
    """
    elem = data_batch[0]
    elem_type = type(elem)
    if isinstance(elem, (str, bytes)):
        return data_batch
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(pseudo_collate(samples)
                           for samples in zip(*data_batch)))
    elif isinstance(elem, Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(data_batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                'each element in list of batch should be of equal size')
        transposed = list(zip(*data_batch))

        if isinstance(elem, tuple):
            return [pseudo_collate(samples)
                    for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type(
                    [pseudo_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)`
                # (e.g., `range`).
                return [pseudo_collate(samples) for samples in transposed]
    elif isinstance(elem, Mapping):
        return elem_type({
            key: pseudo_collate([d[key] for d in data_batch])
            for key in elem
        })
    else:
        return data_batch


@COLLATE_FUNCTIONS.register_module()
def default_collate(data_batch: Sequence) -> Any:
    """Convert list of data sampled from dataset into a batch of data, of which
    type consistent with the type of each element in ``data_batch``.

    Different from :func:`pseudo_collate`, ``default_collate`` will stack
    tensor contained in ``data_batch`` into a batched tensor with the
    first dimension batch size, and then move input tensor to the target
    device.

    Different from ``default_collate`` in pytorch, ``default_collate`` will
    not process ``BaseDataElement``.

    Note:
        ``default_collate`` only accept input tensor with the same shape.

    Args:
        data_batch (Sequence[T]): Data sampled from dataset.

    Returns:
        T: Data in the same format as the element of ``data_batch``, of which
        tensors have been stacked, and ndarray, int, float have been
        converted to tensors.
    """
    elem = data_batch[0]
    elem_type = type(elem)
    if isinstance(elem, BaseDataElement):
        return data_batch
    elif isinstance(elem, (str, bytes)):
        return data_batch
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples)
                           for samples in zip(*data_batch)))
    elif isinstance(elem, Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(data_batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                'each element in list of batch should be of equal size')
        transposed = list(zip(*data_batch))

        if isinstance(elem, tuple):
            return [default_collate(samples)
                    for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type(
                    [default_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)`
                # (e.g., `range`).
                return [default_collate(samples) for samples in transposed]
    elif isinstance(elem, Mapping):
        return elem_type({
            key: default_collate([d[key] for d in data_batch])
            for key in elem
        })
    else:
        return torch_default_collate(data_batch)
