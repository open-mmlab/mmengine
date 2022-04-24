# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Generator, List

import torch
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)

from mmengine import dist
from mmengine.registry import HOOKS
from .hook import Hook


def _allreduce_coalesced(tensors: List[torch.Tensor],
                         bucket_size_mb: int = -1) -> None:
    """All-reduce a sequence of tensors as a whole.

    Args:
        tensors (List[torch.Tensor]): A sequence of tensors to be
            all-reduced.
        bucket_size_mb (int): The limit of each chunk in megabytes
            for grouping tensors into chunks. Defaults to -1.
    """
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors, op='mean')
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)


def allreduce_params(params: Generator[torch.Tensor, None, None],
                     coalesce: bool = True,
                     bucket_size_mb: int = -1) -> None:
    """All-reduce parameters.

    Args:
        params (Generator[torch.Tensor, None, None]): List of parameters or
            buffers of a model.
        coalesce (bool, optional): Whether to reduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    params_data = [param.data for param in params]
    if coalesce:
        _allreduce_coalesced(params_data, bucket_size_mb)
    else:
        for tensor in params_data:
            dist.all_reduce(tensor, op='mean')


@HOOKS.register_module()
class SyncBuffersHook(Hook):
    """Synchronize model buffers such as running_mean and running_var in BN at
    the end of each epoch."""

    priority = 'NORMAL'

    def __init__(self) -> None:
        self.distributed = dist.is_distributed()

    def after_train_epoch(self, runner) -> None:
        """All-reduce model buffers at the end of each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.distributed:
            allreduce_params(runner.model.buffers())
