# Copyright (c) OpenMMLab. All rights reserved.
# from mmengine.dist import get_dist_info, all_reduce
from collections import OrderedDict
from typing import Generator, List
from unittest.mock import MagicMock, Mock

import torch
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)

from mmengine.registry import HOOKS
from .hook import Hook

# TODO, replace with import mmengine.dist as dist
dist = Mock()
dist.IS_DIST = MagicMock(return_value=True)

# TODO, replace with mmengine.dist.get_dist_info
get_dist_info = MagicMock(return_value=(0, 1))
# TODO, replace with mmengine.dist.all_reduce
all_reduce = MagicMock()


# TODO, may need to move to dist.utils after implementing dist module
def _allreduce_coalesced(tensors: List[torch.Tensor],
                         world_size: int,
                         bucket_size_mb: int = -1) -> None:
    """All-reduce a sequence of tensors as a whole.

    Args:
        tensors (List[torch.Tensor]): A sequence of tensors to be
            all-reduced.
        world_size (int): The world size of the process group.
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
        all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
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
    _, world_size = get_dist_info()
    if world_size == 1:
        return
    params_data = [param.data for param in params]
    if coalesce:
        _allreduce_coalesced(params_data, world_size, bucket_size_mb)
    else:
        for tensor in params_data:
            all_reduce(tensor.div_(world_size))


@HOOKS.register_module()
class SyncBuffersHook(Hook):
    """Synchronize model buffers such as running_mean and running_var in BN at
    the end of each epoch."""

    def __init__(self) -> None:
        self.distributed = dist.IS_DIST

    def after_epoch(self, runner: object) -> None:
        """All-reduce model buffers at the end of each epoch.

        Args:
            runner (object): The runner of the training process.
        """
        if self.distributed:
            allreduce_params(runner.model.buffers())  # type: ignore
