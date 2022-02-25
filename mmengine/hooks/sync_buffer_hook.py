# Copyright (c) OpenMMLab. All rights reserved.
# from mmengine.dist import get_dist_info, all_reduce
from unittest.mock import MagicMock
from .hook import HOOKS, Hook
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)
from collections import OrderedDict

# TODO, need to remove those lines after implementing dist module
get_dist_info = MagicMock(return_value=(0, 1))
all_reduce = MagicMock()


# TODO, may need to move to dist.utils after implementing dist module
def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
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


def allreduce_params(params, coalesce=True, bucket_size_mb=-1):
    """Allreduce parameters.

    Args:
        params (list[torch.Parameters]): List of parameters or buffers of a
            model.
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """
    _, world_size = get_dist_info()
    if world_size == 1:
        return
    params = [param.data for param in params]
    if coalesce:
        _allreduce_coalesced(params, world_size, bucket_size_mb)
    else:
        for tensor in params:
            all_reduce(tensor.div_(world_size))


@HOOKS.register_module()
class SyncBuffersHook(Hook):
    """Synchronize model buffers such as running_mean and running_var in BN at
    the end of each epoch.

    Args:
        distributed (bool): Whether distributed training is used. It is
          effective only for distributed training. Defaults to True.
    """

    def __init__(self, distributed=True):
        self.distributed = distributed

    def after_epoch(self, runner):
        """All-reduce model buffers at the end of each epoch."""
        if self.distributed:
            allreduce_params(runner.model.buffers())
