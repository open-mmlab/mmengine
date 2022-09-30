# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmengine.dist import is_main_process
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

try:
    from torch.distributed.optim import \
        ZeroRedundancyOptimizer as _ZeroReundancyOptimizer
except ImportError as e:
    print(e)
    _ZeroReundancyOptimizer = object

from .builder import OPTIMIZERS


@OPTIMIZERS.register_module()
class ZeroRedundancyOptimizer(_ZeroReundancyOptimizer):
    """A wrapper class of :class:`ZeroRedundancyOptimizer` that gets a
    optimizer type as string. This class wraps an arbitrary
    :class:`optim.Optimizer.

    <torch.optim.Optimizer>` and shards its states across ranks in the group as
    described by ZeRO_. The local optimizer instance in each rank is only
    responsible for updating approximately ``1 / world_size`` parameters and
    hence only needs to keep ``1 / world_size`` optimizer states. After
    parameters are updated locally, each rank will broadcast its parameters to
    all other peers to keep all model replicas in the same state.
    ``ZeroRedundancyOptimizer`` can be used in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel` to reduce per-rank peak
    memory consumption.
    ``ZeroRedundancyOptimizer`` uses a sorted-greedy algorithm to pack a number
    of parameters at each rank. Each parameter belongs to a single rank and is
    not divided among ranks. The partition is arbitrary and might not match the
    the parameter registration or usage order.
    Warnings:
        ``ZeroRedundancyOptimizer`` requires PyTorch >= 1.8.
    Args:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_type (str): the string of the local optimizer class.
    """

    def __init__(self, params, optimizer_type: str, **kwargs):
        assert digit_version(TORCH_VERSION) >= digit_version('1.8.0'), (
            '`torch.distributed.optim.ZeroReundancyOptimizer` is only '
            'available when pytorch version >= 1.8')
        optimizer_class = getattr(torch.optim, optimizer_type)
        super().__init__(params, optimizer_class, **kwargs)

    def state_dict(self):
        """Consolidate `state_dict`s from ranks to save the `state_dict`"""
        self.consolidate_state_dict()
        if is_main_process():
            return super().state_dict()
