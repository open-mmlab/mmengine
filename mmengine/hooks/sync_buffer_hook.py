# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dist import all_reduce_params, is_distributed
from mmengine.registry import HOOKS
from .hook import Hook


@HOOKS.register_module()
class SyncBuffersHook(Hook):
    """Synchronize model buffers such as running_mean and running_var in BN at
    the end of each epoch."""

    priority = 'NORMAL'

    def __init__(self) -> None:
        self.distributed = is_distributed()

    def after_train_epoch(self, runner) -> None:
        """All-reduce model buffers at the end of each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.distributed:
            all_reduce_params(runner.model.buffers(), op='mean')
