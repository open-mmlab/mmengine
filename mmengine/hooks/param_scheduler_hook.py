# Copyright (c) OpenMMLab. All rights reserved.
from .hook import HOOKS, Hook


@HOOKS.register_module()
class ParamSchedulerHook(Hook):
    """A hook to update some hyper-parameters in optimizer, e.g learning rate
    and momentum."""

    def after_iter(self, runner, data_batch=None, outputs=None) -> None:
        for scheduler in runner.schedulers:
            if not scheduler.by_epoch:
                scheduler.step()

    def after_epoch(self, runner, data_batch=None, outputs=None) -> None:
        for scheduler in runner.schedulers:
            if scheduler.by_epoch:
                scheduler.step()
