# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmengine.registry import HOOKS
from .hook import Hook

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class ParamSchedulerHook(Hook):
    """A hook to update some hyper-parameters in optimizer, e.g., learning rate
    and momentum."""

    priority = 'LOW'

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Call step function for each scheduler after each iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                In order to keep this interface consistent with other hooks,
                we keep ``data_batch`` here. Defaults to None.
            outputs (dict, optional): Outputs from model.
                In order to keep this interface consistent with other hooks, we
                keep ``data_batch`` here. Defaults to None.
        """

        def step(param_schedulers):
            assert isinstance(param_schedulers, list)
            for scheduler in param_schedulers:
                if not scheduler.by_epoch:
                    scheduler.step()

        if runner.param_schedulers is None:
            return

        if isinstance(runner.param_schedulers, list):
            step(runner.param_schedulers)
        elif isinstance(runner.param_schedulers, dict):
            for param_schedulers in runner.param_schedulers.values():
                step(param_schedulers)
        else:
            raise TypeError(
                'runner.param_schedulers should be list of ParamScheduler or '
                'a dict containing list of ParamScheduler, '
                f'but got {runner.param_schedulers}')

    def after_train_epoch(self, runner) -> None:
        """Call step function for each scheduler after each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """

        def step(param_schedulers):
            assert isinstance(param_schedulers, list)
            for scheduler in param_schedulers:
                if scheduler.by_epoch:
                    scheduler.step()

        if runner.param_schedulers is None:
            return

        if isinstance(runner.param_schedulers, list):
            step(runner.param_schedulers)
        elif isinstance(runner.param_schedulers, dict):
            for param_schedulers in runner.param_schedulers.values():
                step(param_schedulers)
        else:
            raise TypeError(
                'runner.param_schedulers should be list of ParamScheduler or '
                'a dict containing list of ParamScheduler, '
                f'but got {runner.param_schedulers}')
