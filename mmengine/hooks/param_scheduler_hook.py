# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

from mmengine.optim import ReduceOnPlateauParamScheduler, _ParamScheduler
from mmengine.registry import HOOKS
from mmengine.runner import BaseLoop
from .hook import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]


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
            data_batch (dict or tuple or list, optional): Data from dataloader.
                In order to keep this interface consistent with other hooks,
                we keep ``data_batch`` here.
            outputs (dict, optional): Outputs from model.
                In order to keep this interface consistent with other hooks, we
                keep ``data_batch`` here.
        """

        if runner.param_schedulers is None:
            return

        def step(param_schedulers):
            assert isinstance(param_schedulers, list)
            for scheduler in param_schedulers:
                if not scheduler.by_epoch:
                    scheduler.step()

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

        if runner.param_schedulers is None:
            return

        def step(param_schedulers):
            assert isinstance(param_schedulers, list)
            for scheduler in param_schedulers:
                if scheduler.by_epoch:
                    scheduler.step()

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

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:

        if runner.param_schedulers is None:
            return

        if not self._should_after_val_epoch(runner):
            return

        def step(param_schedulers):
            assert isinstance(param_schedulers, list)
            for scheduler in param_schedulers:
                if isinstance(scheduler, ReduceOnPlateauParamScheduler):
                    if scheduler.by_epoch and (scheduler.monitor_stage
                                               == 'val'):
                        scheduler.metric = metrics[scheduler.monitor_key]
                        scheduler.step()

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

    def _should_after_val_epoch(self, runner) -> bool:
        """Check whether train_loop and param_schedulers is built."""
        self._should: bool

        if hasattr(self, '_should'):
            # to save check time
            return self._should

        # Check train_loop is built
        # Need to skip building train_loop when call runner.train_loop,
        # so use runner._train_loop. This is a hacky approach.
        if not isinstance(runner._train_loop, BaseLoop):
            self._should = False
            return self._should

        # Check param_schedulers is built
        # copy from runner.scale_lr method
        def _is_built(schedulers) -> bool:
            if isinstance(schedulers, dict):
                return False if 'type' in schedulers else any(
                    _is_built(s) for s in schedulers.values())
            if isinstance(schedulers, list):
                return any(_is_built(s) for s in schedulers)
            return isinstance(schedulers, _ParamScheduler)

        self._should = _is_built(runner.param_schedulers)
        return self._should
