# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmengine.data import BaseDataSample
from mmengine.registry import HOOKS
from .hook import Hook


@HOOKS.register_module()
class ParamSchedulerHook(Hook):
    """A hook to update some hyper-parameters in optimizer, e.g learning rate
    and momentum."""

    def after_iter(self,
                   runner: object,
                   data_batch: Optional[Sequence[BaseDataSample]] = None,
                   outputs: Optional[Sequence[BaseDataSample]] = None) -> None:
        """Call step function for each scheduler after each iteration.

        Args:
            runner (object): The runner of the training process.
            data_batch (Sequence[BaseDataSample]): Data from dataloader. In
                order to keep this interface consistent with other hooks, we
                keep ``data_batch`` here. Defaults to None.
            outputs (Sequence[BaseDataSample]): Outputs from model. In
                order to keep this interface consistent with other hooks, we
                keep ``data_batch`` here. Defaults to None.
        """
        for scheduler in runner.schedulers:  # type: ignore
            if not scheduler.by_epoch:
                scheduler.step()

    def after_epoch(self, runner: object) -> None:
        """Call step function for each scheduler after each epoch.

        Args:
            runner (object): The runner of the training process.
        """
        for scheduler in runner.schedulers:  # type: ignore
            if scheduler.by_epoch:
                scheduler.step()
