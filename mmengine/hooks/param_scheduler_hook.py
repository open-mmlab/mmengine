# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Sequence, Tuple

from mmengine.data import BaseDataElement
from mmengine.registry import HOOKS
from .hook import Hook

DATA_BATCH = Optional[Sequence[Tuple[Any, BaseDataElement]]]


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
            data_batch (Sequence[Tuple[Any, BaseDataElement]], optional): Data
                from dataloader. In order to keep this interface consistent
                with other hooks, we keep ``data_batch`` here.
                Defaults to None.
            outputs (dict, optional): Outputs from model.
                In order to keep this interface consistent with other hooks, we
                keep ``data_batch`` here. Defaults to None.
        """
        for scheduler in runner.param_schedulers:  # type: ignore
            if not scheduler.by_epoch:
                scheduler.step()

    def after_train_epoch(self, runner) -> None:
        """Call step function for each scheduler after each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        for scheduler in runner.param_schedulers:  # type: ignore
            if scheduler.by_epoch:
                scheduler.step()
