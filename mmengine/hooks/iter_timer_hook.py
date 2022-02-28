# Copyright (c) OpenMMLab. All rights reserved.
import time
from typing import Optional, Sequence

from mmengine.data import BaseDataSample
from mmengine.registry import HOOKS
from .hook import Hook


@HOOKS.register_module()
class IterTimerHook(Hook):
    """A hook that logs the time spent during iteration.

    Eg. ``data_time`` for loading data and ``time`` for a model train step.
    """

    def before_epoch(self, runner: object) -> None:
        """Record time flag before start a epoch.

        Args:
            runner (object): The runner of the training process.
        """
        self.t = time.time()

    def before_iter(
            self,
            runner: object,
            data_batch: Optional[Sequence[BaseDataSample]] = None) -> None:
        """Logging time for loading data and update the time flag.

        Args:
            runner (object): The runner of the training process.
            data_batch (Sequence[BaseDataSample]): Data from dataloader.
                Defaults to None.
        """
        # TODO: update for new logging system
        runner.log_buffer.update({  # type: ignore
            'data_time': time.time() - self.t
        })

    def after_iter(self,
                   runner: object,
                   data_batch: Optional[Sequence[BaseDataSample]] = None,
                   outputs: Optional[Sequence[BaseDataSample]] = None) -> None:
        """Logging time for a iteration and update the time flag.

        Args:
            runner (object): The runner of the training process.
            data_batch (Sequence[BaseDataSample]): Data from dataloader.
                Defaults to None.
            outputs (Sequence[BaseDataSample]): Outputs from model.
                Defaults to None.
        """
        # TODO: update for new logging system
        runner.log_buffer.update({  # type: ignore
            'time': time.time() - self.t
        })
        self.t = time.time()
