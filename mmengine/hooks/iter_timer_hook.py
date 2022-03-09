# Copyright (c) OpenMMLab. All rights reserved.
import time
from typing import Any, Optional, Sequence, Tuple, Union

from mmengine.data import BaseDataSample
from mmengine.registry import HOOKS
from .hook import Hook

DATA_BATCH = Optional[Sequence[Tuple[Any, BaseDataSample]]]


@HOOKS.register_module()
class IterTimerHook(Hook):
    """A hook that logs the time spent during iteration.

    Eg. ``data_time`` for loading data and ``time`` for a model train step.
    """

    priority = 'NORMAL'

    def before_epoch(self, runner) -> None:
        """Record time flag before start a epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self.t = time.time()

    def before_iter(self, runner, data_batch: DATA_BATCH = None) -> None:
        """Logging time for loading data and update the time flag.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional): Data
                from dataloader. Defaults to None.
        """
        # TODO: update for new logging system
        runner.log_buffer.update({'data_time': time.time() - self.t})

    def after_iter(self,
                   runner,
                   data_batch: DATA_BATCH = None,
                   outputs:
                   Optional[Union[dict, Sequence[BaseDataSample]]] = None) \
            -> None:
        """Logging time for a iteration and update the time flag.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional): Data
                from dataloader. Defaults to None.
            outputs (dict or sequence, optional): Outputs from model. Defaults
                to None.
        """
        # TODO: update for new logging system
        runner.log_buffer.update({'time': time.time() - self.t})
        self.t = time.time()
