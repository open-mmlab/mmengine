# Copyright (c) OpenMMLab. All rights reserved.
import time
from typing import Any, Optional, Sequence, Tuple, Union

from mmengine.data import BaseDataElement
from mmengine.registry import HOOKS
from .hook import Hook

DATA_BATCH = Optional[Sequence[Tuple[Any, BaseDataElement]]]


@HOOKS.register_module()
class IterTimerHook(Hook):
    """A hook that logs the time spent during iteration.

    E.g. ``data_time`` for loading data and ``time`` for a model train step.
    """

    priority = 'NORMAL'

    def _before_epoch(self, runner, mode: str = 'train') -> None:
        """Record time flag before start a epoch.

        Args:
            runner (Runner): The runner of the training process.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        self.t = time.time()

    def _before_iter(self,
                     runner,
                     batch_idx: int,
                     data_batch: DATA_BATCH = None,
                     mode: str = 'train') -> None:
        """Logging time for loading data and update the time flag.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the loop.
            data_batch (Sequence[Tuple[Any, BaseDataElement]], optional): Data
                from dataloader. Defaults to None.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        # TODO: update for new logging system
        runner.message_hub.update_log(f'{mode}/data_time',
                                      time.time() - self.t)

    def _after_iter(self,
                    runner,
                    batch_idx: int,
                    data_batch: DATA_BATCH = None,
                    outputs: Optional[Union[dict,
                                            Sequence[BaseDataElement]]] = None,
                    mode: str = 'train') -> None:
        """Logging time for a iteration and update the time flag.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the loop.
            data_batch (Sequence[Tuple[Any, BaseDataElement]], optional): Data
                from dataloader. Defaults to None.
            outputs (dict or sequence, optional): Outputs from model. Defaults
                to None.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        # TODO: update for new logging system

        runner.message_hub.update_log(f'{mode}/time', time.time() - self.t)
        self.t = time.time()
