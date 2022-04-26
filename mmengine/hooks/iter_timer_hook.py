# Copyright (c) OpenMMLab. All rights reserved.
import time
from typing import Optional, Sequence, Union

from mmengine.data import BaseDataElement
from mmengine.registry import HOOKS
from .hook import Hook

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class IterTimerHook(Hook):
    """A hook that logs the time spent during iteration.

    E.g. ``data_time`` for loading data and ``time`` for a model train step.
    """

    priority = 'NORMAL'

    def __init__(self):
        self.time_sec_tot = 0
        self.start_iter = 0

    def before_run(self, runner) -> None:
        """Synchronize the number of iterations with the runner.

        Args:
            runner: The runner of the training, validation or testing
                process.
        """
        self.start_iter = runner.iter

    def _before_epoch(self, runner, mode: str = 'train') -> None:
        """Record timestamp before start an epoch.

        Args:
            runner (Runner): The runner of the training validation and
                testing process.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        self.t = time.time()

    def _before_iter(self,
                     runner,
                     batch_idx: int,
                     data_batch: DATA_BATCH = None,
                     mode: str = 'train') -> None:
        """Calculating time for loading data and updating "data_time"
        ``HistoryBuffer`` of ``runner.message_hub``.

        Args:
            runner (Runner): The runner of the training, validation and
                testing process.
            batch_idx (int): The index of the current batch in the loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        # Update data loading time in `runner.message_hub`.
        runner.message_hub.update_scalar(f'{mode}/data_time',
                                         time.time() - self.t)

    def _after_iter(self,
                    runner,
                    batch_idx: int,
                    data_batch: DATA_BATCH = None,
                    outputs: Optional[Union[dict,
                                            Sequence[BaseDataElement]]] = None,
                    mode: str = 'train') -> None:
        """Calculating time for an iteration and updating "time"
        ``HistoryBuffer`` of ``runner.message_hub``.

        Args:
            runner (Runner): The runner of the training validation and
                testing process.
            batch_idx (int): The index of the current batch in the loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict or sequence, optional): Outputs from model. Defaults
                to None.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        # Update iteration time in `runner.message_hub`.
        message_hub = runner.message_hub
        message_hub.update_scalar(f'{mode}/time', time.time() - self.t)
        self.t = time.time()
        window_size = runner.log_processor.window_size
        # Calculate eta every `window_size` iterations. Since test and val
        # loop will not update runner.iter, use `every_n_innter_iters`to check
        # the interval.
        if self.every_n_inner_iters(batch_idx, window_size):
            iter_time = message_hub.get_scalar(f'{mode}/time').mean(
                window_size)
            if mode == 'train':
                self.time_sec_tot += iter_time * window_size
                # Calculate average iterative time.
                time_sec_avg = self.time_sec_tot / (
                    runner.iter - self.start_iter + 1)
                # Calculate eta.
                eta_sec = time_sec_avg * (
                    runner.train_loop.max_iters - runner.iter - 1)
                runner.message_hub.update_info('eta', eta_sec)
            else:
                if mode == 'val':
                    cur_dataloader = runner.val_loop.dataloader
                else:
                    cur_dataloader = runner.test_loop.dataloader

                eta_sec = iter_time * (len(cur_dataloader) - batch_idx - 1)
                runner.message_hub.update_info('eta', eta_sec)
