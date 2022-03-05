# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import torch

from mmengine.data import BaseDataSample
from mmengine.registry import HOOKS
from .hook import Hook


@HOOKS.register_module()
class EmptyCacheHook(Hook):
    """Releases all unoccupied cached GPU memory during the process of
    training.

    Args:
        before_epoch (bool): Whether to release cache before an epoch. Defaults
            to False.
        after_epoch (bool): Whether to release cache after an epoch. Defaults
            to True.
        after_iter (bool): Whether to release cache after an iteration.
            Defaults to False.
    """

    priority = 'NORMAL'

    def __init__(self,
                 before_epoch: bool = False,
                 after_epoch: bool = True,
                 after_iter: bool = False) -> None:
        self._before_epoch = before_epoch
        self._after_epoch = after_epoch
        self._after_iter = after_iter

    def after_iter(self,
                   runner: object,
                   data_batch: Optional[Sequence[BaseDataSample]] = None,
                   outputs: Optional[Sequence[BaseDataSample]] = None) -> None:
        """Empty cache after an iteration.

        Args:
            runner (object): The runner of the training process.
            data_batch (Sequence[BaseDataSample]): Data from dataloader.
                Defaults to None.
            outputs (Sequence[BaseDataSample]): Outputs from model.
                Defaults to None.
        """
        if self._after_iter:
            torch.cuda.empty_cache()

    def before_epoch(self, runner: object) -> None:
        """Empty cache before an epoch.

        Args:
            runner (object): The runner of the training process.
        """
        if self._before_epoch:
            torch.cuda.empty_cache()

    def after_epoch(self, runner: object) -> None:
        """Empty cache after an epoch.

        Args:
            runner (object): The runner of the training process.
        """
        if self._after_epoch:
            torch.cuda.empty_cache()
