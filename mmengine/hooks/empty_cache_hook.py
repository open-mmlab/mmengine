# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Sequence, Tuple, Union

import torch

from mmengine.data import BaseDataElement
from mmengine.registry import HOOKS
from .hook import Hook

DATA_BATCH = Optional[Sequence[Tuple[Any, BaseDataElement]]]


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
        self._do_before_epoch = before_epoch
        self._do_after_epoch = after_epoch
        self._do_after_iter = after_iter

    def _after_iter(self,
                    runner,
                    batch_idx: int,
                    data_batch: DATA_BATCH = None,
                    outputs: Optional[Union[dict,
                                            Sequence[BaseDataElement]]] = None,
                    mode: str = 'train') -> None:
        """Empty cache after an iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the loop.
            data_batch (Sequence[Tuple[Any, BaseDataElement]], optional): Data
                from dataloader. Defaults to None.
            outputs (dict or sequence, optional): Outputs from model.
                Defaults to None.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        if self._do_after_iter:
            torch.cuda.empty_cache()

    def _before_epoch(self, runner, mode: str = 'train') -> None:
        """Empty cache before an epoch.

        Args:
            runner (Runner): The runner of the training process.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        if self._do_before_epoch:
            torch.cuda.empty_cache()

    def _after_epoch(self, runner, mode: str = 'train') -> None:
        """Empty cache after an epoch.

        Args:
            runner (Runner): The runner of the training process.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        if self._do_after_epoch:
            torch.cuda.empty_cache()
