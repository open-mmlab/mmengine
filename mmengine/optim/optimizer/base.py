# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List

import torch


class BaseOptimWrapper(metaclass=ABCMeta):

    def __init__(self, optimizer):
        self.optimizer = optimizer

    @abstractmethod
    def update_params(self, *args, **kwargs):
        """Update parameters in :attr:`optimizer`."""

    @abstractmethod
    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """Perform gradient back propagation."""

    @abstractmethod
    def step(self, **kwargs):
        """Call the step method of optimizer."""

    @abstractmethod
    def get_lr(self):
        """Get the learning rate of the optimizer.

        Provide unified interface to get learning rate of optimizer.

        Returns:
            Dict[str, List[float]]:
            param_groups learning rate of the optimizer.
        """

    @abstractmethod
    def state_dict(self) -> dict:
        """A wrapper of ``Optimizer.state_dict``."""

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        """A wrapper of ``Optimizer.load_state_dict``."""

    @property
    def param_groups(self) -> List[dict]:
        """A wrapper of ``Optimizer.param_groups``.

        Make OptimizeWrapper compatible with :class:`_ParamScheduler`.

        Returns:
             dict: the ``param_groups`` of :attr:`optimizer`.
        """
        return self.optimizer.param_groups

    @property
    def defaults(self) -> dict:
        """A wrapper of ``Optimizer.defaults``.

        Make OptimizeWrapper compatible with :class:`_ParamScheduler`.

        Returns:
             dict: the ``param_groups`` of :attr:`optimizer`.
        """
        return self.optimizer.defaults
