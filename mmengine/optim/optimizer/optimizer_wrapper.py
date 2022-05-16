# Copyright (c) OpenMMLab. All rights reserved.
import logging
from abc import ABCMeta, abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad
from torch.optim import Optimizer

from mmengine.logging import MessageHub, MMLogger


class _BaseOptimizerWrapper(metaclass=ABCMeta):

    def __init__(self, model, optimizer, grad_clip) -> None:
        self.model = model
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.logger = MMLogger.get_current_instance()
        self.messge_hub = MessageHub.get_current_instance()

    @abstractmethod
    def optimizer_step(self, loss):
        """_summary_

        Args:
            loss (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """

    @abstractmethod
    def backward(self, loss):
        """_summary_

            Returns:
                _type_: _description_
        """

    @abstractmethod
    def zero_grad(self, loss):
        """_summary_

        Args:
            loss (_type_): _description_

        Returns:
            _type_: _description_
        """

    @abstractmethod
    def grad_clips(self):
        """_summary_

        Returns:
            _type_: _description_
        """

    @abstractmethod
    def state_dict(self):
        """_summary_

        Returns:
            _type_: _description_
        """

    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        """_summary_

        Args:
            state_dict (dict): _description_

        Returns:
            _type_: _description_
        """

    @property
    def param_groups(self):
        return self.optimizer.param_groups


class _OptimizerWrapper(_BaseOptimizerWrapper):

    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 grad_clip=None,
                 detect_anomalous_params=None):
        super().__init__(model, optimizer, grad_clip)
        self.detect_anomalous_params = detect_anomalous_params

    def clip_grads(self, params: List[nn.Parameter]) -> Optional[torch.Tensor]:
        """Clip the gradients of parameters.

        Args:
            params (list[Parameter]): Model's parameters.

        Returns:
            Optional[torch.Tensor]: Total norm of the parameters if there is
            at least one param requiring gradient, else None.
        """
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)
        return None

    def optimizer_step(self, loss: torch.Tensor = None) -> None:
        self.zero_grad()
        self.backward(loss)
        if self.grad_clip:
            self.grad_clips()
        self.step()

    def backward(self, loss):
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(loss)
        loss.backward()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def grad_clips(self):
        grad_norm = self.clip_grads(self.model.parameters())
        if grad_norm is not None:
            self.message_hub.update_scalar('train/grad_norm', float(grad_norm))

    def detect_anomalous_parameters(self, loss: torch.Tensor) -> None:
        """Detect anomalous parameters that are not included in the graph.

        Args:
            loss (torch.Tensor): The loss of current iteration.
        """
        parameters_in_graph = set()
        visited = set()

        def traverse(grad_fn):
            if grad_fn is None:
                return
            if grad_fn not in visited:
                visited.add(grad_fn)
                if hasattr(grad_fn, 'variable'):
                    parameters_in_graph.add(grad_fn.variable)
                parents = grad_fn.next_functions
                if parents is not None:
                    for parent in parents:
                        grad_fn = parent[0]
                        traverse(grad_fn)

        traverse(loss.grad_fn)
        for n, p in self.model.named_parameters():
            if p not in parameters_in_graph and p.requires_grad:
                self.logger.log(
                    level=logging.ERROR,
                    msg=f'{n} with shape {p.size()} is not '
                    f'in the computational graph \n')

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
