# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional, Iterable

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad
from torch.optim import Optimizer
from torch.nn.modules.batchnorm import _BatchNorm
from torch.cuda.amp import GradScaler

from mmengine.logging import MessageHub, MMLogger


class OptimizerWrapper:
    """

    """

    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 cumulative_iters: int = 1,
                 grad_clip: Optional[dict] = None,
                 detect_anomalous_params: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.logger = MMLogger.get_current_instance()
        self.message_hub = MessageHub.get_current_instance()
        self.cumulative_iters = cumulative_iters
        self.detect_anomalous_params = detect_anomalous_params
        self.initialized = False

    def _init(self):
        cur_iter = self.message_hub.get_info('iter', 0)
        max_iters = self.message_hub.get_info('max_iters')
        if cur_iter % self.cumulative_iters != 0:
            self.logger.warning(
                'Resume iter number is not divisible by cumulative_iters in '
                'GradientCumulativeOptimizerHook, which means the gradient of '
                'some iters is lost and the result may be influenced slightly.'
            )

        if self._has_batch_norm(self.model) and self.cumulative_iters > 1:
            self.logger.warning(
                'Gradient accumulative may slightly decrease '
                'performance if the model has BatchNorm layers.')

        residual_iters = max_iters - cur_iter

        self.divisible_iters = (
                residual_iters // self.cumulative_iters * self.cumulative_iters)
        self.remainder_iters = residual_iters - self.divisible_iters

        self.initialized = True
        self.zero_grad()

    def _has_batch_norm(self, model: nn.Module) -> bool:
        """

        Args:
            model:

        Returns:

        """
        if isinstance(model, _BatchNorm):
            return True
        for m in model.children():
            if self._has_batch_norm(m):
                return True
        return False

    def prepare_loss(self, cur_iter, loss):
        if cur_iter < self.divisible_iters:
            loss_factor = self.cumulative_iters
        else:
            loss_factor = self.remainder_iters
        loss = loss / loss_factor
        return loss

    def optimizer_step(self, loss: torch.Tensor) -> None:
        """

        Args:
            loss:

        Returns:

        """
        cur_iter = self.message_hub.get_info('iter', 0)
        max_iters = self.message_hub.get_info('max_iters', 0)
        if not self.initialized:
            self._init()

        loss = self.prepare_loss(cur_iter, loss)
        self.backward(loss)
        if self._should_update(cur_iter, max_iters):
            self._update_model()

    def _should_update(self, cur_iter, max_iters):
        return ((cur_iter + 1) % self.cumulative_iters == 0 or
                cur_iter + 1 == max_iters)

    def _update_model(self):
        self.step()
        self.zero_grad()

    def backward(self, loss):
        """
        Args:
            loss:

        Returns:

        """
        if self.detect_anomalous_params:
            self._detect_anomalous_params(loss)
        loss.backward()

    def zero_grad(self):
        """"""
        self.optimizer.zero_grad()

    def step(self):
        """"""
        if self.grad_clip:
            self._clip_grads()
        self.optimizer.step()

    def _clip_grads(self):
        """Clip the gradients of parameters.

         Args:
             params (list[Parameter]): Model's parameters.

         Returns:
             Optional[torch.Tensor]: Total norm of the parameters if there is
             at least one param requiring gradient, else None.
         """
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None,
                   self.model.parameters()))
        if len(params) > 0:
            grad_norm = clip_grad.clip_grad_norm_(params, **self.grad_clip)
            self.message_hub.update_scalar('train/grad_norm', float(grad_norm))

    def _detect_anomalous_params(self, loss: torch.Tensor) -> None:
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

    @property
    def param_groups(self):
        return self.optimizer.param_groups


class AmpOptimizerWrapper(OptimizerWrapper):
    def __init__(self,
                 loss_scale=512.,
                 **kwargs):
        super().__init__(**kwargs)
        self._scale_update_param = None
        if loss_scale == 'dynamic':
            self.loss_scaler = GradScaler()
        elif isinstance(loss_scale, float):
            self._scale_update_param = loss_scale
            self.loss_scaler = GradScaler(init_scale=loss_scale)
        elif isinstance(loss_scale, dict):
            self.loss_scaler = GradScaler(**loss_scale)
        else:
            raise ValueError('loss_scale must be of type float, dict, or '
                             f'"dynamic", got {loss_scale}')

    def zero_grad(self):
        # clear grads of last iteration
        self.model.zero_grad()
        self.optimizer.zero_grad()

    def backward(self, loss):
        if self.detect_anomalous_params:
            self._detect_anomalous_params(loss)
        self.loss_scaler.scale(loss).backward()
        self.loss_scaler.unscale_(self.optimizer)

    def step(self):
        if self.grad_clip:
            self._clip_grads()
        self.loss_scaler.step(self.optimizer)
        self.loss_scaler.update(self._scale_update_param)

    def state_dict(self):
        # save state_dict of loss_scaler
        self.message_hub.update_info('loss_scalar',
                                     self.loss_scaler.state_dict())
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        scaler_state_dict = self.message_hub.get_info['loss_scalar']
        self.loss_scaler.load_state_dict(scaler_state_dict)
        self.optimizer.load_state_dict(state_dict)
