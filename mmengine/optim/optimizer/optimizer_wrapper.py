# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad
from torch.optim import Optimizer

from mmengine.logging import MessageHub, MMLogger
from mmengine.registry import OPTIM_WRAPPERS
from mmengine.utils import has_batch_norm


@OPTIM_WRAPPERS.register_module()
class OptimWrapper:
    """Optimizer wrapper provides a common interface for updating parameters.

    Optimizer wrapper provides a unified interface for single precision
    training and automatic mixed precision training with different hardware.
    OptimWrapper encapsulates optimizer to provide simplified interfaces
    for commonly used training techniques such as gradient accumulative and
    grad clips. ``OptimWrapper`` implements the basic logic of gradient
    accumulation and gradient clipping based on ``torch.optim.Optimizer``.
    The subclasses only need to override some methods to implement the mixed
    precision training. See more information in :class:`AmpOptimWrapper`.

    Args:
        optimizer (Optimizer): Optimizer used to update model parameters.
        accumulative_iters (int): The number of iterations to accumulate
            gradients. The parameters will be updated per
            ``accumulative_iters``.
        clip_grad (dict, optional): If ``clip_grad`` is not None, it will be
            the arguments of ``torch.nn.utils.clip_grad``.

    Warnings:
        If ``accumulative_iters`` is larger than 1, :meth:`update_params` must
        be called in the context of ``optimizer_context``.

    Examples:
        >>> # Config sample of OptimWrapper.
        >>> optim_wrapper_cfg = dict(
        >>>     type='OptimWrapper',
        >>>     accumulative_iters=3,
        >>>     clip_grad=dict(max_norm=0.2))
        >>> # Use OptimWrapper to update model.
        >>> import torch.nn as nn
        >>> import torch
        >>> from torch.optim import SGD
        >>> from torch.utils.data import DataLoader
        >>> from mmengine.optim import OptimWrapper
        >>>
        >>> model = nn.Linear(1, 1)
        >>> dataset = torch.randn(10, 1, 1)
        >>> dataloader = DataLoader(dataset)
        >>> optimizer = SGD(model.parameters(), lr=0.1)
        >>> optim_wrapper = OptimWrapper(optimizer)
        >>>
        >>> for data in dataloader:
        >>>     loss = model(data)
        >>>     optim_wrapper.update_params(loss)
        >>> # Enable gradient accumulation. If model is a subclass instance of
        >>> # DistributedDataParallel, ``optimizer_context`` context manager
        >>> # can avoid unnecessary gradient synchronize.
        >>> for iter, data in enumerate(dataloader):
        >>>     with optim_wrapper.optimizer_context(
        >>>         model, iter, len(dataloader)):
        >>>         loss = model(data)
        >>>         optim_wrapper.update_params(loss)
    """

    def __init__(self,
                 optimizer: Optimizer,
                 accumulative_iters: int = 1,
                 clip_grad: Optional[dict] = None):
        assert accumulative_iters > 0, (
            'accumulative_iters at least greater than or equal to 1')
        self.accumulative_iters = accumulative_iters

        assert isinstance(optimizer, Optimizer), (
            'optimizer must be a `torch.optim.Optimizer` instance, but got '
            f'{type(optimizer)}')
        self.optimizer = optimizer

        if clip_grad is not None:
            # clip_grad_kwargs should not be non-empty dict.
            assert isinstance(clip_grad, dict) and clip_grad, (
                'If `clip_grad` is not None, it should be a `dict` '
                'which is the arguments of `torch.nn.utils.clip_grad`')
        self.clip_grad_kwargs = clip_grad
        self.logger = MMLogger.get_current_instance()
        # Used to update `grad_norm` log message.
        self.message_hub = MessageHub.get_current_instance()
        self._inner_count = 0
        # initiate an unreachable `max_iters`. If `initilize_iter_status` is
        # not called, `max_iters` will not influence the gradient accumulation
        # process.
        self.max_iters = int(1e9)
        # If `_inner_count` smaller than `divisible_iters`, the loss factor
        # used for gradient accumulation should be the same as
        # `accumulative_iters`. If `max_iters` has not been initialized,
        # it should also be unreachable to make loss factor equal to
        # `accumulative_iters`
        self.divisible_iters = self.max_iters
        # The `remainder_iter` is used for calculating `loss_scaler` at the
        # last few iterations. If `max_iters` has not been initialized,
        # `remainder_iters` will not be used.
        self.remainder_iters = self.accumulative_iters

    def update_params(self, loss: torch.Tensor) -> None:
        """Update parameters in :attr:`optimizer`.

        Args:
            loss (torch.Tensor): A tensor for back propagation.
        """
        if self.accumulative_iters == 1:
            # update parameters without gradient accumulation. The gradient
            # should not be rescaled and `loss_factor=1`.
            loss_factor = 1
        else:
            # if `self.accumulative_iters > 1`, the gradient needs to be
            # rescaled and accumulated. In most cases, `loss_factor` equals to
            # `self.accumulative_iters`. However `self.max_iters` may not be
            # divisible `self.by accumulative_iters`, so the `loss_scale` for
            # the last few iterations needs to be recalculated.
            if self._inner_count < self.divisible_iters:
                loss_factor = self.accumulative_iters
            else:
                loss_factor = self.remainder_iters
            assert loss_factor > 0, (
                'loss_factor should be larger than zero! This error could '
                'happened when initilize_iter_status called with an '
                'error `cur_iter` or `max_iters`')

        loss = loss / loss_factor
        self.backward(loss)
        # Update parameters only if `self.cur_iter` is divisible by
        # `self.accumulative_iters` or `self.cur_iter` equals to
        # `self.max_iters`
        if self._should_update(self._inner_count, self.max_iters):
            self.step()
            self.zero_grad()
        self._inner_count += 1

    def backward(self, loss: torch.Tensor) -> None:
        """Perform gradient back propagation.

        Provide unified ``backward`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic. For example, ``torch.cuda.amp`` require some extra
        operation on GradScaler during backward process.

        Args:
            loss (torch.Tensor): The loss of current iteration.
        """
        loss.backward()

    def zero_grad(self) -> None:
        """A wrapper of ``Optimizer.zero_grad``.

        Provide unified ``zero_grad`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic.
        """
        self.optimizer.zero_grad()

    def step(self) -> None:
        """A wrapper of ``Optimizer.step``.

        Provide unified ``step`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic. For example, ``torch.cuda.amp`` require some extra
        operation on ``GradScaler`` during step process.

        Clip grad if :attr:`clip_grad_kwargs` is not None, and then update
        parameters.
        """
        if self.clip_grad_kwargs:
            self._clip_grad()
        self.optimizer.step()

    def state_dict(self) -> dict:
        """A wrapper of ``Optimizer.state_dict``.

        Provide unified ``state_dict`` interface compatible with automatic
        mixed precision training. Subclass can overload this method to
        implement the required logic. For example, the state dictionary of
        GradScaler should be saved when training with ``torch.cuda.amp``.

        Returns:
            dict: The state dictionary of :attr:`optimizer`.
        """
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """A wrapper of ``Optimizer.load_state_dict``. load the state dict of
        :attr:`optimizer`.

        Provide unified ``load_state_dict`` interface compatible with automatic
        mixed precision training. Subclass can overload this method to
        implement the required logic. For example, the state dictionary of
        GradScaler should be loaded when training with ``torch.cuda.amp``.

        Args:
            state_dict (dict): The state dictionary of :attr:`optimizer`.
        """
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self) -> List[dict]:
        """A wrapper of ``Optimizer.param_groups``.

        Make OptimizeWrapper compatible with :class:`_ParamScheduler`.

        Returns:
             dict: the ``param_groups`` of :attr:`optimizer`.
        """
        return self.optimizer.param_groups

    def get_lr(self) -> Dict[str, List[float]]:
        """Get the learning rate of the optimizer.

        Provide unified interface to get learning rate of optimizer.

        Returns:
            List[float]: Learning rate of the optimizer.
        """
        lr = [group['lr'] for group in self.param_groups]
        return dict(lr=lr)

    def get_momentum(self) -> Dict[str, List[float]]:
        """Get the momentum of the optimizer.

        Provide unified interface to get momentum of optimizer.

        Returns:
            List[float]: Momentum of the optimizer.
        """
        momentum = []
        for group in self.param_groups:
            # Get momentum of SGD.
            if 'momentum' in group.keys():
                momentum.append(group['momentum'])
            # Get momentum of Adam.
            elif 'betas' in group.keys():
                momentum.append(group['betas'][0])
            else:
                momentum.append(0)
        return dict(momentum=momentum)

    @contextmanager
    def optimizer_context(self, model: nn.Module):
        # TODO
        """A Context manager for gradient accumulation and avoiding unnecessary
        gradient synchronization during gradient accumulation.

        If model is an instance with ``no_sync`` method (which means
        blocking the gradient synchronization) and
        ``self.accumulative_iters != 1``. The model will not automatically
        synchronize gradients if ``cur_iter`` is divisible by
        ``self.accumulative_iters``. Otherwise, this method will enable an
        empty context.

        Warnings:
            This context manager must be enabled if you want to use
            gradient accumulation.

        Args:
            model (nn.Module): The training model.
        """
        # During gradient accumulation process, the gradient synchronize
        # should only happen before updating parameters.
        if (not self._should_update(self._inner_count, self.max_iters)
                and hasattr(model, 'no_sync')):
            with model.no_sync():
                yield
        else:
            yield

    def _clip_grad(self) -> None:
        """Clip the gradients of parameters."""
        params: List[torch.Tensor] = []
        for param_group in self.optimizer.param_groups:
            params.extend(param_group['params'])

        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            grad_norm = clip_grad.clip_grad_norm_(params,
                                                  **self.clip_grad_kwargs)
            self.message_hub.update_scalar('train/grad_norm', float(grad_norm))

    def initilize_iter_status(self, model: nn.Module, cur_iter,
                              max_iters) -> None:
        """Initialize gradient accumulation related attributes.

        Args:
            model (nn.Module): Training model
        """
        self._inner_count = cur_iter
        self.max_iters = max_iters
        if self._inner_count % self.accumulative_iters != 0:
            self.logger.warning(
                'Resume iter number is not divisible by accumulative_iters in '
                'GradientCumulativeOptimizerHook, which means the gradient of '
                'some iters is lost and the result may be influenced slightly.'
            )

        if has_batch_norm(model) and self.accumulative_iters > 1:
            self.logger.warning(
                'Gradient accumulative may slightly decrease '
                'performance because the model has BatchNorm layers.')
        residual_iters = max_iters - cur_iter
        # The maximum number of training iteration that is divisible by
        # accumulative_iters.
        self.divisible_iters = (
            residual_iters // self.accumulative_iters *
            self.accumulative_iters)
        # Remainder of ``max_iters`` divided by ``accumulative_iters``
        self.remainder_iters = residual_iters - self.divisible_iters

    def _should_update(self, cur_iter: int, max_iters: int) -> bool:
        """Should optim_wrapper update parameters or synchronized gradient at
        current iteration.

        Args:
            cur_iter (int): Current iteration of training process.
            max_iters (int): Maximum iterations of training process.

        Returns:
            bool: Whether to update parameters or synchronized gradient.
        """
        return ((cur_iter + 1) % self.accumulative_iters == 0
                or cur_iter + 1 == max_iters)

    def __repr__(self):
        wrapper_info = f'Type: {type(self).__name__}\n' \
                       f'accumulative_iters: {self.accumulative_iters}\n' \
                       f'optimizer: \n'
        optimizer_str = repr(self.optimizer) + '\n'
        return wrapper_info + optimizer_str
