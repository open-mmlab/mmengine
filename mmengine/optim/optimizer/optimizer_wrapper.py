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
        accumulative_counts (int): The number of iterations to accumulate
            gradients. The parameters will be updated per
            ``accumulative_counts``.
        clip_grad (dict, optional): If ``clip_grad`` is not None, it will be
            the arguments of ``torch.nn.utils.clip_grad``.

    Note:
        If ``accumulative_counts`` is larger than 1, perform
        :meth:`update_params` under the context of  ``optim_context``
        could avoid unnecessary gradient synchronization.

    Note:
        If you use ``IterBasedRunner`` and enable gradient accumulation,
        the original `max_iters` should be multiplied by
        ``accumulative_counts``.

    Note:
        The subclass should ensure that once :meth:`update_params` is called,
        ``_inner_count += 1`` is automatically performed.

    Examples:
        >>> # Config sample of OptimWrapper.
        >>> optim_wrapper_cfg = dict(
        >>>     type='OptimWrapper',
        >>>     _accumulative_counts=1,
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
        >>> # Enable gradient accumulation
        >>> optim_wrapper_cfg = dict(
        >>>     type='OptimWrapper',
        >>>     _accumulative_counts=3,
        >>>     clip_grad=dict(max_norm=0.2))
        >>> ddp_model = DistributedDataParallel(model)
        >>> optimizer = SGD(ddp_model.parameters(), lr=0.1)
        >>> optim_wrapper = OptimWrapper(optimizer)
        >>> optim_wrapper.initialize_count_status(0, len(dataloader))
        >>> # If model is a subclass instance of DistributedDataParallel,
        >>> # `optim_context` context manager can avoid unnecessary gradient
        >>> #  synchronize.
        >>> for iter, data in enumerate(dataloader):
        >>>     with optim_wrapper.optim_context(ddp_model):
        >>>         loss = model(data)
        >>>     optim_wrapper.update_params(loss)
    """

    def __init__(self,
                 optimizer: Optimizer,
                 accumulative_counts: int = 1,
                 clip_grad: Optional[dict] = None):
        assert accumulative_counts > 0, (
            '_accumulative_counts at least greater than or equal to 1')
        self._accumulative_counts = accumulative_counts

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
        # `_max_counts` means the total number of parameter updates.  It
        # ensures that the gradient of the last few iterations will not be
        # lost when the `_max_counts` is not divisible by
        # `accumulative_counts`.
        self._max_counts = -1
        # If `_inner_count` is smaller than `_divisible_counts`, the loss
        # factor used for gradient accumulation should be the same as
        # `_accumulative_counts`. If `_max_counts` has not been initialized,
        # the loss factor will always be the same as `_accumulative_counts`.
        self._divisible_counts = -1
        # The `_remainder_iter` is used for calculating loss factor at the
        # last few iterations. If `_max_counts` has not been initialized,
        # the loss factor will always be the same as `_accumulative_counts`.
        self._remainder_counts = -1

    def update_params(self, loss: torch.Tensor) -> None:
        """Update parameters in :attr:`optimizer`.

        Args:
            loss (torch.Tensor): A tensor for back propagation.
        """
        loss = self.scale_loss(loss)
        self.backward(loss)
        # Update parameters only if `self._inner_count` is divisible by
        # `self._accumulative_counts` or `self._inner_count` equals to
        # `self._max_counts`
        if self.should_update():
            self.step()
            self.zero_grad()

    def backward(self, loss: torch.Tensor) -> None:
        """Perform gradient back propagation.

        Provide unified ``backward`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic. For example, ``torch.cuda.amp`` require some extra
        operation on GradScaler during backward process.

        Note:
            If subclasses inherit from ``OptimWrapper`` override
            ``backward``, ``_inner_count +=1`` must be implemented.

        Args:
            loss (torch.Tensor): The loss of current iteration.
        """
        loss.backward()
        self._inner_count += 1

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
            Dict[str, List[float]]: Learning rate of the optimizer.
        """
        lr = [group['lr'] for group in self.param_groups]
        return dict(lr=lr)

    def get_momentum(self) -> Dict[str, List[float]]:
        """Get the momentum of the optimizer.

        Provide unified interface to get momentum of optimizer.

        Returns:
            Dict[str, List[float]]: Momentum of the optimizer.
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
    def optim_context(self, model: nn.Module):
        """A Context for gradient accumulation and automatic mix precision
        training.

        If subclasses need to enable the context for mix precision training,
        e.g., ``:class:`AmpOptimWrapper``,  the corresponding context should be
        enabled in `optim_context`. Since ``OptimWrapper`` uses default fp32
        training, ``optim_context`` will only enable the context for
        blocking the unnecessary gradient synchronization during gradient
        accumulation

        If model is an instance with ``no_sync`` method (which means
        blocking the gradient synchronization) and
        ``self._accumulative_counts != 1``. The model will not automatically
        synchronize gradients if ``cur_iter`` is divisible by
        ``self._accumulative_counts``. Otherwise, this method will enable an
        empty context.

        Args:
            model (nn.Module): The training model.
        """
        # During gradient accumulation process, the gradient synchronize
        # should only happen before updating parameters.
        if not self.should_sync() and hasattr(model, 'no_sync'):
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

    def initialize_count_status(self, model: nn.Module, init_counts: int,
                                max_counts: int) -> None:
        """Initialize gradient accumulation related attributes.

        ``OptimWrapper`` can be used without calling
        ``initialize_iter_status``. However, Consider the case of  ``len(
        dataloader) == 10``, and the ``accumulative_iter == 3``. Since 10 is
        not divisible by 3, the last iteration will not trigger
        ``optimizer.step()``, resulting in one less parameter updating.

        Args:
            model (nn.Module): Training model
            init_counts (int): The initial value of the inner count.
            max_counts (int): The maximum value of the inner count.
        """
        self._inner_count = init_counts
        self._max_counts = max_counts
        if self._inner_count % self._accumulative_counts != 0:
            self.logger.warning(
                'Resumed iteration number is not divisible by '
                '`_accumulative_counts` in `GradientCumulativeOptimizerHook`, '
                'which means the gradient of some iterations is lost and the '
                'result may be influenced slightly.')

        if has_batch_norm(model) and self._accumulative_counts > 1:
            self.logger.warning(
                'Gradient accumulative may slightly decrease '
                'performance because the model has BatchNorm layers.')
        residual_counts = max_counts - init_counts
        # The maximum number of training iteration that is divisible by
        # `_accumulative_counts`.
        self._divisible_counts = (
            residual_counts // self._accumulative_counts *
            self._accumulative_counts)
        # Remainder of `_max_counts` divided by `_accumulative_counts`
        self._remainder_counts = residual_counts - self._divisible_counts

    def should_update(self) -> bool:
        """Decide whether the parameters should be updated at the current
        iteration.

        Called by :meth:`update_params` and check whether the optimizer
        wrapper should update parameters at current iteration.

        Returns:
            bool: Whether to update parameters.
        """
        return (self._inner_count % self._accumulative_counts == 0
                or self._inner_count == self._max_counts)

    def should_sync(self) -> bool:
        """Decide whether the automatic gradient synchronization should be
        allowed at the current iteration.

        It takes effect when gradient accumulation is used to skip
        synchronization at the iterations where the parameter is not updated.

        Since ``should_sync`` is called by :meth:`optim_context`, and it is
        called before :meth:`backward` which means ``self._inner_count += 1``
        has not happened yet. Therefore, ``self._inner_count += 1`` should be
        performed manually here.

        Returns:
            bool: Whether to block the automatic gradient synchronization.
        """
        return ((self._inner_count + 1) % self._accumulative_counts == 0
                or (self._inner_count + 1) == self._max_counts)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Get scaled loss according to ``_accumulative_counts``,
        ``_inner_count`` and max_counts.

        Args:
            loss (torch.Tensor): Original loss calculated by model.

        Returns:
            loss (torch.Tensor): Scaled loss.
        """
        if self._accumulative_counts == 1:
            # update parameters without gradient accumulation. The gradient
            # should not be rescaled and `loss_factor=1`.
            loss_factor = 1
        elif self._max_counts == -1:
            loss_factor = self._accumulative_counts
        else:
            # if `self._accumulative_counts > 1`, the gradient needs to be
            # rescaled and accumulated. In most cases, `loss_factor` equals to
            # `self._accumulative_counts`. However, `self._max_counts` may not
            # be divisible by `self._accumulative_counts`, so the
            # `loss_scale` for the last few iterations needs to be
            # recalculated.
            if self._inner_count < self._divisible_counts:
                loss_factor = self._accumulative_counts
            else:
                loss_factor = self._remainder_counts
            assert loss_factor > 0, (
                'loss_factor should be larger than zero! This error could '
                'happened when initialize_iter_status called with an '
                'error `init_counts` or `max_counts`')

        loss = loss / loss_factor
        return loss

    @property
    def inner_count(self):
        """Get the number of updating parameters of optimizer wrapper."""
        return self._inner_count

    def __repr__(self):
        wrapper_info = (f'Type: {type(self).__name__}\n'
                        f'_accumulative_counts: {self._accumulative_counts}\n'
                        'optimizer: \n')
        optimizer_str = repr(self.optimizer) + '\n'
        return wrapper_info + optimizer_str
