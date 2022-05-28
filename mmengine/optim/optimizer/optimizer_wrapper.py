# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad
from torch.optim import Optimizer

from mmengine.logging import MessageHub, MMLogger
from mmengine.registry import OPTIMIZER_WRAPPERS
from mmengine.utils import has_batch_norm


@OPTIMIZER_WRAPPERS.register_module()
class OptimizerWrapper:
    """OptimizerWrapper provides a common interface for updating parameters.

    OptimizerWrapper provides a unified interface for single precision training
    and automatic mixed precision training with different hardware.
    OptimizerWrapper is also a higher-order abstraction of optimizer which
    provides simplified interface for commonly used training techniques,
    such as gradient accumulative and grad clips.

    Args:
        optimizer (Optimizer): Optimizer used to update model parameters.
        cumulative_iters (int): Number of gradient accumulation. Defaults to 1.
        clip_grad_kwargs (dict, optional): If ``clip_grad_kwargs`` is not
            None, it will be the arguments of ``torch.nn.utils.clip_grad``.

    Warnings:
        If ``cumulative_iters`` is larger than 1, :meth:`update_params` must be
        called in the context of ``accumulate_grad``.

    Examples:
        OptimizerWrapper config sample.
        >>> optimizer_wrapper_cfg = dict(
        >>>     type='OptimizerWrapper',
        >>>     cumulative_iters=3,
        >>>     clip_grad_kwargs=dict(max_norm=0.2))
        Using OptimizerWrapper to update model.
        >>> import torch.nn as nn
        >>> import torch
        >>> from torch.optim import SGD
        >>> from torch.utils.data import DataLoader
        >>> from mmengine.optim import OptimizerWrapper
        >>>
        >>> model = nn.Linear(1, 1)
        >>> dataset = torch.randn(10, 1, 1)
        >>> dataloader = DataLoader(dataset)
        >>> optimizer = SGD(model.parameters(), lr=0.1)
        >>> optim_wrapper = OptimizerWrapper(optimizer)
        >>>
        >>> for data in dataloader:
        >>>     loss = model(data)
        >>>     optim_wrapper.update_params(loss)
        >>> # Enable gradient accumulation. If model is a subclass instance of
        >>> # DistributedDataParallel, ``accumulate_grad`` context manager can
        >>> # avoid unnecessary gradient synchronize.
        >>> for iter, data in enumerate(dataloader):
        >>>     with optim_wrapper.accumulate_grad(
        >>>         model, iter, len(dataloader)):
        >>>         loss = model(data)
        >>>         optim_wrapper.update_params(loss)
    """

    def __init__(self,
                 optimizer: Optimizer,
                 cumulative_iters: int = 1,
                 clip_grad_kwargs: Optional[dict] = None):
        assert cumulative_iters > 0, (
            'cumulative_iters at least greater than or equal to 1')
        # `max_iters` and `cur_iter` is only valid in gradient accumulative
        # mode (`cumulative_iters` > 1).
        self.cur_iter = 0
        self.max_iters = 0

        self.optimizer = optimizer

        if clip_grad_kwargs is not None:
            # clip_grad_kwargs should not be non-empty dict.
            assert isinstance(clip_grad_kwargs, dict) and clip_grad_kwargs, (
                'if `clip_grad_kwargs` is not None, it should be a `dict` '
                'which is the arguments of `torch.nn.utils.clip_grad`')
        self.clip_grad_kwargs = clip_grad_kwargs
        self.logger = MMLogger.get_current_instance()
        self.cumulative_iters = cumulative_iters
        # Using to update `grad_norm` log message.
        self.message_hub = MessageHub.get_current_instance()

    def update_params(self, loss: torch.Tensor) -> None:
        """Update parameters in :attr:`optimizer`.

        Args:
            loss (torch.Tensor): A tensor for back propagation.
        """
        if self.cumulative_iters == 1:
            # update parameters without gradient accumulation. The gradient
            # should not be rescaled and `loss_factor=1`.
            loss_factor = 1
        else:
            # gradient accumulation must be called in the context of
            # ``accumulate_grad``.
            assert hasattr(self, 'divisible_iters'), (
                'gradient accumulation must be performed in the context of'
                '`OptimizerWrapper.accumulate_grad`')
            # if `self.cumulative_iters > 1`, the gradient needs to be
            # rescaled and accumulated. In most cases, `loss_factor` equals to
            # `self.cumulative_iters`. However `self.max_iters` may not be
            # divisible `self.by cumulative_iters`, so the `loss_scale` for the
            # last few iterations needs to be recalculated.
            if self.cur_iter < self.divisible_iters:
                loss_factor = self.cumulative_iters
            else:
                loss_factor = self.remainder_iters
            assert loss_factor > 0, (
                'loss_factor should be larger than zero! This error could '
                'happened when gradient accumulation context enabled with an '
                'error `cur_iter` or `max_iters` please check your loop')

        loss = loss / loss_factor
        self.backward(loss)
        # Update parameters only if `self.cur_iter` is divisible by
        # `self.cumulative_iters` or `self.cur_iter` equals to
        # `self.max_iters`
        if self._should_update(self.cur_iter, self.max_iters):
            if self.clip_grad_kwargs:
                self._clip_grad()
            self.step()
            self.zero_grad()

    def backward(self, loss: torch.Tensor) -> None:
        """Perform gradient back propagation.

        Args:
            loss (torch.Tensor): The loss of current iteration.
        """
        loss.backward()

    def zero_grad(self) -> None:
        """A wrapper of ``Optimizer.zero_grad``.

        clear grads of last iteration.
        """
        self.optimizer.zero_grad()

    def step(self) -> None:
        """A wrapper of ``Optimizer.step``.

        Update the parameters in :attr:`optimizer`
        """
        self.optimizer.step()

    def state_dict(self) -> dict:
        """A wrapper of ``Optimizer.state_dict``.

        Return the state dict of :attr:`optimizer`.
        """
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """A wrapper of ``Optimizer.load_state_dict``. load the state dict of
        :attr:`optimizer`.

        Args:
            state_dict (dict): The state dict of :attr:`optimizer`.
        """
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self) -> List[dict]:
        """A wrapper of ``Optimizer.param_groups``.

        Return the ``param_groups`` of :attr:`optimizer`.
        """
        return self.optimizer.param_groups

    @contextmanager
    def accumulate_grad(self, model: nn.Module, cur_iter: int, max_iters: int):
        """A Context manager for gradient accumulation and avoiding unnecessary
        gradient synchronization during gradient accumulation.

        If model is a ``DistributedDataParallel`` instance and
        ``self.cumulative_iters != 1``. The model will not automatically
        synchronize gradients if ``cur_iter`` is divisible by
        ``self.cumulative_iters``. Otherwise, this method will enable an
        empty context.

        Warnings:
            This context manager must be enabled if you want to use
            gradient accumulation.

        Args:
            model (nn.Module): The training model.
            cur_iter (int): Current iteration during training process.
            max_iters (int): Maximum training iteration.
        """
        assert max_iters > 0, '`max_iters` must be larger than zero'
        self.cur_iter = cur_iter
        self.max_iters = max_iters
        if not hasattr(self, 'divisible_iters'):
            self._iter_status_initialized(model)
        # During gradient accumulation process, the gradient synchronize
        # should only happen before updating parameters.
        if (not self._should_update(cur_iter, max_iters)
                and hasattr(model, 'no_sync')):
            with model.no_sync():
                yield
        else:
            yield

    @contextmanager
    def precision_context(self):
        """precision context, default enable an empty context."""
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

    def _iter_status_initialized(self, model: nn.Module) -> None:
        """Initialize gradient accumulation related attributes.

        Args:
            model: Training model
        """
        if self.max_iters % self.cumulative_iters != 0:
            self.logger.warning(
                'Resume iter number is not divisible by cumulative_iters in '
                'GradientCumulativeOptimizerHook, which means the gradient of '
                'some iters is lost and the result may be influenced slightly.'
            )

        if has_batch_norm(model) and self.cumulative_iters > 1:
            self.logger.warning(
                'Gradient accumulative may slightly decrease '
                'performance if the model has BatchNorm layers.')
        residual_iters = self.max_iters - self.cur_iter
        # The maximum number of training iteration that is divisible by
        # cumulative_iters.
        self.divisible_iters = (
            residual_iters // self.cumulative_iters * self.cumulative_iters)
        # Remainder of ``self.max_iters`` divided by ``self.max_iters``
        self.remainder_iters = residual_iters - self.divisible_iters

    def _should_update(self, cur_iter: int, max_iters: int) -> bool:
        """Should optimizer_wrapper update parameters or synchronized gradient
        at current iteration.

        Args:
            cur_iter (int): Current iteration of training process.
            max_iters (int): Maximum iterations of training process.

        Returns:
            bool: Whether to update parameters or synchronized gradient.
        """
        return ((cur_iter + 1) % self.cumulative_iters == 0
                or cur_iter + 1 == max_iters)


@OPTIMIZER_WRAPPERS.register_module()
class AmpOptimizerWrapper(OptimizerWrapper):
    """A subclass of :class:`OptimizerWrapper` that supports automatic mixed
    precision training based on torch.cuda.amp.

    ``AmpOptimizerWrapper`` provide unified interface with
    ``OptimizerWrapper``, and ``AmpOptimizerWrapper`` and ``OptimizerWrapper``
     can be used in the same way.

    Warnings:
        You should use AmpOptimizerWrapper with PyTorch >= 1.6

    Args:
        loss_scale (float or str or dict): Scale factor configuration.
            If loss_scale is a float, static loss scaling will be used with
            the specified scale. If loss_scale is a string, it must be
            'dynamic', then dynamic loss scaling will be used.
            It can also be a dict containing arguments of GradScalar.
            Defaults to 512. For Pytorch >= 1.6, mmcv uses official
            implementation of GradScaler. If you use a dict version of
            loss_scale to create GradScaler, please refer to:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
            for the parameters.
    """

    def __init__(self, loss_scale=512., **kwargs):
        super().__init__(**kwargs)
        self._scale_update_param = None
        if loss_scale == 'dynamic':
            self.loss_scalar = GradScaler()
        elif isinstance(loss_scale, float):
            self._scale_update_param = loss_scale
            self.loss_scalar = GradScaler(init_scale=loss_scale)
        elif isinstance(loss_scale, dict):
            self.loss_scalar = GradScaler(**loss_scale)
        else:
            raise TypeError('loss_scale must be of type float, dict, or '
                            f'"dynamic", got {loss_scale}')

    def backward(self, loss: torch.Tensor):
        """Perform gradient backpropagation with :attr:`loss_scalar`.

        Args:
            loss (torch.Tensor): The loss of current iteration.
        """
        self.loss_scalar.scale(loss).backward()
        self.loss_scalar.unscale_(self.optimizer)

    def step(self):
        """Update parameters with :attr:`loss_scalar`."""
        if self.clip_grad_kwargs:
            self._clip_grad()
        self.loss_scalar.step(self.optimizer)
        self.loss_scalar.update(self._scale_update_param)

    def state_dict(self) -> dict:
        """Get the state dictionary of :attr:`optimizer` and
        :attr:`loss_scalar`.

        based on the state dictionary of optimizer, The returned state
        dictionary will add a key named "loss_scalar".

        Returns:
            dict: The merged state dict of :attr:`loss_scalar` and
            :attr:`optimizer`.
        """
        # save state_dict of loss_scalar
        state_dict = self.optimizer.state_dict()
        state_dict['loss_scalar'] = self.loss_scalar.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        """Load and parse the state dictionary of :attr:`optimizer` and
        :attr:`loss_scalar`.

        If state_dict contains the key starts with "loss_scalar.", the
        :attr:`loss_scalar` will load the corresponding keys. Otherwise only
        the :attr:`optimizer` will load the state dictionary.

        Args:
            state_dict (dict): The state dict of :attr:`optimizer` and
                :attr:`loss_scalar`
        """
        if 'loss_scalar' in state_dict:
            self.loss_scalar.load_state_dict(state_dict.pop('loss_scalar'))
        self.optimizer.load_state_dict(state_dict)

    @contextmanager
    def precision_context(self):
        """A wrapper of ``torch.autocast``"""
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        with torch.autocast(device):
            yield
