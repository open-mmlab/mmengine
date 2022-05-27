# Copyright (c) OpenMMLab. All rights reserved.
import logging
from contextlib import ExitStack, contextmanager
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad
from torch.optim import Optimizer

from mmengine.logging import MessageHub, MMLogger
from mmengine.registry import OPTIMIZER_WRAPPERS


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
        grad_clip (dict, optional): Configuration of gradient cropping.
        detect_anomalous_params (bool): Whether to detect anomalous
            parameters. Only used in debug mode. Defaults to False.

    Warnings:
        If ``cumulative_iters`` is larger than 1, :meth:`update_params` must be
        called in the context of ``gradient_accumulation``.

    Examples:
        OptimizerWrapper config sample
        >>> optimizer_wrapper_cfg = dict(
        >>>     type='OptimizerWrapper',
        >>>     cumulative_iters=3,
        >>>     grad_clip=dict(max_norm=0.2))
        Use OptimizerWrapper to update parameters in model.
        >>> class MyMODEL(BaseModel)
        >>>     ...
        >>>     def train_step(self, data, optimizer_wrapper):
        >>>         # calculate loss
        >>>         ...
        >>>         optimizer_wrapper.update_params(loss)
        Use OptimizerWrapper independently
        >>> # initialize train_dataloader before
        >>> for idx, data_batch in enumerate(train_dataloader)
        >>> # enable gradient accumulation
        >>>     with optimizer_wrapper.gradient_accumulation(
        >>>         idx, len(train_dataloader), model):
        >>>         loss = model(data)
        >>>         optimizer_wrapper.update_params(loss)
    """
    def __init__(self,
                 optimizer: Optimizer,
                 cumulative_iters: int = 1,
                 grad_clip: Optional[dict] = None,
                 detect_anomalous_params: bool = False):
        assert cumulative_iters > 0, (
            'cumulative_iters at least greater than or equal to 1')
        # `max_iters` and `cur_iter` is only valid in gradient accumulative
        # mode (`cumulative_iters` > 1).
        self.cur_iter = 0
        self.max_iters = 0

        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.logger = MMLogger.get_current_instance()
        self.cumulative_iters = cumulative_iters
        self.detect_anomalous_params = detect_anomalous_params
        # Using to update `grad_norm` log message.
        self.message_hub = MessageHub.get_current_instance()

    def update_params(self,
                      loss: torch.Tensor,
                      model: Optional[nn.Module] = None) -> None:
        """Update parameters in :attr:`optimizer`.

        Args:
            loss (torch.Tensor): A tensor for back propagation.
            model (nn.Module, optional): Training model. ``update_params``
                will check whether the model has BatchNormalization layer in
                model in gradient_acc
        """
        if self.cumulative_iters > 1:
            # gradient accumulation must be called in the context of
            # ``gradient_accumulation``.
            assert hasattr(self, 'divisible_iters'), (
                'gradient accumulation must be performed in the context of'
                '`OptimizerWrapper.gradient_accumulation`')
            # if `self.cumulative_iters > 1`, the gradient needs to be
            # rescaled and accumulated. In most cases, `loss_factor` equals to
            # `self.cumulative_iters`. However `self.max_iters` may not be
            # divisible `self.by cumulative_iters`, so the `loss_scale` for the
            # last few iterations needs to be recalculated.
            if self.cur_iter < self.divisible_iters:
                loss_factor = self.cumulative_iters
            else:
                loss_factor = self.remainder_iters
            assert loss_factor != 0, (
                'loss_factor should not be zero! This error could happened '
                'when message_hub update with an error `iter` or `max_iters`, '
                'please check your loop')
        else:
            # update parameters without gradient accumulation. The gradient
            # should not be rescaled and `loss_factor=1`.
            loss_factor = 1
        loss = loss / loss_factor
        self.backward(loss, model)
        # Update parameters only if `self.cur_iter` is divisible by
        # `self.cumulative_iters` or `self.cur_iter` equals to
        # `self.max_iters`
        if self._should_update(self.cur_iter, self.max_iters):
            if self.grad_clip:
                self._clip_grads()
            self.step()
            self.zero_grad()

    def backward(self,
                 loss: torch.Tensor,
                 model: Optional[nn.Module] = None) -> None:
        """Perform gradient backpropagation.

        Args:
            loss (torch.Tensor): The loss of current iteration.
            model (nn.Module, optional): The training model.
        """
        if self.detect_anomalous_params:
            assert model is not None
            self._detect_anomalous_params(loss, model)
        loss.backward()

    def zero_grad(self) -> None:
        """A wrapper of ``Optimizer.zero_grad``.

        clear grads of last iteration.
        """
        self.optimizer.zero_grad()

    def step(self) -> None:
        """A wrapper of ``Optimizer.step``.

        Update the parameters in
        :attr:`optimizer`
        """
        self.optimizer.step()

    def state_dict(self) -> dict:
        """A wrapper of ``Optimizer.state_dict``.

        Return the state dict of
        :attr:`optimizer`.
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
    def gradient_accumulation(self, model: nn.Module, cur_iter: int,
                              max_iters: int) -> None:
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
        if not self._should_update(cur_iter, max_iters):
            if isinstance(model, DistributedDataParallel):
                with model.no_sync():
                    yield None
            else:
                yield None
        else:
            yield None

    @contextmanager
    def precision_context(self) -> None:
        """precision context, default enable an empty context."""
        yield None

    def _clip_grads(self) -> None:
        """Clip the gradients of parameters."""
        params: List[torch.Tensor] = []
        for param_group in self.optimizer.param_groups:
            params.extend(param_group['params'])

        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            grad_norm = clip_grad.clip_grad_norm_(params, **self.grad_clip)
            self.message_hub.update_scalar('train/grad_norm', float(grad_norm))

    def _detect_anomalous_params(self, loss: torch.Tensor,
                                 model: nn.Module) -> None:
        """Detect anomalous parameters that are not included in the graph.

        Args:
            loss (torch.Tensor): The loss of current iteration.
            model (nn.Module): Training model.
        """
        parameters_in_graph = set()
        visited = set()

        # Traverse the grad function of loss and detects all
        # parameters participating in the forward inference
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
        # Detect parameters that do not appear in the graph.
        for n, p in model.named_parameters():
            if p not in parameters_in_graph and p.requires_grad:
                self.logger.log(
                    level=logging.ERROR,
                    msg=f'{n} with shape {p.size()} is not '
                    f'in the computational graph \n')

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

        if self._has_batch_norm(model) and self.cumulative_iters > 1:
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

    def _has_batch_norm(self, model: nn.Module) -> bool:
        """Detect whether model has a BatchNormalization layer.

        Args:
            model (nn.Module): training model.

        Returns:
            bool: whether model has a BatchNormalization layer
        """
        if isinstance(model, _BatchNorm):
            return True
        for m in model.children():
            if self._has_batch_norm(m):
                return True
        return False

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

    def backward(self, loss: torch.Tensor, model: Optional[nn.Module] = None):
        """Perform gradient backpropagation with :attr:`loss_scalar`.

        Args:
            loss (torch.Tensor): The loss of current iteration.
            model (nn.Module): The training model.
        """
        if self.detect_anomalous_params:
            assert model is not None
            self._detect_anomalous_params(loss, model)
        self.loss_scalar.scale(loss).backward()
        self.loss_scalar.unscale_(self.optimizer)

    def step(self):
        """Update parameters with :attr:`loss_scalar`."""
        if self.grad_clip:
            self._clip_grads()
        self.loss_scalar.step(self.optimizer)
        self.loss_scalar.update(self._scale_update_param)

    def state_dict(self) -> dict:
        """Get the state dict of :attr:`optimizer` and :attr:`loss_scalar`. The
        key of :attr:`loss_scalar` will be added with the prefix
        "loss_scalar.".

        Returns:
            dict: The merged state dict of :attr:`loss_scalar` and
            :attr:`optimizer`.
        """
        # save state_dict of loss_scalar
        optim_state_dict = self.optimizer.state_dict()
        for key, value in self.loss_scalar.state_dict().items():
            key = f'loss_scalar.{key}'
            optim_state_dict[key] = value
        return optim_state_dict

    def load_state_dict(self, state_dict: dict):
        """Load and parse the state dict of :attr:`optimizer` and
        :attr:`loss_scalar`.

        If state_dict contains the key starts with "loss_scalar.", the
        :attr:`loss_scalar` will load the corresponding keys. Otherwise only
        the :attr:`optimizer` will load the state dict.

        Args:
            state_dict (dict): The state dict of :attr:`optimizer` and
                :attr:`loss_scalar`
        """
        loss_scalar_dict = dict()
        for key in list(state_dict.keys()):
            if key.startswith('loss_scalar.'):
                ori_key = key.replace('loss_scalar.', '')
                loss_scalar_dict[ori_key] = state_dict.pop(key)

        # Load an empty loss_scalar_dict will raise RuntimeError.
        if loss_scalar_dict:
            self.loss_scalar.load_state_dict(loss_scalar_dict)
        self.optimizer.load_state_dict(state_dict)

    @contextmanager
    def precision_context(self) -> None:
        """A wrapper of ``torch.autocast``"""
        if torch.cuda.is_available():
            with torch.autocast('cuda'):
                yield None
        else:
            with torch.autocast('cpu'):
                yield None


@contextmanager
def gradient_accumulative_context(
        optimizer_wrapper: Union[dict, OptimizerWrapper]) -> None:
    """A context manager that enables ``gradient_accumulation`` context of
    multiple optimizers.

    Args:
        optimizer_wrapper (dict or OptimizerWrapper): Single or multiple
            OptimizerWrapper instances which need to enable
            ``gradient_accumulation`` context.
    """
    if isinstance(optimizer_wrapper, OptimizerWrapper):
        with optimizer_wrapper.gradient_accumulative_context():
            yield None
    # If `optimizer_wrapper` is a dict, we should make sure all
    # `optimizer_wrapper` have the same `cumulative_iters`, and they will
    # have the same gradient accumulative behavior.
    elif isinstance(optimizer_wrapper, dict):
        optimizer_wrapper_list = list(optimizer_wrapper.values())
        with ExitStack() as stack:
            for optim_wrapper in optimizer_wrapper_list:
                stack.enter_context(
                    optim_wrapper.gradient_accumulative_context())
            yield None
    else:
        raise TypeError('optimizer_wrapper should be `OptimizerWrapper`,'
                        f'but got {type(optimizer_wrapper)}')
