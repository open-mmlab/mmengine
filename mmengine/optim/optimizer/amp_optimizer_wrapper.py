# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager

import torch
import torch.nn as nn

from mmengine.device import is_cuda_available, is_npu_available
from mmengine.registry import OPTIM_WRAPPERS
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from .optimizer_wrapper import OptimWrapper

if is_npu_available():
    from torch.npu.amp import GradScaler
else:
    from torch.cuda.amp import GradScaler


@OPTIM_WRAPPERS.register_module()
class AmpOptimWrapper(OptimWrapper):
    """A subclass of :class:`OptimWrapper` that supports automatic mixed
    precision training based on torch.cuda.amp.

    ``AmpOptimWrapper`` provides a unified interface with
    ``OptimWrapper``, so ``AmpOptimWrapper`` can be used in the same way
    as ``OptimWrapper``.

    Warnings:
        ``AmpOptimWrapper`` requires PyTorch >= 1.6.

    Args:
        loss_scale (float or str or dict): The initial configuration of
            `torch.cuda.amp.GradScaler`. See more specific arguments
            introduction at `PyTorch AMP <https://pytorch.org/docs/stable/amp.html?highlight=gradscalertorch.cuda.amp.GradScaler>`_ # noqa: E501
            Defaults to ``dynamic``.

            - "dynamic": Initialize GradScale without any arguments.
            - float: Initialize GradScaler with ``init_scale``.
            - dict: Initialize GradScaler with more detail configuration.

        **kwargs: Keyword arguments passed to OptimWrapper.

    Note:
        If you use ``IterBasedRunner`` and enable gradient accumulation,
        the original `max_iters` should be multiplied by
        ``accumulative_counts``.
    """

    def __init__(self, loss_scale='dynamic', **kwargs):
        assert digit_version(TORCH_VERSION) >= digit_version('1.6.0'), (
            '`torch.cuda.amp` is only available when pytorch version >= 1.6')
        assert is_cuda_available() or is_npu_available(), (
            '``AmpOptimizerWrapper`` is only available training on gpu or npu')
        super().__init__(**kwargs)
        self._scale_update_param = None
        if loss_scale == 'dynamic':
            #  If loss_scale is a string, it must be 'dynamic', then dynamic
            #  loss scaling will be used.
            self.loss_scaler = GradScaler()
        elif isinstance(loss_scale, float):
            # Static loss scaling
            self._scale_update_param = loss_scale
            self.loss_scaler = GradScaler(init_scale=loss_scale)
        elif isinstance(loss_scale, dict):
            # More specific configuration.
            self.loss_scaler = GradScaler(**loss_scale)
        else:
            raise TypeError('loss_scale must be of type float, dict, or '
                            f'"dynamic", but got {loss_scale}')

    def backward(self, loss: torch.Tensor, **kwargs):
        """Perform gradient back propagation with :attr:`loss_scaler`.

        Args:
            loss (torch.Tensor): The loss of current iteration.
            kwargs: Keyword arguments passed to :meth:`torch.Tensor.backward`
        """
        self.loss_scaler.scale(loss).backward(**kwargs)
        self._inner_count += 1

    def step(self, **kwargs):
        """Update parameters with :attr:`loss_scaler`.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`torch.optim.Optimizer.step`.
        """
        if self.clip_grad_kwargs:
            self.loss_scaler.unscale_(self.optimizer)
            self._clip_grad()
        self.loss_scaler.step(self.optimizer, **kwargs)
        self.loss_scaler.update(self._scale_update_param)

    def state_dict(self) -> dict:
        """Get the state dictionary of :attr:`optimizer` and
        :attr:`loss_scaler`.

        Based on the state dictionary of the optimizer, the returned state
        dictionary will add a key named "loss_scaler".

        Returns:
            dict: The merged state dict of :attr:`loss_scaler` and
            :attr:`optimizer`.
        """
        # save state_dict of loss_scaler
        state_dict = self.optimizer.state_dict()
        state_dict['loss_scaler'] = self.loss_scaler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        """Load and parse the state dictionary of :attr:`optimizer` and
        :attr:`loss_scaler`.

        If state_dict contains "loss_scaler.", the :attr:`loss_scaler` will
        load the corresponding keys. Otherwise, only the :attr:`optimizer`
        will load the state dictionary.

        Args:
            state_dict (dict): The state dict of :attr:`optimizer` and
                :attr:`loss_scaler`
        """
        if 'loss_scaler' in state_dict:
            self.loss_scaler.load_state_dict(state_dict.pop('loss_scaler'))
        self.optimizer.load_state_dict(state_dict)

    @contextmanager
    def optim_context(self, model: nn.Module):
        """Enables the context for mixed precision training, and enables the
        context for disabling gradient synchronization during gradient
        accumulation context.

        Args:
            model (nn.Module): The training model.
        """
        from mmengine.runner.amp import autocast
        with super().optim_context(model), autocast():
            yield
