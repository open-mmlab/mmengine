# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager

import torch
import torch.nn as nn

from mmengine.registry import OPTIM_WRAPPERS
from .optimizer_wrapper import OptimWrapper

try:
    import apex.amp as apex_amp
except ImportError:
    pass


@OPTIM_WRAPPERS.register_module()
class ApexOptimWrapper(OptimWrapper):
    """A subclass of :class:`OptimWrapper` that supports automatic mixed
    precision training based on apex.amp.

    ``ApexOptimWrapper`` provides a unified interface with
    ``OptimWrapper``, so ``ApexOptimWrapper`` can be used in the same way
    as ``OptimWrapper``.

    Warnings:
        ``ApexOptimWrapper`` requires
            [nvidia apex](https://github.com/NVIDIA/apex).

    Args:

        **kwargs: Keyword arguments passed to OptimWrapper.

    Note:
        If you use ``IterBasedRunner`` and enable gradient accumulation,
        the original `max_iters` should be multiplied by
        ``accumulative_counts``.
    """

    def __init__(self, opt_level='O1', loss_scale='dynamic', **kwargs):
        super().__init__(**kwargs)
        self.opt_level = opt_level
        self.loss_scale = loss_scale

    def backward(self, loss: torch.Tensor, **kwargs):
        """Perform gradient back propagation with :attr:`loss_scaler`.

        Args:
            loss (torch.Tensor): The loss of current iteration.
            kwargs: Keyword arguments passed to :meth:`torch.Tensor.backward`
        """
        with apex_amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        self._inner_count += 1

    def state_dict(self) -> dict:
        """Get the state dictionary of :attr:`optimizer` and
        :attr:`apex_amp`.

        Based on the state dictionary of the optimizer, the returned state
        dictionary will add a key named "apex_amp".

        Returns:
            dict: The merged state dict of :attr:`apex_amp` and
            :attr:`optimizer`.
        """
        state_dict = self.optimizer.state_dict()
        state_dict['apex_amp'] = apex_amp.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        """Load and parse the state dictionary of :attr:`optimizer` and
        :attr:`apex_amp`.

        If state_dict contains "apex_amp", the :attr:`apex_amp` will
        load the corresponding keys. Otherwise, only the :attr:`optimizer`
        will load the state dictionary.

        Args:
            state_dict (dict): The state dict of :attr:`optimizer` and
                :attr:`apex_amp`
        """
        if 'apex_amp' in state_dict:
            apex_amp.load_state_dict(state_dict.pop('apex_amp'))
        self.optimizer.load_state_dict(state_dict)

    @contextmanager
    def optim_context(self, model: nn.Module):
        """Enables the context for mixed precision training, and enables the
        context for disabling gradient synchronization during gradient
        accumulation context.

        Args:
            model (nn.Module): The training model.
        """
        if hasattr(self.optimizer, '_amp_stash'):
            yield
        else:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
            with super().optim_context(model):
                model, self.optimizer = apex_amp.initialize(
                    model,
                    self.optimizer,
                    opt_level=self.opt_level,
                    loss_scale=self.loss_scale)
                yield
