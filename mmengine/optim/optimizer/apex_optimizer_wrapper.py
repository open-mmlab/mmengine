# Copyright (c) OpenMMLab. All rights reserved.
import torch

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def backward(self, loss: torch.Tensor, **kwargs):
        """Perform gradient back propagation with :attr:`loss_scaler`.

        Args:
            loss (torch.Tensor): The loss of current iteration.
            kwargs: Keyword arguments passed to :meth:`torch.Tensor.backward`
        """
        with apex_amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        self._inner_count += 1
