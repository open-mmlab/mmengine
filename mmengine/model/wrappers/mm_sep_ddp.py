# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import ExitStack, contextmanager
from typing import List

import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel

from mmengine.data import BaseDataElement
from mmengine.optim import OptimWrapperDict
from mmengine.registry import MODEL_WRAPPERS
from .mm_ddp import MMDistributedDataParallel


@MODEL_WRAPPERS.register_module()
class MMSeparateDDPWrapper(DistributedDataParallel):
    """A DistributedDataParallel wrapper for models in MMGeneration.

    In MMedting and MMGeneration there is a need to wrap different modules in
    the models with separate DistributedDataParallel. Otherwise, it will cause
    errors for GAN training. More specific, the GAN model, usually has two
    submodules: generator and discriminator. If we wrap both of them in one
    standard DistributedDataParallel, it will cause errors during training,
    because when we update the parameters of the generator (or discriminator),
    the parameters of the discriminator (or generator) is not updated, which is
    not allowed for DistributedDataParallel. So we design this wrapper to
    separately wrap DistributedDataParallel for generator and discriminator.
    In this wrapper, we perform two operations:
    1. Wrap each module in the models with separate MMDistributedDataParallel.
        Note that only modules with parameters will be wrapped.
    2. Do scatter operation for 'forward', 'train_step' and 'val_step'.
    Note that the arguments of this wrapper is the same as those in
    `torch.nn.parallel.distributed.DistributedDataParallel`.

    Args:
        module (nn.Module): model contain multiple submodules which have
            separately updating strategy.
        *args: list arguments passed to ``MMDistributedDataParallel``
        **kwargs: keyword arguments passed to ``MMDistributedDataParallel``.
    """

    def __init__(self, module: nn.Module, *args, **kwargs):
        super(DistributedDataParallel, self).__init__()
        self.module = module
        # Wrap the submodule with parameters of `self.module` to
        # `MMDistributedDataParallel`
        for name, _module in module._modules.items():
            # module without parameters.
            if next(_module.parameters(), None) is None:
                _module = _module.cuda()
            elif all(not p.requires_grad for p in module.parameters()):
                _module = _module.cuda()
            else:
                _module = MMDistributedDataParallel(
                    module=_module.cuda(), *args, **kwargs)
            module._modules[name] = _module

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapperDict) -> dict:
        """Train step function.

        Args:
            data: Data sampled by dataloader.
            optim_wrapper (OptimWrapperDict): A wrapper of optimizer to
                update parameters.
        """
        output = self.module.train_step(data, optim_wrapper)
        return output

    def val_step(self, data) -> List[BaseDataElement]:
        """Get the prediction of module during validation process.

        Args:
            data (List[dict]): Data sampled by dataloader.

        Returns:
            List[BaseDataElement]: The predictions or losses.
        """
        return self.module.val_step(data)

    def test_step(self, data: List[dict]) -> List[BaseDataElement]:
        """Get the predictions of module during testing process.

        Args:
            data: Data sampled by dataloader.

        Returns:
            List[BaseDataElement]: The predictions of module.
        """
        return self.module.test_step(data)

    @contextmanager
    def no_sync(self):
        """enable ``no_sync`` context of all sub ``MMDistributedDataParallel``
        modules."""
        with ExitStack() as stack:
            for sub_ddp_model in self.module._modules.values():
                stack.enter_context(sub_ddp_model.no_sync())
                yield

    def train(self, mode=True):
        """Sets the module in training mode.

        In order to make the ddp wrapper inheritance hierarchy more uniform,
        ``MMSeparateDDPWrapper`` inherits from ``DistributedDataParallel``,
        but will not call its constructor. Since the attributes of
        ``DistributedDataParallel`` have not been initialized, call the
        ``train`` method of ``DistributedDataParallel`` will report an
        error if pytorch version <= 1.9. Therefore, override this method
        to call the ``train`` method of submodules.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self.
        """
        self.training = mode
        return self.module.train(mode)
