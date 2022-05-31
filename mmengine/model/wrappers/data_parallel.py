# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import ExitStack, contextmanager
from typing import List, Union

import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from mmengine.data import BaseDataElement
from mmengine.optim import OptimWrapper, OptimWrapperDict
from mmengine.registry import MODEL_WRAPPERS
from mmengine.utils import detect_anomalous_params

MODEL_WRAPPERS.register_module(module=DataParallel)
MODEL_WRAPPERS.register_module(module=DistributedDataParallel)


@MODEL_WRAPPERS.register_module()
class MMDistributedDataParallel(DistributedDataParallel):
    """A distributed model wrapper used for training,testing and validation in
    loop.

    Different from DistributedDataParallel, MMDistributedDataParallel
    implements a :meth:`train_step`, :meth:`test_step`  and :meth:`val_step`,
    which will be called by ``train_loop``, ``val_loop`` and ``test_loop``.

    - ``train_step``: Called by ``runner.train_loop``, and implement
      default model forward, gradient back propagation, parameter updating
      logic. To take advantage of DistributedDataParallel's automatic gradient
      synchronization, ``train_step`` calls ``DistributedDataParallel.forward``
      to calculate the losses, and call other methods of :obj:`BaseModel` to
      pre-process data and parse losses. Finally, update model parameters by
      :obj:``OptimWrapper`` and return the loss dictionary used for logging.

    - ``val_step``: Called by ``runner.val_loop`` and get the inference
      results. Since there is no gradient synchronization requirement,
      this procedure is equivalent to ``BaseModel.val_step``

    - ``test_step``: Called by ``runner.test_loop``, equivalent ``val_step``.

    Args:
        detect_anomalous_params (bool): This option is only used for
            debugging which will slow down the training speed.
            Detect anomalous parameters that are not included in
            the computational graph with `loss` as the root.
            There are two cases

                - Parameters were not used during
                  forward pass.
                - Parameters were not used to produce
                  loss.
            Default: False.

        *args: list arguments passed to ``DistributedDataParallel``
        **kwargs: keyword arguments passed to ``DistributedDataParallel``.

    Note:
        If model have multiple submodules and each module have
        separately optimization strategies. :class:`MMSeparateDDPWrapper`
        should be used to wrap the model.

    Note:
        If model itself has custom optimization strategy, rather than
        simply forward model and update model. A custom model wrapper
        inherit from ``MMDistributedDataParallel`` should be defined and
        override the ``train_step`` method.
    """

    def __init__(self, detect_anomalous_params: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detect_anomalous_params = detect_anomalous_params

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> dict:
        """Interface for model forward, backward and parameters updating during
        training process.

        :meth:`train_step` will perform the following steps in order:

        - If :attr:`module` defines the preprocess method,
            call ``module.preprocess`` to pre-processing data.
        - Call ``module.forward(**data)`` and get losses.
        - Parse losses.
        - Call ``optim_wrapper.optimizer_step`` to update parameters.
        - Return log messages of losses.

        Args:
            data (List[dict]): Data sampled by dataloader.
            optim_wrapper (OptimWrapper): A wrapper of optimizer to
                update parameters.

        Returns:
            dict: A tensor dict used to log training losses.
        """
        data = self.module.data_preprocessor(data, training=True)
        with optim_wrapper.precision_context():
            losses = self(*data, mode='loss')
        if self.detect_anomalous_params:
            detect_anomalous_params(losses, model=self)
        parsed_loss, log_vars = self.module.parse_losses(losses)
        optim_wrapper.update_params(parsed_loss)
        return log_vars

    def val_step(self, data: List[dict]) -> Union[List[BaseDataElement], dict]:
        """Get the prediction of module during validation process.

        Args:
            data (List[dict]): Data sampled by dataloader.

        Returns:
            List[BaseDataElement] or dict: The predictions or losses.
        """
        return self.module.val_step(data)

    def test_step(self, data: List[dict]) -> List[BaseDataElement]:
        """Get the predictions of module during testing process.

        Args:
            data: Data sampled by dataloader.

        Returns:
            List[BaseDataElement]: The predictions of module.
        """
        return self.module.test_step(data)  # type: ignore


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
    1. Wrap the modules in the models with separate MMDistributedDataParallel.
        Note that only modules with parameters will be wrapped.
    2. Do scatter operation for 'forward', 'train_step' and 'val_step'.
    Note that the arguments of this wrapper is the same as those in
    `torch.nn.parallel.distributed.DistributedDataParallel`.

    Args:
        module (nn.Module): model contain multiple submodules which have
            separately updating strategy.
        *args: list arguments passed to ``DistributedDataParallel``
        **kwargs: keyword arguments passed to ``DistributedDataParallel``.
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
            optim_wrapper (OptimWrapper): A wrapper of optimizer to
                update parameters.
        """
        output = self.module.train_step(data, optim_wrapper)
        return output

    def val_step(self, data) -> List[BaseDataElement]:
        """Get the losses or prediction of module during validation process.

        ``val_step`` will return losses if ``return_loss==True``, otherwise it
        returns the predictions.

        Args:
            data (List[dict]): Data sampled by dataloader.

        Returns:
            List[BaseDataElement] or dict: The predictions or losses.
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
