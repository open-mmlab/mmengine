# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from itertools import chain
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from mmengine.data import BaseDataElement
from mmengine.optim import OptimizerWrapper, gradient_accumulative_context
from mmengine.registry import MODEL_WRAPPERS

MODEL_WRAPPERS.register_module(module=DataParallel)
MODEL_WRAPPERS.register_module(module=DistributedDataParallel)


@MODEL_WRAPPERS.register_module()
class MMDataParallel(DataParallel):
    """There is no difference between MMDataParallel and pytorch's
    DataParallel, "train_step" and "val_step" are added just to avoid bc
    breaking.

    Warning:
        MMDataParallel only supports single GPU training, if you
        need to  train with multiple GPUs, please use MMDistributedDataParallel
        instead. If you have multiple GPUs and you just want to use
        MMDataParallel, you can set the environment variable
        ``CUDA_VISIBLE_DEVICES=0`` or instantiate ``MMDataParallel`` with
        ``device_ids=[0]``.
    """

    def train_step(self, *inputs, **kwargs):
        assert len(self.device_ids) == 1, \
            ('MMDataParallel only supports single GPU training, if you need to'
             ' train with multiple GPUs, please use MMDistributedDataParallel'
             ' instead.')
        assert hasattr(self.module, 'train_step')
        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')
        return self.module.train_step(*inputs, **kwargs)

    def val_step(self, *inputs, **kwargs):
        assert len(self.device_ids) == 1, \
            ('MMDataParallel only supports single GPU training, if you need to'
             ' train with multiple GPUs, please use MMDistributedDataParallel'
             ' instead.')
        assert hasattr(self.module, 'val_step')
        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')
        return self.module.val_step(*inputs, **kwargs)


@MODEL_WRAPPERS.register_module()
class MMDistributedDataParallel(DistributedDataParallel):
    """There is no difference between MMDistributedDataParallel and pytorch's
    DistributedDataParallel, "train_step" and "val_step" are added just to
    avoid bc breaking."""

    def train_step(self, data: List[dict],
                   optimizer_wrapper: OptimizerWrapper) -> dict:
        """Interface for model forward, backward and parameter updating during
        training process.

        :meth:`train_step` will perform the following steps in order:

        - If :attr:`module` defines the preprocess method,
            call ``module.preprocess`` to pre-processing data.
        - Call ``module.forward(**data)`` and get losses.
        - Parse losses.
        - Call ``optimizer_wrapper.optimizer_step`` to update parameters.
        - Return log messages of losses.

        Args:
            data (List[dict]): Data sampled by dataloader.
            optimizer_wrapper (OptimizerWrapper): A wrapper of optimizer to
                update parameters.

        Returns:
            dict: A tensor dict used to log training losses.
        """
        with gradient_accumulative_context(optimizer_wrapper):
            data = self.module.preprocess(data, training=True)
            losses = self(*data, return_loss=True)
            parsed_loss, log_vars = self._parse_losses(losses)
            optimizer_wrapper.optimizer_step(parsed_loss)
            return log_vars

    def val_step(
            self,
            data: List[dict],
            return_loss: bool = False) -> Union[List[BaseDataElement], dict]:
        """Get the losses or prediction of module during validation process.

        ``val_step`` will return losses if ``return_loss==True``, otherwise it
        returns the predictions.

        Args:
            data (List[dict]): Data sampled by dataloader.
            return_loss (bool): Decide whether to return the losses or
                predictions.

        Returns:
            List[BaseDataElement] or dict: The predictions or losses.
        """
        if return_loss:
            data = self.module.preprocess(data, training=True)
            losses = self(*data, return_loss=True)
            _, log_vars = self._parse_losses(losses)
            return log_vars
        else:
            data = self.module.preprocess(data, training=False)
            return self.module(*data, return_loss=False)

    def test_step(self, data: List[dict]) -> List[BaseDataElement]:
        """Get the predictions of module during testing process.

        Args:
            data: Data sampled by dataloader.

        Returns:
            List[BaseDataElement]: The predictions of module.
        """
        return self.val_step(data, return_loss=False)  # type: ignore

    def _parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)
        log_vars['loss'] = loss

        return loss, log_vars


@MODEL_WRAPPERS.register_module()
class MMSeporateDDPWrapper(nn.Module):
    """A DistributedDataParallel wrapper for models in MMGeneration.

    In MMedting, there is a need to wrap different modules in the models
    with separate DistributedDataParallel. Otherwise, it will cause
    errors for GAN training.
    More specific, the GAN model, usually has two sub-modules:
    generator and discriminator. If we wrap both of them in one
    standard DistributedDataParallel, it will cause errors during training,
    because when we update the parameters of the generator (or discriminator),
    the parameters of the discriminator (or generator) is not updated, which is
    not allowed for DistributedDataParallel.
    So we design this wrapper to separately wrap DistributedDataParallel
    for generator and discriminator.
    In this wrapper, we perform two operations:
    1. Wrap the modules in the models with separate MMDistributedDataParallel.
        Note that only modules with parameters will be wrapped.
    2. Do scatter operation for 'forward', 'train_step' and 'val_step'.
    Note that the arguments of this wrapper is the same as those in
    `torch.nn.parallel.distributed.DistributedDataParallel`.
    Args:
        module (nn.Module): Module that needs to be wrapped.
        device_ids (list[int | `torch.device`]): Same as that in
            `torch.nn.parallel.distributed.DistributedDataParallel`.
        dim (int, optional): Same as that in the official scatter function in
            pytorch. Defaults to 0.
        broadcast_buffers (bool): Same as that in
            `torch.nn.parallel.distributed.DistributedDataParallel`.
            Defaults to False.
        find_unused_parameters (bool, optional): Same as that in
            `torch.nn.parallel.distributed.DistributedDataParallel`.
            Traverse the autograd graph of all tensors contained in returned
            value of the wrapped module’s forward function. Defaults to False.
        kwargs (dict): Other arguments used in
            `torch.nn.parallel.distributed.DistributedDataParallel`.
    """

    def __init__(self,
                 module,
                 dim=0,
                 broadcast_buffers=False,
                 find_unused_parameters=False,
                 **kwargs):
        super().__init__()
        self.module = module
        self.dim = dim
        self.to_ddp(
            dim=dim,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
            **kwargs)

    def to_ddp(self, dim, broadcast_buffers, find_unused_parameters, **kwargs):
        """Wrap models with separate MMDistributedDataParallel.

        It only wraps the modules with parameters.
        """
        for name, module in self.module._modules.items():
            if next(module.parameters(), None) is None:
                module = module.cuda()
            elif all(not p.requires_grad for p in module.parameters()):
                module = module.cuda()
            else:
                module = MMDistributedDataParallel(
                    module.cuda(),
                    dim=dim,
                    broadcast_buffers=broadcast_buffers,
                    find_unused_parameters=find_unused_parameters,
                    **kwargs)
            self.module._modules[name] = module

    def train_step(self, data: List[dict], optimizer_wrapper: dict):
        """Train step function.

        Args:
            data: Data sampled by dataloader.
            optimizer_wrapper (OptimizerWrapper): A wrapper of optimizer to
                update parameters.
        """
        with gradient_accumulative_context(optimizer_wrapper):
            output = self.module.train_step(data, optimizer_wrapper)
            return output

    def val_step(self, data, return_loss=False):
        """Get the losses or prediction of module during validation process.

        ``val_step`` will return losses if ``return_loss==True``, otherwise it
        returns the predictions.

        Args:
            data (List[dict]): Data sampled by dataloader.
            return_loss (bool): Decide whether to return the losses or
                predictions.

        Returns:
            List[BaseDataElement] or dict: The predictions or losses.
        """
        return self.module.val_step(data, return_loss=return_loss)

    def test_step(self, data: List[dict]) -> List[BaseDataElement]:
        """Get the predictions of module during testing process.

        Args:
            data: Data sampled by dataloader.

        Returns:
            List[BaseDataElement]: The predictions of module.
        """
        return self.module.test_step(data)
