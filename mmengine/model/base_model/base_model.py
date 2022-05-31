# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmengine.config import Config
from mmengine.data import BaseDataElement
from mmengine.optim import OptimWrapper
from mmengine.registry import MODELS

DataSamples = Optional[Union[list, torch.Tensor]]
ForwardResults = Union[Dict[str, torch.Tensor], List[BaseDataElement],
                       Tuple[torch.Tensor], torch.Tensor]


# TODO inherit from BaseModule
class BaseModel(nn.Module, metaclass=ABCMeta):
    """Base class for all algorithmic models.

    BaseModel implements the basic functions of the algorithmic model, such as
    weights initialize, batch inputs preprocess(see more information in
    :class:`BaseDataPreprocessor`), parse losses, and update model parameters.
    You can set up your model

    Model inherits from BaseModel only needs to implement the forward
    method, and then can be trained in the runner.

    Examples:
        >>> @MODELS.register_module()
        >>> class ToyModel(BaseModel):
        >>>
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.backbone = nn.Sequential()
        >>>         self.backbone.add_module('conv1', nn.Conv2d(3, 6, 5))
        >>>         self.backbone.add_module('pool', nn.MaxPool2d(2, 2))
        >>>         self.backbone.add_module('conv2', nn.Conv2d(6, 16, 5))
        >>>         self.backbone.add_module('fc1', nn.Linear(16 * 5 * 5, 120))
        >>>         self.backbone.add_module('fc2', nn.Linear(120, 84))
        >>>         self.backbone.add_module('fc3', nn.Linear(84, 10))
        >>>
        >>>         self.criterion = nn.CrossEntropyLoss()
        >>>
        >>>     def forward(self, batch_inputs, data_samples, mode='feat'):
        >>>         data_samples = torch.stack(data_samples)
        >>>         if mode == 'feat':
        >>>             return self.backbone(batch_inputs)
        >>>         elif mode == 'predict':
        >>>             feats = self.backbone(batch_inputs)
        >>>             predictions = torch.argmax(feats, 1)
        >>>             return predictions
        >>>         elif mode == 'loss':
        >>>             feats = self.backbone(batch_inputs)
        >>>             loss = self.criterion(feats, data_samples)
        >>>             return dict(loss=loss)

    Args:
        init_cfg (dict or Config or ConfigDict, optional): The weight
            initialized config for :class:`BaseModule`.
        data_preprocessor (dict or Config, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.

    Attributes:
        data_preprocessor (:obj:`BaseDataPreprocessor`): Used for
            pre-processing data sampled by dataloader to the format accepted by
            :meth:`forward`.
    """

    def __init__(self,
                 data_preprocessor: Optional[Union[dict, Config]] = None):
        super().__init__()
        if data_preprocessor is None:
            data_preprocessor = dict(type='BaseDataPreprocessor')
        self.data_preprocessor = MODELS.build(data_preprocessor)

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implement the default model training process including pre-
        processing, model forward propagation, loss calculation, optimization,
        and back-propagation.

        During non-distributed training.If subclass does not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        - call ``self.data_processor(data, training=False) to collext
          batch_inputs and corresponding data_samples(labels).
        - call ``self(batch_inputs, data_samples, mode='loss')`` to get raw
          loss
        - call ``self.parse_losses`` to get ``parsed_losses`` tensor used to
          backward and dict of loss tensor used to log messages.
        - call ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (List[dict]): Data sampled from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: dict of tensor for logging.
        """
        inputs, data_sample = self.data_preprocessor(data, True)
        with optim_wrapper.precision_context():
            losses = self(inputs, data_sample, mode='loss')
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data: List[dict]) -> List[BaseDataElement]:
        """Get the predictions of given data.

        Call ``self.data_preprocessor(data, False)`` and
        ``self(inputs, data_sample, mode='predict')`` in sequence. Return the
        predictions used by evaluator.

        Args:
            data (List[dict]): Data sampled from dataloader.

        Returns:
            List[BaseDataElement]: The predictions of given data.
        """
        inputs, data_sample = self.data_preprocessor(data, False)
        return self(inputs, data_sample, mode='predict')

    def test_step(self, data: List[dict]) -> List[BaseDataElement]:
        """BaseModel implement ``test_step`` the same as ``val_step``.

        Args:
            data (List[dict]): Data sampled from dataloader.

        Returns:
            List[BaseDataElement]: The predictions of given data.
        """
        inputs, data_sample = self.data_preprocessor(data, False)
        return self(inputs, data_sample, mode='predict')

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
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

    def to(self, device: Optional[Union[int, torch.device]], *args,
           **kwargs) -> nn.Module:
        """Override this method to set the ``device`` attribute of
        :obj:`BaseDataPreprocessor` additionally

        Args:
            device (int or torch.device, optional): the desired device of the
                parameters and buffers in this module.

        Returns:
            nn.Module: The model itself.
        """
        self.data_preprocessor.device = torch.device(device)
        return super().to(device)

    def cuda(self, *args, **kwargs) -> nn.Module:
        """Override this method to set the ``device`` attribute of
        :obj:`BaseDataPreprocessor` additionally

        Returns:
            nn.Module: The model itself.
        """
        self.data_preprocessor.device = torch.cuda.current_device()
        return super().cuda()

    @abstractmethod
    def forward(self,
                batch_inputs: torch.Tensor,
                data_samples: DataSamples = None,
                mode: str = 'feat') -> ForwardResults:
        """Get losses or predictions of training, validation, testing, and
        simple inference process.

        ``forward`` method of BaseModel is an abstract method, subclass must
        override this method.

        Accept ``batch_inputs`` and ``data_samples`` processed by
        :attr:`data_preprocessor`, and return results according to mode
        arguments.

        During non-dist training, validation and testing  process, ``forward``
        will be called by ``BaseModel.train_step``, ``BaseModel.val_step`` and
        ``BaseModel.val_step`` directly.

        During distributed data parallel training process, since calling
        ``DistributedDataParallel.forward`` can achieve automatic gradient
        synchronization, :obj:`MMDistributedDataParallel` will call
        ``DistributedDataParallel.forward`` in ``train_step``, and
        further calls the ``forward`` method of ``BaseModel`` subclass.

        Args:
            batch_inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (torch.Tensor, list, optional):
                data samples collated by :attr:`data_preprocessor`.
            mode (str): mode should be one of ``loss``, ``predict`` and
                ``feat``

                - ``loss``: Called by ``train_step`` and return loss dict used
                  for logging
                - ``predict``: Called by ``val_step`` and ``test_step``
                  and return list of ``BaseDataElement`` results used for
                  computing metric.
                - ``feat``: Called by custom use to get a ``Tensor`` type
                  results.

        Returns:
            dict or list or torch.Tensor or tuple:
                - dict of loss tensor used for logging.
                - list of :BaseDataElement:`BaseDataElement` for
                  computing metric.
                - Tensor or tuple of tensor or dict or tensor for custom use.
        """
