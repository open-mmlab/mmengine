# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmengine.data import BaseDataElement
from mmengine.optim import OptimizerWrapper
from mmengine.registry import MODELS
from mmengine.config import Config


# TODO inherit from BaseModule
@MODELS.register_module()
class BaseModel(nn.Module):
    """Base class for all algorithmic models.

    BaseModel implements the basic functions of the algorithmic model, such as
    weights initialize, batch inputs preprocess(see more information in
    :class:`BaseDataPreprocessor`), parse losses, and update model parameters.
    You can set up your model

    Model inherits from BaseModel only needs to implement the forward
    method, and then can be trained in the runner.

    Examples:
        >>> @MODELS.register_module(BaseModel)
        >>> class ToyModel(nn.Module):
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
        init_cfg (dict or Config, optional): The weight initialized config for
            :class:`BaseModule`.
        data_preprocessor (dict or Config, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
    """
    def __init__(self,
                 data_preprocessor: Optional[Union[dict, Config]] = None):
        super().__init__()
        if data_preprocessor is None:
            data_preprocessor = dict(type='BaseDataPreprocessor')
        self.data_preprocessor = MODELS.build(data_preprocessor)

    def train_step(
            self, data: List[dict],
            optimizer_wrapper: OptimizerWrapper) -> Dict[str, torch.Tensor]:
        """Implement the default model parameter update process。

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
        - call ``optimizer_wrapper.update_params(loss)`` to update model.

        Args:
            data (List[dict]): Data sampled from dataloader.
            optimizer_wrapper (OptimizerWrapper): OptimizerWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: dict of tensor for logging.
        """
        inputs, data_sample = self.data_preprocessor(data, True)
        losses = self(inputs, data_sample, mode='loss')
        parsed_losses, log_vars = self.parse_losses(losses)
        optimizer_wrapper.update_params(parsed_losses)
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

    def to(self, device, *args, **kwargs) -> nn.Module:
        """Override this method to set the ``device`` attribute of
        :obj:`BaseDataPreprocessor` additionally

        Returns:
            nn.Module: The model itself.
        """
        self.data_preprocessor.device = torch.device(device)
        return super(BaseModel, self).to(device)

    def cuda(self, *args, **kwargs) -> nn.Module:
        """Override this method to set the ``device`` attribute of
        :obj:`BaseDataPreprocessor` additionally

        Returns:
            nn.Module: The model itself.
        """
        self.data_preprocessor.device = torch.cuda.current_device()
        return super(BaseModel, self).cuda()

