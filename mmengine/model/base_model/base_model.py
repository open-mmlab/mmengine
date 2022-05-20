# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmengine.data import InstanceData
from mmengine.optim import OptimizerWrapper
from mmengine.utils import stack_batch


class BaseModel(nn.Module):

    def __init__(
        self,
        preprocess_cfg: Optional[dict] = None,
        feat_keys: Optional[tuple] = None,
    ):
        super().__init__()
        self.parse_preprocess_cfg(preprocess_cfg)
        self.parse_feat_keys(feat_keys)

    def train_step(
            self, data: List[dict],
            optimizer_wrapper: OptimizerWrapper) -> Dict[str, torch.Tensor]:
        """_summary_

        Args:
            data (List[dict]): _description_
            optimizer_wrapper (OptimizerWrapper): _description_

        Returns:
            Dict[str, torch.Tensor]: _description_
        """
        inputs, data_sample = self.preprocess(data, True)
        losses = self(inputs, data_sample, return_loss=True)
        parsed_losses, log_vars = self._parse_losses(losses)
        optimizer_wrapper.optimizer_step(parsed_losses)
        return log_vars

    def val_step(self, data: List[dict], return_loss) -> List[InstanceData]:
        """_summary_

        Args:
            data (List[dict]): _description_
            return_loss (_type_): _description_

        Returns:
            List[InstanceData]: _description_
        """
        inputs, data_sample = self.preprocess(data, False)
        return self(inputs, data_sample, return_loss=return_loss)

    def test_step(self, data: List[dict]) -> List[InstanceData]:
        """_summary_

        Args:
            data (List[dict]): _description_

        Returns:
            List[InstanceData]: _description_
        """
        inputs, data_sample = self.preprocess(data, False)
        return self(inputs, data_sample, return_loss=False)

    def preprocess(self, data: Union[List[dict], torch.Tensor],
                   training: bool) -> Tuple[torch.Tensor, List[InstanceData]]:
        """_summary_

        Args:
            data (Union[List[dict], torch.Tensor]): _description_

        Returns:
            _type_: _description_
        """
        inputs = [data_['inputs'] for data_ in data]
        data_samples = [data_['data_sample'] for data_ in data]

        batch_data_samples = [
            data_sample.to(self.device) for data_sample in data_samples
        ]
        inputs = [_input.to(self.device) for _input in inputs]

        if self.preprocess_cfg is None:
            return stack_batch(inputs).float(), batch_data_samples

        if self.to_rgb and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]
        inputs = [(_input - self.pixel_mean) / self.pixel_std
                  for _input in inputs]
        batch_inputs = stack_batch(inputs, self.pad_size_divisor,
                                   self.pad_value)
        return batch_inputs, batch_data_samples

    def parse_preprocess_cfg(self, preprocess_cfg):
        if preprocess_cfg is None:
            preprocess_cfg = dict()
        self.preprocess_cfg = preprocess_cfg
        self.to_rgb = preprocess_cfg.get('to_rgb', False)
        pixel_mean = preprocess_cfg.get('pixel_mean', [127.5, 127.5, 127.5])
        pixel_std = preprocess_cfg.get('pixel_std', [127.5, 127.5, 127.5])
        self.register_buffer('pixel_mean',
                             torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std',
                             torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.pad_size_divisor = preprocess_cfg.get('pad_size_divisor', 1)
        self.pad_value = preprocess_cfg.get('pad_value', 0)

    def parse_feat_keys(self, feat_keys):
        self.feats_dict = dict()
        if feat_keys:
            for feat_key in feat_keys:
                self.feats_dict[feat_key] = None

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

    def register_get_feats_hook(self):
        """_summary_"""

        def get_feat_hook(self, module_name, module, inputs, outputs):
            self._feats_dict[module_name] = outputs

        for module_keys in self._feats_dict.keys():
            assert module_keys in self._modules
            get_feat_hook_fn = partial(get_feat_hook, self, module_keys)
            getattr(self, module_keys).register_forward_hook(get_feat_hook_fn)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, inputs: torch.Tensor,
                data_samples: List[InstanceData]) -> Dict[str, torch.Tensor]:
        """_summary_"""

    def forward_train(self,
                      inputs: torch.Tensor,
                      data_samples: List[InstanceData],
                      return_loss=False):
        """_summary_

        Args:
            inputs (torch.Tensor): _description_
            data_samples (List[InstanceData]): _description_
            return_loss (bool, optional): _description_. Defaults to False.
        """

    def predict(self, feats: torch.Tensor, data_samples: List[InstanceData]):
        """

        Args:
            feats:
            data_samples:

        Returns:

        """

    def loss(self, feats: torch.Tensor, data_samples: List[InstanceData]):
        """

        Args:
            feats:
            data_samples:

        Returns:

        """

    # TODO
    def aug_test(self):
        """

        Returns:

        """

    # TODO
    def preprocess_aug(self):
        """

        Returns:

        """
