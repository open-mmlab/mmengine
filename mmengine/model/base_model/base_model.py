from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import mmengine.optim.optimizer.optimizer_wrapper as optim_wrapper
from mmengine.data import InstanceData


class BaseModel(nn.Module):

    def __init__(
        self,
        preprocess_cfg: Optional[dict] = None,
        feat_keys: Optional[tuple] = None,
    ):
        super().__init__()
        if preprocess_cfg is None:
            preprocess_cfg = dict()
        self.to_rgb = preprocess_cfg.get('to_rgb', False)
        self._fp16_enabled = False
        pixel_mean = preprocess_cfg.get('pixel_mean', [127.5, 127.5, 127.5])
        pixel_std = preprocess_cfg.get('pixel_std', [127.5, 127.5, 127.5])
        self.register_buffer("pixel_mean",
                             torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std",
                             torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.size_divisor = preprocess_cfg.get('size_divisor', 1)

        self._feats_dict = dict()
        if feat_keys:
            for feat_key in feat_keys:
                self.feature_dict[feat_key] = None

    def train_step(
        self, data: List[dict],
        optimizer_wrapper: optim_wrapper._BaseOptimizerWrapper
    ) -> Dict[str, torch.Tensor]:
        inputs, data_sample = self.preprocess(data)
        losses = self(inputs, data_sample, return_loss=True)
        parsed_losses, log_vars = self._parse_losses(losses)
        optimizer_wrapper.optimizer_step(parsed_losses)
        return log_vars

    def val_step(self, data: List[dict], return_loss) -> List[InstanceData]:
        inputs, data_sample = self.preprocess(data)
        return self(inputs, data_sample, return_loss=return_loss)

    def test_step(self, data: List[dict]) -> List[InstanceData]:
        inputs, data_sample = self.preprocess(data)
        return self(inputs, data_sample, return_loss=False)

    def preprocess(self, data: Union[List[dict], torch.Tensor]):
        inputs: List[torch.Tensor] = []
        data_samples: List[InstanceData] = []
        if isinstance(data, list):
            for item in data:
                inputs.append(item['inputs'])
                data_samples.append(item['data_sample'].to(
                    self.pixel_mean.device))
        else:
            # data should be HWC format image.
            assert isinstance(data, torch.Tensor)
            assert len(data.shape) == 3
            inputs = data.unsqueeze(0)
            data_samples = None

        image_size = [torch.as_tensor(input.shape[-2:]) for input in inputs]
        max_size = torch.stack(image_size).max(0).values
        max_h, max_w = (max_size + (self.size_divisor - 1)).div(
            self.size_divisor, rounding_mode="floor") * self.size_divisor
        for idx, size in enumerate(image_size):
            h, w = size
            pad_h, pad_w = max_h - h, max_w - w
            inputs[idx] = (
                (inputs[idx].to(self.pixel_mean.device) - self.pixel_mean) /
                self.pixel_std)
            inputs[idx] = F.pad(inputs[idx], (0, pad_w, 0, pad_h))
        if self.to_rgb:
            assert input.shape[1] == 3
            inputs = inputs[:, [2, 1, 0], ...]
        return inputs, data_samples

    def _parse_losses(self, losses):
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

    def register_get_feature_hook(self):

        def get_feat_hook(self, module_name, module, inputs, outputs):
            self.feature_dict[module_name] = outputs

        for module_keys in self.feature_dict.keys():
            assert module_keys in self._modules
            get_feat_hook_fn = partial(get_feat_hook, self, module_keys)
            getattr(self, module_keys).register_forward_hook(get_feat_hook_fn)

    @property
    def feats_dict(self):
        return self._feats_dict

    def forward_trace(self):
        """_summary_
        """

    def forward(self):
        """_summary_
        """
