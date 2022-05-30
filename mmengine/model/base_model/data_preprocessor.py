from typing import Optional, Sequence, Tuple, List, Union

import torch.nn as nn
import torch

from mmengine.registry import MODELS
from mmengine.utils import stack_batch
from mmengine.data import BaseDataElement


@MODELS.register_module()
class BaseDataPreprocessor(nn.Module):
    """Base data pre-processor used for collating and moving data to the
    target device.

    ``BaseDataPreprocessor`` perform data pre-processing according to the
    following steps:

    - Take the data sampled from dataloader and unpack them into list of tensor
      and list of labels.
    - Moving data to the target device.
    - Stack the input tensor at the first dimension.

    Subclass inherit ``BaseDataPreprocessor`` could override the forward method
    to implement custom data pre-processing, such as batch-resize, MixUp and
    CutMix.

    Warnings:
        Each item of data sampled from dataloader must a dict and at least
        contain the ``inputs`` key. Furthermore, the value of ``inputs``
        must be a ``Tensor`` with the same shape.

    Args:
        preprocess_cfg (dict, optional): The configuration of pre-processing
            data. The base class does not need to be configured with
            it.
        device (str): the target device, Defaults to 'auto'.

            - 'auto': If cuda is available, the device is the current cuda
               device; otherwise device will be cpu.
            - other string: Specific target device.
    """
    def __init__(self,
                 preprocess_cfg: Optional[dict] = None):
        super().__init__()
        self.device = 'cpu'

    def collate_data(self, data: Sequence[dict]
                     ) -> Tuple[List[torch.Tensor], list]:
        """Collating and moving data to the target device.

        Take the data sampled from dataloader and unpack them into list of
        tensor and list of labels. Then moving tensor to the target device.

        Subclass could override it to be compatible with the custom format
        data sampled from custom dataloader.

        Args:
            data (Sequence[dict]): data sampled from dataloader.

        Returns:
            Tuple[List[torch.Tensor], list]: Unstacked list of input tensor
            and list of labels at target device.
        """
        inputs = [data_['inputs'] for data_ in data]
        batch_data_samples: List[BaseDataElement] = []
        # Allow no `data_samples` in data
        for data_ in data:
            if 'data_sample' in data_:
                batch_data_samples.append(data_['data_sample'])

        # Move data from CPU to corresponding device.
        batch_data_samples = [
            self.cast_device(data_sample) for data_sample in
            batch_data_samples]
        inputs = [_input.to(self.cast_device(_input)) for _input in inputs]
        return inputs, batch_data_samples

    def cast_device(self, data: Union[torch.Tensor, BaseDataElement]
                    ) -> Union[torch.Tensor, BaseDataElement]:
        """Moving data to target device.

        Args:
            data (torch.Tensor or BaseDataElement): unpacked data sampled from
                dataloader.

        Returns:
            torch.Tensor or BaseDataElement: data at target device.
        """
        return data.to(self.device)

    def forward(self, data: Sequence[dict], training: bool = False
                ) -> Tuple[torch.Tensor, list]:
        """Pre-process the data into the model input format.

        After the data pre-processing of :meth:`collate_data`, ``forward``
        will stack the input tensor list to a batch tensor at the first
        dimension.

        Args:
            data (Sequence[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Tuple[torch.Tensor, list]: Data in the same format as the model
            input.
        """
        inputs, batch_data_samples = self.collate_data(data)
        batch_inputs = torch.stack(inputs, dim=0)
        return batch_inputs, batch_data_samples


@MODELS.register_module()
class ImgDataPreprocessor(BaseDataPreprocessor):
    """Image pre-processor for normalization and bgr to rgb conversion.

    Accept the data sampled by the dataLoader, and preprocesses it into the
    format of the model input. ``ImgDataPreprocessor`` provides the
    basic data pre-processing as follows

    - Collate and move data to the target device.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.

    For ``ImgDataPreprocessor``, the dimension of the single inputs must be
    (3, H, W).

    Note:
        ``ImgDataPreprocessor`` and its subclass is built in the
        constructor of :class:`BaseDataset`.

    Args:
        preprocess_cfg (dict, optional): The config of `to_rgb`, `pixel_mean`,
            `pixel_mean`, ``pad_size_divisor`` and ``pad_value``.
    """
    def __init__(self,
                 preprocess_cfg: Optional[dict] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if preprocess_cfg is None:
            preprocess_cfg = dict()
        self.to_rgb = preprocess_cfg.get('to_rgb', False)
        pixel_mean = preprocess_cfg.get('pixel_mean', [127.5, 127.5, 127.5])
        pixel_std = preprocess_cfg.get('pixel_std', [127.5, 127.5, 127.5])
        self.register_buffer('pixel_mean',
                             torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std',
                             torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.pad_size_divisor = preprocess_cfg.get('pad_size_divisor', 1)
        self.pad_value = preprocess_cfg.get('pad_value', 0)

    def forward(self, data: Sequence[dict], training: bool = False
                ) -> Tuple[torch.Tensor, list]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (Sequence[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Tuple[torch.Tensor, list]: Data in the same format as the model
            input.
        """
        inputs, batch_data_samples = self.collate_data(data)

        if self.to_rgb and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]
        inputs = [(_input - self.pixel_mean) / self.pixel_std
                  for _input in inputs]
        batch_inputs = stack_batch(inputs, self.pad_size_divisor,
                                   self.pad_value)
        return batch_inputs, batch_data_samples
