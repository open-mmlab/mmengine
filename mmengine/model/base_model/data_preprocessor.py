# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn

from mmengine.data import BaseDataElement
from mmengine.registry import MODELS
from mmengine.utils import stack_batch


@MODELS.register_module()
class BaseDataPreprocessor(nn.Module):
    """Base data pre-processor used for collating and moving data to the target
    device.

    ``BaseDataPreprocessor`` perform data pre-processing according to the
    following steps:

    - Take the data sampled from dataloader and unpack them into list of tensor
      and list of labels.
    - Move data to the target device.
    - Stack the input tensor at the first dimension.

    Subclass inherit ``BaseDataPreprocessor`` could override the forward method
    to implement custom data pre-processing, such as batch-resize, MixUp and
    CutMix.

    Warnings:
        Each item of data sampled from dataloader must a dict and at least
        contain the ``inputs`` key. Furthermore, the value of ``inputs``
        must be a ``Tensor`` with the same shape.
    """

    def __init__(self):
        super().__init__()
        self.device = 'cpu'

    def collate_data(self,
                     data: Sequence[dict]) -> Tuple[List[torch.Tensor], list]:
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
            self.cast_device(data_sample) for data_sample in batch_data_samples
        ]
        inputs = [_input.to(self.cast_device(_input)) for _input in inputs]
        return inputs, batch_data_samples

    def cast_device(
        self, data: Union[torch.Tensor, BaseDataElement]
    ) -> Union[torch.Tensor, BaseDataElement]:
        """Moving data to target device.

        Args:
            data (torch.Tensor or BaseDataElement): unpacked data sampled from
                dataloader.

        Returns:
            torch.Tensor or BaseDataElement: data at target device.
        """
        return data.to(self.device)

    def forward(self,
                data: Sequence[dict],
                training: bool = False) -> Tuple[torch.Tensor, list]:
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
        mean (Sequence[float or int]): The pixel mean of R, G, B channels.
            Defaults to (127.5, 127.5, 127.5).
        std (Sequence[float or int]): The pixel standard deviation of R, G, B
            channels. (127.5, 127.5, 127.5)
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False
    """

    def __init__(self,
                 mean: Sequence[Union[float, int]] = (127.5, 127.5, 127.5),
                 std: Sequence[Union[float, int]] = (127.5, 127.5, 127.5),
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 to_rgb: bool = False):
        super().__init__()
        self.register_buffer('pixel_mean',
                             torch.tensor(mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std',
                             torch.tensor(std).view(-1, 1, 1), False)
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.to_rgb = to_rgb

    def forward(self,
                data: Sequence[dict],
                training: bool = False) -> Tuple[torch.Tensor, list]:
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
