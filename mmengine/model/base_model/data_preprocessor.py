# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from mmengine.data import BaseDataElement
from mmengine.registry import MODELS
from ..utils import stack_batch


@MODELS.register_module()
class BaseDataPreprocessor(nn.Module):
    """Base data pre-processor used for collating and copying data to the
    target device.

    ``BaseDataPreprocessor`` performs data pre-processing according to the
    following steps:

    - Collates the data sampled from dataloader.
    - Copies data to the target device.
    - Stacks the input tensor at the first dimension.

    Subclasses inherit from ``BaseDataPreprocessor`` could override the
    forward method to implement custom data pre-processing, such as
    batch-resize, MixUp, or CutMix.

    Warnings:
        Each item of data sampled from dataloader must be a dict and at least
        contain the ``inputs`` key. Furthermore, the value of ``inputs``
        must be a ``Tensor`` with the same shape.
    """

    def __init__(self):
        super().__init__()
        self._device = torch.device('cpu')

    def collate_data(
            self,
            data: Sequence[dict]) -> Tuple[List[torch.Tensor], Optional[list]]:
        """Collating and copying data to the target device.

        Collates the data sampled from dataloader into a list of tensor and
        list of labels, and then copies tensor to the target device.

        Subclasses could override it to be compatible with the custom format
        data sampled from custom dataloader.

        Args:
            data (Sequence[dict]): Data sampled from dataloader.

        Returns:
            Tuple[List[torch.Tensor], Optional[list]]: Unstacked list of input
            tensor and list of labels at target device.
        """
        inputs = [_data['inputs'].to(self._device) for _data in data]
        batch_data_samples: List[BaseDataElement] = []
        # Model can get predictions without any data samples.
        for _data in data:
            if 'data_sample' in _data:
                batch_data_samples.append(_data['data_sample'])
        # Move data from CPU to corresponding device.
        batch_data_samples = [
            data_sample.to(self._device) for data_sample in batch_data_samples
        ]

        if not batch_data_samples:
            batch_data_samples = None  # type: ignore

        return inputs, batch_data_samples

    def forward(self,
                data: Sequence[dict],
                training: bool = False) -> Tuple[torch.Tensor, Optional[list]]:
        """Preprocesses the data into the model input format.

        After the data pre-processing of :meth:`collate_data`, ``forward``
        will stack the input tensor list to a batch tensor at the first
        dimension.

        Args:
            data (Sequence[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Tuple[torch.Tensor, Optional[list]]: Data in the same format as the
            model input.
        """
        inputs, batch_data_samples = self.collate_data(data)
        batch_inputs = torch.stack(inputs, dim=0)
        return batch_inputs, batch_data_samples

    @property
    def device(self):
        return self._device

    def to(self, device: Optional[Union[int, torch.device]], *args,
           **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Args:
            device (int or torch.device, optional): The desired device of the
                parameters and buffers in this module.

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(device)
        return super().to(device)

    def cuda(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.cuda.current_device())
        return super().cuda()

    def cpu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device('cpu')
        return super().cpu()


@MODELS.register_module()
class ImgDataPreprocessor(BaseDataPreprocessor):
    """Image pre-processor for normalization and bgr to rgb conversion.

    Accepts the data sampled by the dataloader, and preprocesses it into the
    format of the model input. ``ImgDataPreprocessor`` provides the
    basic data pre-processing as follows

    - Collates and moves data to the target device.
    - Converts inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalizes image with defined std and mean.
    - Pads inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.

    For ``ImgDataPreprocessor``, the dimension of the single inputs must be
    (3, H, W).

    Note:
        ``ImgDataPreprocessor`` and its subclass is built in the
        constructor of :class:`BaseDataset`.

    Args:
        mean (Sequence[float or int]): The pixel mean of image channels. If
            ``bgr_to_rgb=True`` it means the mean value of R, G, B channels.
            If ``mean`` and ``std`` are not specified, ``ImgDataPreprocessor``
            will normalize images to [-1, 1]. Defaults to (127.5, 127.5,
            127.5).
        std (Sequence[float or int]): The pixel standard deviation of image
            channels. If ``bgr_to_rgb=True`` it means the standard deviation of
            R, G, B channels. If ``mean`` and ``std`` are not specified,
            ImgDataPreprocessor will normalize images to [-1, 1]. Defaults
            to (127.5, 127.5, 127.5).
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
    """

    def __init__(self,
                 mean: Sequence[Union[float, int]] = (127.5, 127.5, 127.5),
                 std: Sequence[Union[float, int]] = (127.5, 127.5, 127.5),
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False):
        super().__init__()
        assert len(mean) == 3 or len(mean) == 1, (
            'The length of mean should be 1 or 3 to be compatible with RGB '
            f'or gray image, but got {len(mean)}')
        assert len(std) == 3 or len(std) == 1, (
            'The length of std should be 1 or 3 to be compatible with RGB '
            f'or gray image, but got {len(std)}')
        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb
        self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1), False)
        self.register_buffer('std', torch.tensor(std).view(-1, 1, 1), False)
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value

    def forward(self,
                data: Sequence[dict],
                training: bool = False) -> Tuple[torch.Tensor, Optional[list]]:
        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (Sequence[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.

        Returns:
            Tuple[torch.Tensor, Optional[list]]: Data in the same format as the
            model input.
        """
        inputs, batch_data_samples = self.collate_data(data)
        # channel transform
        if self.channel_conversion:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]
        # Normalization.
        inputs = [(_input - self.mean) / self.std for _input in inputs]
        # Pad and stack Tensor.
        batch_inputs = stack_batch(inputs, self.pad_size_divisor,
                                   self.pad_value)
        return batch_inputs, batch_data_samples
