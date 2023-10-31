# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Any, List, Sequence, Union

import numpy as np
import torch
from torch.nn.functional import interpolate, pad

from .base_data_element import BaseDataElement


class PixelData(BaseDataElement):
    """Data structure for pixel-level annotations or predictions.

    All data items in ``data_fields`` of ``PixelData`` meet the following
    requirements:

    - They all have 3 dimensions in orders of channel, height, and width.
    - They should have the same height and width.

    Examples:
        >>> metainfo = dict(
        ...     img_id=random.randint(0, 100),
        ...     img_shape=(random.randint(400, 600), random.randint(400, 600)))
        >>> image = np.random.randint(0, 255, (4, 20, 40))
        >>> featmap = torch.randint(0, 255, (10, 20, 40))
        >>> pixel_data = PixelData(metainfo=metainfo,
        ...                        image=image,
        ...                        featmap=featmap)
        >>> print(pixel_data.shape)
        (20, 40)

        >>> # slice
        >>> slice_data = pixel_data[10:20, 20:40]
        >>> assert slice_data.shape == (10, 20)
        >>> slice_data = pixel_data[10, 20]
        >>> assert slice_data.shape == (1, 1)

        >>> # set
        >>> pixel_data.map3 = torch.randint(0, 255, (20, 40))
        >>> assert tuple(pixel_data.map3.shape) == (1, 20, 40)
        >>> with self.assertRaises(AssertionError):
        ...     # The dimension must be 3 or 2
        ...     pixel_data.map2 = torch.randint(0, 255, (1, 3, 20, 40))
    """

    def __setattr__(self, name: str, value: Union[torch.Tensor, np.ndarray]):
        """Set attributes of ``PixelData``.

        If the dimension of value is 2 and its shape meet the demand, it
        will automatically expand its channel-dimension.

        Args:
            name (str): The key to access the value, stored in `PixelData`.
            value (Union[torch.Tensor, np.ndarray]): The value to store in.
                The type of value must be `torch.Tensor` or `np.ndarray`,
                and its shape must meet the requirements of `PixelData`.
        """
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable.')

        else:
            assert isinstance(value, (torch.Tensor, np.ndarray)), \
                f'Can not set {type(value)}, only support' \
                f' {(torch.Tensor, np.ndarray)}'

            if self.shape:
                assert tuple(value.shape[-2:]) == self.shape, (
                    'The height and width of '
                    f'values {tuple(value.shape[-2:])} is '
                    'not consistent with '
                    'the shape of this '
                    ':obj:`PixelData` '
                    f'{self.shape}')
            assert value.ndim in [
                2, 3
            ], f'The dim of value must be 2 or 3, but got {value.ndim}'
            if value.ndim == 2:
                value = value[None]
                warnings.warn('The shape of value will convert from '
                              f'{value.shape[-2:]} to {value.shape}')
            super().__setattr__(name, value)

    # TODO torch.Long/bool
    def __getitem__(self, item: Sequence[Union[int, slice]]) -> 'PixelData':
        """
        Args:
            item (Sequence[Union[int, slice]]): Get the corresponding values
                according to item.

        Returns:
            :obj:`PixelData`: Corresponding values.
        """

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, tuple):

            assert len(item) == 2, 'Only support to slice height and width'
            tmp_item: List[slice] = list()
            for index, single_item in enumerate(item[::-1]):
                if isinstance(single_item, int):
                    tmp_item.insert(
                        0, slice(single_item, None, self.shape[-index - 1]))
                elif isinstance(single_item, slice):
                    tmp_item.insert(0, single_item)
                else:
                    raise TypeError(
                        'The type of element in input must be int or slice, '
                        f'but got {type(single_item)}')
            tmp_item.insert(0, slice(None, None, None))
            item = tuple(tmp_item)
            for k, v in self.items():
                setattr(new_data, k, v[item])
        else:
            raise TypeError(
                f'Unsupported type {type(item)} for slicing PixelData')
        return new_data

    @property
    def shape(self):
        """The shape of pixel data."""
        if len(self._data_fields) > 0:
            return tuple(self.values()[0].shape[-2:])
        else:
            return None

    # TODO padding, resize
    def resize(self,
               size: Sequence[int],
               interpolation: str = 'bilinear') -> 'PixelData':
        """Resize all values to the given `size`, and return a new `PixelData`.

        Args:
            size (Sequence[int]): Output spatial size,
              should be (height, width)
            interpolation (str, optional): The algorithm used in interpolation.
              available for resizing are: `nearest`, `bilinear`, `bicubic`,
              `area`, `nearest-exact`. Defaults to 'bilinear'.

        Returns:
            PixelData: A resized new `PixelData`
        """
        assert len(size) == 2, 'Size should be (height, width)'
        new_h, new_w = size
        old_h, old_w = self.shape
        if new_h == old_h and new_w == old_w:
            return self.clone()
        new_data = self.__class__(metainfo=self.metainfo)
        for k, v in self.items():
            if isinstance(v, np.ndarray):
                data = torch.from_numpy(v)
            else:
                data = v
            # torch.nn.functional need a batch dim,
            # and do not support some dtype
            data = data.unsqueeze(0).to(torch.float32)
            resized_data = interpolate(
                data, size=size, mode=interpolation).squeeze()
            if isinstance(v, np.ndarray):
                resized_data = resized_data.numpy().astype(v.dtype)
            else:
                resized_data = resized_data.to(v.dtype)
            setattr(new_data, k, resized_data)
        return new_data

    def padding(self,
                pad_size: Sequence[int],
                mode: str = 'constant',
                value: Any = 0) -> 'PixelData':
        """Pad all values with the given `pad_size`, and return a new
        `PixelData`.

        Args:
            pad_size (Sequence[int]): The size need to pad.
              See detail in `torch.nn.functional.pad`.
              length is 2:
              (padding_left, padding_right),
              length is 4:
              (padding_left, padding_right, padding_top, padding_bottom)
              length is 6:
              (padding_left, padding_right,
                padding_top, padding_bottom, padding_front, padding_back)
            mode (str, optional): Padding mode.
              'constant', 'reflect', 'replicate' or 'circular'.
              Defaults to 'constant'.
            value (Any, optional): Fill value. Defaults to 0.

        Returns:
            PixelData: A Padded new `PixelData`
        """
        assert len(pad_size) in (2, 4,
                                 6), 'Pad size length should be 2, 4 or 6'
        if sum(pad_size) == 0:
            return self.clone()
        new_data = self.__class__(metainfo=self.metainfo)
        for k, v in self.items():
            if isinstance(v, np.ndarray):
                data = torch.from_numpy(v)
            else:
                data = v

            # some pad mode do not support some dtype
            data = data.to(torch.float32)
            pad_data = pad(data, pad=pad_size, mode=mode, value=value)
            if isinstance(v, np.ndarray):
                pad_data = pad_data.numpy().astype(v.dtype)
            else:
                pad_data = pad_data.to(v.dtype)
            setattr(new_data, k, pad_data)
        return new_data
