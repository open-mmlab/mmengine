# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Sequence, Union

import numpy as np
import torch

from .base_data_element import BaseDataElement


class PixelData(BaseDataElement):
    """Data structure for pixel-level annnotations or predictions.

    All value in `data_fields` of `PixelData` have the following requirements:

    - All value must be 3-dimension.
    - The order of value's dimension is channel, height and width.
    - Values must have same height and width.
    """

    def __setattr__(self, name: str, value: Union[torch.Tensor, np.ndarray]):
        """if the dimension of value is 2 and its shape meet the demand, it
        will automatically expend its channel-dimension.

        Args:
            name (str): The key to access the value, stored in `PixelData`.
            value (Union[torch.Tensor, np.ndarray]): The value to store in.
                The type of value must be  `torch.Tensor` or `np.ndarray`,
                and its shape must meet the requirements of `PixelData`.
        """
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f'{name} has been used as a '
                    f'private attribute, which is immutable. ')

        else:
            assert isinstance(value, (torch.Tensor, np.ndarray)), \
                f'Can set {type(value)}, only support' \
                f' {(torch.Tensor, np.ndarray)}'

            if self.shape:
                assert tuple(value.shape[-2:]) == self.shape, (
                    f'the height and width of '
                    f'values {tuple(value.shape[-2:])} is '
                    f'not consistent with'
                    f' the length of this '
                    f':obj:`PixelData` '
                    f'{self.shape} ')
            assert value.ndim in [
                2, 3
            ], f'The dim of value must be 2 or 3, but got {value.ndim}'
            if value.ndim == 2:
                value = value[None]
                warnings.warn(f'The shape of value will convert from '
                              f'{value.shape[-2:]} to {value.shape}')
            super().__setattr__(name, value)

    # TODO torch.Long/bool
    def __getitem__(self, item: Sequence[Union[int, slice]]) -> 'PixelData':
        """
        Args:
            item (Sequence[Union[int, slice]]): get the corresponding values
            according to item.

        Returns:
            obj:`PixelData`: Corresponding values.
        """

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, tuple):

            assert len(item) == 2, 'Only support slice height and width'
            tmp_item: List[slice] = list()
            for index, single_item in enumerate(item[::-1]):
                if isinstance(single_item, int):
                    tmp_item.insert(
                        0, slice(single_item, None, self.shape[-index - 1]))
                elif isinstance(single_item, slice):
                    tmp_item.insert(0, single_item)
                else:
                    raise TypeError
            tmp_item.insert(0, slice(None, None, None))
            item = tuple(tmp_item)
            for k, v in self.items():
                setattr(new_data, k, v[item])
        else:
            raise TypeError
        return new_data

    @property
    def shape(self):
        """The shape of pixel data."""
        if len(self._data_fields) > 0:
            return tuple(self.values()[0].shape[-2:])
        else:
            return None

    # TODO padding, resize
