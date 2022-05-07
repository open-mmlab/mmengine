# Copyright (c) OpenMMLab. All rights reserved.

from typing import Union

import torch

from .base_data_element import BaseDataElement


class LabelData(BaseDataElement):
    """Data structure for label-level annnotations or predictions."""

    @property
    def num_classes(self) -> int:
        """int: the number of classes.
        """
        return self._num_classes  # type: ignore

    @num_classes.setter
    def num_classes(self, value) -> None:
        """set `num_classes`"""
        self.set_field(value, '_num_classes', dtype=int, field_type='metainfo')

    @property
    def item(self) -> Union[torch.TensorType, str]:
        """(tensor.Tensor, str): the item of label."""
        return self._item  # type: ignore

    @item.setter
    def item(self, value) -> None:
        """set `item`"""
        self.set_field(value, '_item', dtype=(torch.Tensor, str))

    @item.deleter
    def item(self) -> None:
        """delete `item`"""
        del self._item  # type: ignore

    def to_onehot(self) -> None:
        """convert `item` to onehot.

        The `num_classes` must be set.
        """
        assert 'item' in self, 'please set `item`'
        assert 'num_classes' in self, 'please set `num_classes`'
        item = torch.zeros((self.num_classes, ), dtype=torch.int64)
        item[self.item] = 1
        self.item = item

    def to_label(self) -> None:
        """Convert `item` to label if its type is onehot."""
        if (isinstance(self.item, torch.Tensor) and self.item.ndim == 1
                and self.item.shape[0] == self.num_classes
                and self.item.max().item() <= 1
                and self.item.min().item() >= 0):
            self.item = self.item.nonzero().squeeze()
        else:
            raise ValueError(
                '`item` is not onehot and can not convert to label')
