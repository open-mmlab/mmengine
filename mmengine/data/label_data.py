# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional, Union

import torch

from .base_data_element import BaseDataElement


class LabelData(BaseDataElement):
    """Data structure for label-level annnotations or predictions.

    ``LabelData`` set ``item`` as default attributes. The type of ``item`` must
    be ``str`` or ``torch.Tensor``. ``LabelData`` also support ``to_onehot``
    and ``to_label`` when the type of ``item`` is ``torch.Tensor``.
    """

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

    def to_onehot(self, num_classes: Optional[int] = None) -> None:
        """convert `item` to onehot in place. If num_classes is None,
        `num_classes` must in `LabelData.metainfo_keys()`

        Args:
            num_classes (int, optional): The number of classes. Defaults
                to None.
        """
        if num_classes is None:
            assert 'num_classes' in self.metainfo_keys(), (
                'Please provide `num_classes` in metainfo at first')
            num_classes = self.num_classes  # type: ignore
        assert 'item' in self, 'Please set `item` in `LabelData` at first'
        self.item = self.get_onehot(self.item, num_classes=num_classes)

    def get_onehot(self,
                   item: Optional[torch.Tensor] = None,
                   num_classes: Optional[int] = None) -> torch.Tensor:
        """Convert `item` to onehot, when `item` is not None, Otherwise convert
        `self.item` to onehot. If `item` is None, `item` must be set in
        `LabelData` at first. If num_classes is None, `num_classes` must be in
        `LabelData.metainfo_keys()`

        Args:
            item (torch.Tensor, optional): The value to convert to onehot.
                Defaults to None.
            num_classes (int, optional): The number of classes. Defaults
                to None.

        Return:
            (torch.Tensor): the onehot results.
        """
        if item is None:
            assert 'item' in self, 'Please set `item` in `LabelData` at first'
            item = self.item
        assert isinstance(item, torch.Tensor)
        if num_classes is None:
            assert 'num_classes' in self.metainfo_keys(), (
                'Please provide `num_classes` in metainfo at first '
                'or pass `num_classes` parameter')
            num_classes = self.num_classes  # type: ignore
        new_item = torch.zeros((num_classes, ), dtype=torch.int64)
        assert item.max().item() < num_classes
        new_item[item] = 1
        return new_item

    def to_label(self) -> None:
        """Convert `item` to label in place if its type is onehot."""
        assert 'item' in self, 'Please set `item` in `LabelData` at first'
        self.item = self.get_label(self.item)

    def get_label(self, item: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert `item` to label if its type is onehot.

        Args:
            item (torch.Tensor, optional): The value to convert to label.
                if it is None, it will convert `self.item` to label.
                Defaults to None.

        Return:
            (torch.Tensor): the label results.
        """
        if item is None:
            assert 'item' in self, 'Please set `item` in `LabelData` at first'
            item = self.item
        assert isinstance(item, torch.Tensor)
        if (item.ndim == 1 and item.max().item() <= 1
                and item.min().item() >= 0):
            return item.nonzero().squeeze()
        else:
            raise ValueError(
                '`item` is not onehot and can not convert to label')
