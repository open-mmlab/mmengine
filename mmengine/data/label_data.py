# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional, Union

import torch

# from .base_data_element import BaseDataElement
from mmengine.data import BaseDataElement


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
        """convert `item` to onehot. If num_classes is None, `num_classes` must
        in `LabelData.metainfo_keys()`

        Args:
            num_classes (Optional[int]): The number of classes.
        """
        assert 'item' in self and isinstance(self.item, torch.Tensor)
        if num_classes is not None:
            self.set_field(
                num_classes, 'num_classes', dtype=int, field_type='metainfo')
        assert 'num_classes' in self.metainfo_keys(), (
            'Please set `num_classes` with '
            '`.set_metainfo({num_classes=num_classes})` '
            'or pass `num_classes` parameter')
        item = torch.zeros(
            (self.num_classes, ),  # type: ignore
            dtype=torch.int64)
        assert self.item.max().item() < self.num_classes  # type: ignore
        item[self.item] = 1
        self.item = item

    def to_label(self) -> None:
        """Convert `item` to label if its type is onehot."""
        assert 'num_classes' in self.metainfo_keys(), (
            'Please set `num_classes` with '
            '`.set_metainfo({num_classes=num_classes})`')
        assert 'item' in self and isinstance(self.item, torch.Tensor)
        if (isinstance(self.item, torch.Tensor) and self.item.ndim == 1
                and self.item.shape[0] == self.num_classes  # type: ignore
                and self.item.max().item() <= 1
                and self.item.min().item() >= 0):
            self.item = self.item.nonzero().squeeze()
        else:
            raise ValueError(
                '`item` is not onehot and can not convert to label')
