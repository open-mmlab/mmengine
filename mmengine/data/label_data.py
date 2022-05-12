# Copyright (c) OpenMMLab. All rights reserved.

import torch

from .base_data_element import BaseDataElement


class LabelData(BaseDataElement):
    """Data structure for label-level annnotations or predictions."""

    @staticmethod
    def onehot_to_label(item: torch.Tensor) -> torch.Tensor:
        """Convert the one-hot input to label.

        Args:
            item (torch.Tensor, optional): The one-hot input. The format
                of item must be one-hot.

        Return:
            torch.Tensor: The converted results.
        """
        assert isinstance(item, torch.Tensor)
        if (item.ndim == 1 and item.max().item() <= 1
                and item.min().item() >= 0):
            return item.nonzero().squeeze()
        else:
            raise ValueError(
                '`item` is not onehot and can not convert to label')

    @staticmethod
    def label_to_onehot(item: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Convert the label-format input to one-hot.

        Args:
            item (torch.Tensor): The label-format input. The format
                of item must be label-format.
            num_classes (int): The number of classes.

        Return:
            torch.Tensor: The converted results.
        """
        assert isinstance(item, torch.Tensor)
        new_item = torch.zeros((num_classes, ), dtype=torch.int64)
        assert item.max().item() < num_classes
        new_item[item] = 1
        return new_item
