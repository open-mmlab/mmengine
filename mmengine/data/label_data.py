# Copyright (c) OpenMMLab. All rights reserved.

import torch

from .base_data_element import BaseDataElement


class LabelData(BaseDataElement):
    """Data structure for label-level annnotations or predictions."""

    @staticmethod
    def onehot_to_label(onehot: torch.Tensor) -> torch.Tensor:
        """Convert the one-hot input to label.

        Args:
            onehot (torch.Tensor, optional): The one-hot input. The format
                of input must be one-hot.

        Return:
            torch.Tensor: The converted results.
        """
        assert isinstance(onehot, torch.Tensor)
        if (onehot.ndim == 1 and onehot.max().item() <= 1
                and onehot.min().item() >= 0):
            return onehot.nonzero().squeeze()
        else:
            raise ValueError(
                'input is not one-hot and can not convert to label')

    @staticmethod
    def label_to_onehot(label: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Convert the label-format input to one-hot.

        Args:
            label (torch.Tensor): The label-format input. The format
                of item must be label-format.
            num_classes (int): The number of classes.

        Return:
            torch.Tensor: The converted results.
        """
        assert isinstance(label, torch.Tensor)
        onehot = torch.zeros((num_classes, ), dtype=torch.int64)
        assert label.max().item() < num_classes
        onehot[label] = 1
        return onehot
