# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmengine.data import LabelData


class TestLabelData(TestCase):

    def test_label_to_onehot(self):
        item = torch.tensor([1], dtype=torch.int64)
        num_classes = 10
        onehot = LabelData.label_to_onehot(item=item, num_classes=num_classes)
        assert tuple(onehot.shape) == (num_classes, )

        # item is not onehot
        with self.assertRaises(AssertionError):
            LabelData.label_to_onehot(item='item', num_classes=num_classes)

        # item'max bigger than num_classes
        with self.assertRaises(AssertionError):
            LabelData.label_to_onehot(
                torch.tensor([11], dtype=torch.int64), num_classes)

    def test_onehot_to_label(self):
        # item is not onehot
        with self.assertRaises(
                ValueError,
                msg='`item` is not onehot and can not convert to label'):
            LabelData.onehot_to_label(
                item=torch.tensor([2], dtype=torch.int64))

        with self.assertRaises(AssertionError):
            LabelData.onehot_to_label(item='item')

        item = torch.arange(0, 9)
        onehot = LabelData.label_to_onehot(item=item, num_classes=10)
        label = LabelData.onehot_to_label(onehot)
        assert (label == item).all()
