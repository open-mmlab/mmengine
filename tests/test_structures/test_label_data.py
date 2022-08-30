# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch

from mmengine.structures import LabelData


class TestLabelData(TestCase):

    def test_label_to_onehot(self):
        item = torch.tensor([1], dtype=torch.int64)
        num_classes = 10
        onehot = LabelData.label_to_onehot(label=item, num_classes=num_classes)
        assert tuple(onehot.shape) == (num_classes, )
        assert onehot.device == item.device
        # item is not onehot
        with self.assertRaises(AssertionError):
            LabelData.label_to_onehot(label='item', num_classes=num_classes)

        # item'max bigger than num_classes
        with self.assertRaises(AssertionError):
            LabelData.label_to_onehot(
                torch.tensor([11], dtype=torch.int64), num_classes)
        onehot = LabelData.label_to_onehot(
            label=torch.tensor([], dtype=torch.int64), num_classes=num_classes)
        assert (onehot == torch.zeros((num_classes, ),
                                      dtype=torch.int64)).all()

    def test_onehot_to_label(self):
        # item is not onehot
        with self.assertRaisesRegex(
                ValueError,
                'input is not one-hot and can not convert to label'):
            LabelData.onehot_to_label(
                onehot=torch.tensor([2], dtype=torch.int64))

        with self.assertRaises(AssertionError):
            LabelData.onehot_to_label(onehot='item')

        item = torch.arange(0, 9)
        onehot = LabelData.label_to_onehot(item, num_classes=10)
        label = LabelData.onehot_to_label(onehot)
        assert (label == item).all()
        assert label.device == item.device
        item = torch.tensor([2])
        onehot = LabelData.label_to_onehot(item, num_classes=10)
        label = LabelData.onehot_to_label(onehot)
        assert label == item
        assert label.device == item.device

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='GPU is required!')
    def test_cuda(self):
        item = torch.arange(0, 9).cuda()
        onehot = LabelData.label_to_onehot(item, num_classes=10)
        assert item.device == onehot.device
        label = LabelData.onehot_to_label(onehot)
        assert label.device == onehot.device
