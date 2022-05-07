# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch

from mmengine.data import LabelData


class TestLabelData(TestCase):

    def test_set_data(self):
        label_data = LabelData(num_classes=10, item=torch.IntTensor([5]))
        assert label_data.num_classes == 10
        assert label_data.item == torch.IntTensor([5])
        assert 'item' in label_data
        del label_data.item
        assert 'item' not in label_data
        with pytest.raises(AssertionError):
            label_data.item = 1
        label_data.item = 'hello world'

    def test_to_onehot(self):
        label_data = LabelData()
        item = torch.tensor([1], dtype=torch.int64)
        # not set item
        with pytest.raises(AssertionError):
            label_data.to_onehot()

        # not set num_classes
        with pytest.raises(AssertionError):
            label_data.item = item
            label_data.to_onehot()
        label_data.item = item
        label_data.num_classes = 10
        label_data.to_onehot()
        assert tuple(label_data.item.shape) == (item.shape[0], 10)

        item = torch.arange(0, 9)
        label_data.item = item
        label_data.to_onehot()
        assert tuple(label_data.item.shape) == (item.shape[0], 10)

    def test_to_label(self):
        label_data = LabelData()
        item = torch.tensor([1], dtype=torch.int64)
        label_data.num_classes = 10

        # not set num_classes
        with pytest.raises(
                ValueError,
                match='item is not onehot and can not convert to label'):
            label_data.item = item
            label_data.to_label()

        item = torch.arange(0, 9)
        label_data.item = item
        label_data.to_onehot()
        label_data.to_label()
        assert (label_data.item == item).all()
