# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch

from mmengine.data import LabelData


class TestLabelData(TestCase):

    def test_set_data(self):
        label_data = LabelData(
            item=torch.IntTensor([5]), metainfo=dict(num_classes=10))
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

        # item is not onehot
        with pytest.raises(AssertionError):
            label_data.item = 'item'
            label_data.set_metainfo(dict(num_classes=10))
            label_data.to_onehot()

        # not set num_classes
        with pytest.raises(AssertionError):
            label_data = LabelData()
            label_data.item = item
            label_data.to_onehot()

        # num_classes not in metainfo
        with pytest.raises(AssertionError):
            label_data = LabelData()
            label_data.item = item
            label_data.num_classes = 10
            label_data.to_onehot()

        label_data = LabelData()
        label_data.item = item
        label_data.set_metainfo(dict(num_classes=10))
        label_data.to_onehot()
        assert tuple(label_data.item.shape) == (10, )

        label_data.item = item
        label_data.to_onehot(11)
        assert tuple(label_data.item.shape) == (11, )

        item = torch.arange(0, 9)
        label_data.item = item
        label_data.to_onehot(10)
        assert tuple(label_data.item.shape) == (10, )

    def test_to_label(self):
        label_data = LabelData()
        item = torch.tensor([1], dtype=torch.int64)

        # without num_classes
        with pytest.raises(AssertionError):
            label_data.to_label()

        # item is None
        with pytest.raises(AssertionError):
            label_data.set_metainfo(dict(num_classes=10))
            label_data.to_label()

        # item is not onehot
        with pytest.raises(
                ValueError,
                match='`item` is not onehot and can not convert to label'):
            label_data.item = item
            label_data.set_metainfo(dict(num_classes=10))
            label_data.to_label()

        # item is not onehot
        with pytest.raises(AssertionError):
            label_data.item = 'item'
            label_data.set_metainfo(dict(num_classes=10))
            label_data.to_label()

        item = torch.arange(0, 9)
        label_data.item = item
        label_data.to_onehot(10)
        label_data.to_label()
        assert (label_data.item == item).all()
