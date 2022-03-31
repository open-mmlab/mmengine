# Copyright (c) OpenMMLab. All rights reserved.
import random
from unittest import TestCase

import numpy as np
import pytest
import torch

from mmengine.data import BaseDataElement, InstanceData


class TestInstanceData(TestCase):

    def setup_data(self):
        metainfo = dict(
            img_id=random.randint(0, 100),
            img_shape=(random.randint(400, 600), random.randint(400, 600)))
        instances_infos = [1] * 5
        bboxes = torch.rand((5, 4))
        labels = np.random.rand(5)
        instance_data = InstanceData(
            metainfo=metainfo,
            bboxes=bboxes,
            labels=labels,
            instances_infos=instances_infos)
        return instance_data

    def test_set_data(self):
        instance_data = self.setup_data()

        # test set '_metainfo_fields' or '_data_fields'
        with self.assertRaises(AttributeError):
            instance_data._metainfo_fields = 1
        with self.assertRaises(AttributeError):
            instance_data._data_fields = 1

        # value only supports (torch.Tensor, np.ndarray, list)
        with self.assertRaises(AssertionError):
            instance_data.v = 'value'

        # The data length in InstanceData must be the same
        with self.assertRaises(AssertionError):
            instance_data.keypoints = torch.rand((17, 2))

        instance_data.keypoints = torch.rand((5, 2))
        assert 'keypoints' in instance_data

    def test_getitem(self):
        instance_data = InstanceData()
        # length must be greater than 0
        with self.assertRaises(AssertionError):
            instance_data[1]

        instance_data = self.setup_data()
        assert len(instance_data) == 5
        slice_instance_data = instance_data[:2]
        assert len(slice_instance_data) == 2

        # assert the index should in 0 ~ len(instance_data) -1
        with pytest.raises(IndexError):
            instance_data[5]

        # isinstance(str, slice, int, torch.LongTensor, torch.BoolTensor)
        item = torch.Tensor([1, 2, 3, 4])  # float
        with pytest.raises(AssertionError):
            instance_data[item]

        # when input is a bool tensor, The shape of
        # the input at index 0 should equal to
        # the value length in instance_data_field
        with pytest.raises(AssertionError):
            instance_data[item.bool()]

        # test Longtensor
        long_tensor = torch.randint(5, (2, ))
        long_index_instance_data = instance_data[long_tensor]
        assert len(long_index_instance_data) == len(long_tensor)

        # test bool tensor
        bool_tensor = torch.rand(5) > 0.5
        bool_index_instance_data = instance_data[bool_tensor]
        assert len(bool_index_instance_data) == bool_tensor.sum()

    def test_cat(self):
        instance_data_1 = self.setup_data()
        instance_data_2 = self.setup_data()
        cat_instance_data = InstanceData.cat(
            [instance_data_1, instance_data_2])
        assert len(cat_instance_data) == 10

        # All inputs must be InstanceData
        instance_data_2 = BaseDataElement(
            bboxes=torch.rand((5, 4)), labels=torch.rand((5, )))
        with self.assertRaises(AssertionError):
            InstanceData.cat([instance_data_1, instance_data_2])

        # Input List length must be greater than 0
        with self.assertRaises(AssertionError):
            InstanceData.cat([])

    def test_len(self):
        instance_data = self.setup_data()
        assert len(instance_data) == 5
        instance_data = InstanceData()
        assert len(instance_data) == 0
