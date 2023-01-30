# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import random
from unittest import TestCase

import numpy as np
import pytest
import torch

from mmengine.structures import BaseDataElement, InstanceData


class TmpObject:

    def __init__(self, tmp) -> None:
        assert isinstance(tmp, list)
        if len(tmp) > 0:
            for t in tmp:
                assert isinstance(t, list)
        self.tmp = tmp

    def __len__(self):
        return len(self.tmp)

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= len(self) or item < -len(self):  # type:ignore
                raise IndexError(f'Index {item} out of range!')
            else:
                # keep the dimension
                item = slice(item, None, len(self))
        return TmpObject(self.tmp[item])

    @staticmethod
    def cat(tmp_objs):
        assert all(isinstance(results, TmpObject) for results in tmp_objs)
        if len(tmp_objs) == 1:
            return tmp_objs[0]
        tmp_list = [tmp_obj.tmp for tmp_obj in tmp_objs]
        tmp_list = list(itertools.chain(*tmp_list))
        new_data = TmpObject(tmp_list)
        return new_data

    def __repr__(self):
        return str(self.tmp)


class TmpObjectWithoutCat:

    def __init__(self, tmp) -> None:
        assert isinstance(tmp, list)
        if len(tmp) > 0:
            for t in tmp:
                assert isinstance(t, list)
        self.tmp = tmp

    def __len__(self):
        return len(self.tmp)

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= len(self) or item < -len(self):  # type:ignore
                raise IndexError(f'Index {item} out of range!')
            else:
                # keep the dimension
                item = slice(item, None, len(self))
        return TmpObjectWithoutCat(self.tmp[item])

    def __repr__(self):
        return str(self.tmp)


class TestInstanceData(TestCase):

    def setup_data(self):
        metainfo = dict(
            img_id=random.randint(0, 100),
            img_shape=(random.randint(400, 600), random.randint(400, 600)))
        instances_infos = [1] * 5
        bboxes = torch.rand((5, 4))
        labels = np.random.rand(5)
        kps = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
        ids = (1, 2, 3, 4, 5)
        name_ids = '12345'
        polygons = TmpObject(np.arange(25).reshape((5, -1)).tolist())
        instance_data = InstanceData(
            metainfo=metainfo,
            bboxes=bboxes,
            labels=labels,
            polygons=polygons,
            kps=kps,
            ids=ids,
            name_ids=name_ids,
            instances_infos=instances_infos)
        return instance_data

    def test_set_data(self):
        instance_data = self.setup_data()

        # test set '_metainfo_fields' or '_data_fields'
        with self.assertRaises(AttributeError):
            instance_data._metainfo_fields = 1
        with self.assertRaises(AttributeError):
            instance_data._data_fields = 1

        # The data length in InstanceData must be the same
        with self.assertRaises(AssertionError):
            instance_data.keypoints = torch.rand((17, 2))

        instance_data.keypoints = torch.rand((5, 2))
        assert 'keypoints' in instance_data

    def test_getitem(self):
        instance_data = InstanceData()
        # length must be greater than 0
        with self.assertRaises(IndexError):
            instance_data[1]

        instance_data = self.setup_data()
        assert len(instance_data) == 5
        slice_instance_data = instance_data[:2]
        assert len(slice_instance_data) == 2
        slice_instance_data = instance_data[1]
        assert len(slice_instance_data) == 1
        # assert the index should in 0 ~ len(instance_data) -1
        with pytest.raises(IndexError):
            instance_data[5]

        # isinstance(str, slice, int, torch.LongTensor, torch.BoolTensor)
        item = torch.Tensor([1, 2, 3, 4])  # float
        with pytest.raises(AssertionError):
            instance_data[item]

        # when input is a bool tensor, the shape of
        # the input at index 0 should equal to
        # the value length in instance_data_field
        with pytest.raises(AssertionError):
            instance_data[item.bool()]

        # test LongTensor
        long_tensor = torch.randint(5, (2, ))
        long_index_instance_data = instance_data[long_tensor]
        assert len(long_index_instance_data) == len(long_tensor)

        # test BoolTensor
        bool_tensor = torch.rand(5) > 0.5
        bool_index_instance_data = instance_data[bool_tensor]
        assert len(bool_index_instance_data) == bool_tensor.sum()
        bool_tensor = torch.rand(5) > 1
        empty_instance_data = instance_data[bool_tensor]
        assert len(empty_instance_data) == bool_tensor.sum()

        # test list index
        list_index = [1, 2]
        list_index_instance_data = instance_data[list_index]
        assert len(list_index_instance_data) == len(list_index)

        # test list bool
        list_bool = [True, False, True, False, False]
        list_bool_instance_data = instance_data[list_bool]
        assert len(list_bool_instance_data) == 2

        # test numpy
        long_numpy = np.random.randint(5, size=2)
        long_numpy_instance_data = instance_data[long_numpy]
        assert len(long_numpy_instance_data) == len(long_numpy)

        bool_numpy = np.random.rand(5) > 0.5
        bool_numpy_instance_data = instance_data[bool_numpy]
        assert len(bool_numpy_instance_data) == bool_numpy.sum()

        # without cat
        instance_data.polygons = TmpObjectWithoutCat(
            np.arange(25).reshape((5, -1)).tolist())
        bool_numpy = np.random.rand(5) > 0.5
        with pytest.raises(
                ValueError,
                match=('The type of `polygons` is '
                       f'`{type(instance_data.polygons)}`, '
                       'which has no attribute of `cat`, so it does not '
                       f'support slice with `bool`')):
            bool_numpy_instance_data = instance_data[bool_numpy]

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
        instance_data_2 = instance_data_1.clone()
        instance_data_2 = instance_data_2[torch.zeros(5) > 0.5]
        cat_instance_data = InstanceData.cat(
            [instance_data_1, instance_data_2])
        cat_instance_data = InstanceData.cat([instance_data_1])
        assert len(cat_instance_data) == 5

        # test custom data cat
        instance_data_1.polygons = TmpObjectWithoutCat(
            np.arange(25).reshape((5, -1)).tolist())
        instance_data_2 = instance_data_1.clone()
        with pytest.raises(
                ValueError,
                match=('The type of `polygons` is '
                       f'`{type(instance_data_1.polygons)}` '
                       'which has no attribute of `cat`')):
            cat_instance_data = InstanceData.cat(
                [instance_data_1, instance_data_2])

    def test_len(self):
        instance_data = self.setup_data()
        assert len(instance_data) == 5
        instance_data = InstanceData()
        assert len(instance_data) == 0
