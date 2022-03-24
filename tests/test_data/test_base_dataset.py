# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from unittest.mock import MagicMock

import pytest
import torch

from mmengine.dataset import (BaseDataset, ClassBalancedDataset, Compose,
                              ConcatDataset, RepeatDataset, force_full_init)
from mmengine.registry import TRANSFORMS


def function_pipeline(data_info):
    return data_info


@TRANSFORMS.register_module()
class CallableTransform:

    def __call__(self, data_info):
        return data_info


@TRANSFORMS.register_module()
class NotCallableTransform:
    pass


class TestBaseDataset:
    dataset_type = BaseDataset
    data_info = dict(
        filename='test_img.jpg', height=604, width=640, sample_idx=0)
    imgs = torch.rand((2, 3, 32, 32))
    pipeline = MagicMock(return_value=dict(imgs=imgs))
    METAINFO: dict = dict()
    parse_data_info = MagicMock(return_value=data_info)

    def _init_dataset(self):
        self.dataset_type.METAINFO = self.METAINFO
        self.dataset_type.parse_data_info = self.parse_data_info

    def test_init(self):
        self._init_dataset()
        # test the instantiation of self.base_dataset
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')
        assert dataset._fully_initialized
        assert hasattr(dataset, 'data_list')
        assert hasattr(dataset, 'data_address')
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img=None),
            ann_file='annotations/dummy_annotation.json')
        assert dataset._fully_initialized
        assert hasattr(dataset, 'data_list')
        assert hasattr(dataset, 'data_address')

        # test the instantiation of self.base_dataset with
        # `serialize_data=False`
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            serialize_data=False)
        assert dataset._fully_initialized
        assert hasattr(dataset, 'data_list')
        assert not hasattr(dataset, 'data_address')
        assert len(dataset) == 2
        assert dataset.get_data_info(0) == self.data_info

        # test the instantiation of self.base_dataset with lazy init
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=True)
        assert not dataset._fully_initialized
        assert not dataset.data_list

        # test the instantiation of self.base_dataset if ann_file is not
        # existed.
        with pytest.raises(FileNotFoundError):
            self.dataset_type(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img='imgs'),
                ann_file='annotations/not_existed_annotation.json')

        # test the instantiation of self.base_dataset when the ann_file is
        # wrong
        with pytest.raises(ValueError):
            self.dataset_type(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img='imgs'),
                ann_file='annotations/annotation_wrong_keys.json')
        with pytest.raises(TypeError):
            self.dataset_type(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img='imgs'),
                ann_file='annotations/annotation_wrong_format.json')
        with pytest.raises(TypeError):
            self.dataset_type(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img=['img']),
                ann_file='annotations/annotation_wrong_format.json')

        # test the instantiation of self.base_dataset when `parse_data_info`
        # return `list[dict]`
        self.dataset_type.parse_data_info = MagicMock(
            return_value=[self.data_info,
                          self.data_info.copy()])
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')
        dataset.pipeline = self.pipeline
        assert dataset._fully_initialized
        assert hasattr(dataset, 'data_list')
        assert hasattr(dataset, 'data_address')
        assert len(dataset) == 4
        assert dataset[0] == dict(imgs=self.imgs)
        assert dataset.get_data_info(0) == self.data_info

        # test the instantiation of self.base_dataset when `parse_data_info`
        # return unsupported data.
        with pytest.raises(TypeError):
            self.dataset_type.parse_data_info = MagicMock(return_value='xxx')
            dataset = self.dataset_type(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img='imgs'),
                ann_file='annotations/dummy_annotation.json')
        with pytest.raises(TypeError):
            self.dataset_type.parse_data_info = MagicMock(
                return_value=[self.data_info, 'xxx'])
            dataset = self.dataset_type(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img='imgs'),
                ann_file='annotations/dummy_annotation.json')

    def test_metainfo(self):
        self._init_dataset()
        # test dataset.meta with setting the metainfo from annotation file as the
        # metainfo of self.base_dataset
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')

        assert dataset.meta == dict(
            dataset_type='test_dataset', task_name='test_task', empty_list=[])

        # test dataset.meta with setting METAINFO in self.base_dataset
        dataset_type = 'new_dataset'
        self.dataset_type.METAINFO = dict(
            dataset_type=dataset_type, classes=('dog', 'cat'))

        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')
        assert dataset.meta == dict(
            dataset_type=dataset_type,
            task_name='test_task',
            classes=('dog', 'cat'),
            empty_list=[])

        # test dataset.meta with passing metainfo into self.base_dataset
        metainfo = dict(classes=('dog', ), task_name='new_task')
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            metainfo=metainfo)
        assert self.dataset_type.METAINFO == dict(
            dataset_type=dataset_type, classes=('dog', 'cat'))
        assert dataset.meta == dict(
            dataset_type=dataset_type,
            task_name='new_task',
            classes=('dog', ),
            empty_list=[])
        # reset `base_dataset.METAINFO`, the `dataset.meta` should not change
        self.dataset_type.METAINFO['classes'] = ('dog', 'cat', 'fish')
        assert self.dataset_type.METAINFO == dict(
            dataset_type=dataset_type, classes=('dog', 'cat', 'fish'))
        assert dataset.meta == dict(
            dataset_type=dataset_type,
            task_name='new_task',
            classes=('dog', ),
            empty_list=[])

        # test dataset.meta with passing metainfo containing a file into
        # self.base_dataset
        metainfo = dict(
            classes=osp.join(
                osp.dirname(__file__), '../data/meta/classes.txt'))
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            metainfo=metainfo)
        assert dataset.meta == dict(
            dataset_type=dataset_type,
            task_name='test_task',
            classes=['dog'],
            empty_list=[])

        # test dataset.meta with passing unsupported metainfo into
        # self.base_dataset
        with pytest.raises(TypeError):
            metainfo = 'dog'
            dataset = self.dataset_type(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img='imgs'),
                ann_file='annotations/dummy_annotation.json',
                metainfo=metainfo)

        # test dataset.meta with passing metainfo into self.base_dataset and
        # lazy_init is True
        metainfo = dict(classes=('dog', ))
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            metainfo=metainfo,
            lazy_init=True)
        # 'task_name' and 'empty_list' not in dataset.meta
        assert dataset.meta == dict(
            dataset_type=dataset_type, classes=('dog', ))

        # test whether self.base_dataset.METAINFO is changed when a customize
        # dataset inherit self.base_dataset
        # test reset METAINFO in ToyDataset.
        class ToyDataset(self.dataset_type):
            METAINFO = dict(xxx='xxx')

        assert ToyDataset.METAINFO == dict(xxx='xxx')
        assert self.dataset_type.METAINFO == dict(
            dataset_type=dataset_type, classes=('dog', 'cat', 'fish'))

        # test update METAINFO in ToyDataset.
        class ToyDataset(self.dataset_type):
            METAINFO = copy.deepcopy(self.dataset_type.METAINFO)
            METAINFO['classes'] = ('bird', )

        assert ToyDataset.METAINFO == dict(
            dataset_type=dataset_type, classes=('bird', ))
        assert self.dataset_type.METAINFO == dict(
            dataset_type=dataset_type, classes=('dog', 'cat', 'fish'))

    @pytest.mark.parametrize('lazy_init', [True, False])
    def test_length(self, lazy_init):
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=lazy_init)
        if not lazy_init:
            assert dataset._fully_initialized
            assert hasattr(dataset, 'data_list')
            assert len(dataset) == 2
        else:
            # test `__len__()` when lazy_init is True
            assert not dataset._fully_initialized
            assert not dataset.data_list
            # call `full_init()` automatically
            assert len(dataset) == 2
            assert dataset._fully_initialized
            assert hasattr(dataset, 'data_list')

    def test_compose(self):
        # test callable transform
        transforms = [function_pipeline]
        compose = Compose(transforms=transforms)
        assert (self.imgs == compose(dict(img=self.imgs))['img']).all()
        # test transform build from cfg_dict
        transforms = [dict(type='CallableTransform')]
        compose = Compose(transforms=transforms)
        assert (self.imgs == compose(dict(img=self.imgs))['img']).all()
        # test return None in advance
        none_func = MagicMock(return_value=None)
        transforms = [none_func, function_pipeline]
        compose = Compose(transforms=transforms)
        assert compose(dict(img=self.imgs)) is None
        # test repr
        repr_str = f'Compose(\n' \
                   f'    {none_func}\n' \
                   f'    {function_pipeline}\n' \
                   f')'
        assert repr(compose) == repr_str
        # non-callable transform will raise error
        with pytest.raises(TypeError):
            transforms = [dict(type='NotCallableTransform')]
            Compose(transforms)

        # transform must be callable or dict
        with pytest.raises(TypeError):
            Compose([1])

    @pytest.mark.parametrize('lazy_init', [True, False])
    def test_getitem(self, lazy_init):
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=lazy_init)
        dataset.pipeline = self.pipeline
        if not lazy_init:
            assert dataset._fully_initialized
            assert hasattr(dataset, 'data_list')
            assert dataset[0] == dict(imgs=self.imgs)
        else:
            # test `__getitem__()` when lazy_init is True
            assert not dataset._fully_initialized
            assert not dataset.data_list
            # call `full_init()` automatically
            assert dataset[0] == dict(imgs=self.imgs)
            assert dataset._fully_initialized
            assert hasattr(dataset, 'data_list')

        # test with test mode
        dataset.test_mode = True
        assert dataset[0] == dict(imgs=self.imgs)

        pipeline = MagicMock(return_value=None)
        dataset.pipeline = pipeline
        # test cannot get a valid image.
        dataset.test_mode = False
        with pytest.raises(Exception):
            dataset[0]

    @pytest.mark.parametrize('lazy_init', [True, False])
    def test_get_data_info(self, lazy_init):
        self._init_dataset()
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=lazy_init)

        if not lazy_init:
            assert dataset._fully_initialized
            assert hasattr(dataset, 'data_list')
            assert dataset.get_data_info(0) == self.data_info
        else:
            # test `get_data_info()` when lazy_init is True
            assert not dataset._fully_initialized
            assert not dataset.data_list
            # call `full_init()` automatically
            assert dataset.get_data_info(0) == self.data_info
            assert dataset._fully_initialized
            assert hasattr(dataset, 'data_list')

    def test_force_full_init(self):
        with pytest.raises(AttributeError):

            class ClassWithoutFullInit:

                @force_full_init
                def foo(self):
                    pass

            class_without_full_init = ClassWithoutFullInit()
            class_without_full_init.foo()

    def test_full_init(self):
        self._init_dataset()
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=True)
        dataset.pipeline = self.pipeline
        # test `full_init()` when lazy_init is True
        assert not dataset._fully_initialized
        assert not dataset.data_list
        # call `full_init()` manually
        dataset.full_init()
        assert dataset._fully_initialized
        assert hasattr(dataset, 'data_list')
        assert len(dataset) == 2
        assert dataset[0] == dict(imgs=self.imgs)
        assert dataset.get_data_info(0) == self.data_info

        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=False)

        dataset.pipeline = self.pipeline
        assert dataset._fully_initialized
        assert hasattr(dataset, 'data_list')
        assert len(dataset) == 2
        assert dataset[0] == dict(imgs=self.imgs)
        assert dataset.get_data_info(0) == self.data_info

        # test the instantiation of self.base_dataset when passing indices
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img=None),
            ann_file='annotations/dummy_annotation.json')
        dataset_sliced = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img=None),
            ann_file='annotations/dummy_annotation.json',
            indices=1
        )
        assert dataset_sliced[0] == dataset[0]
        assert len(dataset_sliced) == 1

    @pytest.mark.parametrize('lazy_init, serialize_data',
                             ([True, False], [False, True],
                              [True, True], [False, False]))
    def test_get_subset_(self, lazy_init, serialize_data):
        indices = 1
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img=None),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=lazy_init,
            serialize_data=serialize_data
        )
        ori_data = dataset[0]
        # Test int indices.
        dataset.get_subset_(indices)
        assert len(dataset) == 1
        assert dataset[0] == ori_data
        # Test list indices.
        indices = [1]
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img=None),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=lazy_init,
            serialize_data=serialize_data
        )
        ori_data = dataset[1]
        # The sample_idx of sliced dataset's first data information should
        # be 0.
        ori_data['sample_idx'] = 0
        dataset.get_subset_(indices)
        assert len(dataset) == 1
        assert dataset[0] == ori_data
        # If type of indices is int, indices must be positive.
        with pytest.raises(TypeError):
            dataset.get_subset(-1)


    @pytest.mark.parametrize('lazy_init, serialize_data',
                             ([True, False], [False, True],
                              [True, True], [False, False]))
    def test_get_subset(self, lazy_init, serialize_data):
        indices = 1
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img=None),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=lazy_init,
            serialize_data=serialize_data
        )
        dataset_sliced = dataset.get_subset(indices)
        assert len(dataset_sliced) == 1
        assert dataset_sliced[0] == dataset[0]
        indices = [1]
        dataset_sliced = dataset.get_subset(indices)
        ori_data = dataset[1]
        ori_data['sample_idx'] = 0
        dataset.get_subset_(indices)
        assert len(dataset) == 1
        assert dataset[0] == ori_data

    def test_rand_another(self):
        # test the instantiation of self.base_dataset when passing num_samples
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img=None),
            ann_file='annotations/dummy_annotation.json',
            indices=1)
        assert dataset._rand_another() >= 0
        assert dataset._rand_another() < len(dataset)


class TestConcatDataset:

    def _init_dataset(self):
        dataset = BaseDataset

        # create dataset_a
        data_info = dict(filename='test_img.jpg', height=604, width=640)
        dataset.parse_data_info = MagicMock(return_value=data_info)
        imgs = torch.rand((2, 3, 32, 32))

        self.dataset_a = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')
        self.dataset_a.pipeline = MagicMock(return_value=dict(imgs=imgs))

        # create dataset_b
        data_info = dict(filename='gray.jpg', height=288, width=512)
        dataset.parse_data_info = MagicMock(return_value=data_info)
        imgs = torch.rand((2, 3, 32, 32))
        self.dataset_b = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')
        self.dataset_b.pipeline = MagicMock(return_value=dict(imgs=imgs))
        # test init
        self.cat_datasets = ConcatDataset(
            datasets=[self.dataset_a, self.dataset_b])

    def test_full_init(self):
        self._init_dataset()
        # test init with lazy_init=True
        self.cat_datasets.full_init()
        assert len(self.cat_datasets) == 4
        self.cat_datasets.full_init()
        self.cat_datasets._fully_initialized = False
        self.cat_datasets[1]
        assert len(self.cat_datasets) == 4

        with pytest.raises(NotImplementedError):
            self.cat_datasets.get_subset_(1)

        with pytest.raises(NotImplementedError):
            self.cat_datasets.get_subset(1)
        # Different meta information will raise error.
        with pytest.raises(ValueError):
            dataset_b = BaseDataset(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img='imgs'),
                ann_file='annotations/dummy_annotation.json',
                metainfo=dict(classes=('cat')))
            ConcatDataset(datasets=[self.dataset_a, dataset_b])

    def test_metainfo(self):
        self._init_dataset()
        assert self.cat_datasets.meta == self.dataset_a.meta

    def test_length(self):
        self._init_dataset()
        assert len(self.cat_datasets) == (
            len(self.dataset_a) + len(self.dataset_b))

    def test_getitem(self):
        self._init_dataset()
        assert (
            self.cat_datasets[0]['imgs'] == self.dataset_a[0]['imgs']).all()
        assert (self.cat_datasets[0]['imgs'] !=
                self.dataset_b[0]['imgs']).all()

        assert (
            self.cat_datasets[-1]['imgs'] == self.dataset_b[-1]['imgs']).all()
        assert (self.cat_datasets[-1]['imgs'] !=
                self.dataset_a[-1]['imgs']).all()

    def test_get_data_info(self):
        self._init_dataset()
        assert self.cat_datasets.get_data_info(
            0) == self.dataset_a.get_data_info(0)
        assert self.cat_datasets.get_data_info(
            0) != self.dataset_b.get_data_info(0)

        assert self.cat_datasets.get_data_info(
            -1) == self.dataset_b.get_data_info(-1)
        assert self.cat_datasets.get_data_info(
            -1) != self.dataset_a.get_data_info(-1)

    def test_get_ori_dataset_idx(self):
        self._init_dataset()
        assert self.cat_datasets._get_ori_dataset_idx(3) == (
            1, 3 - len(self.dataset_a))
        assert self.cat_datasets._get_ori_dataset_idx(-1) == (
            1, len(self.dataset_b) - 1)
        with pytest.raises(ValueError):
            assert self.cat_datasets._get_ori_dataset_idx(-10)


class TestRepeatDataset:

    def _init_dataset(self):
        dataset = BaseDataset
        data_info = dict(filename='test_img.jpg', height=604, width=640)
        dataset.parse_data_info = MagicMock(return_value=data_info)
        imgs = torch.rand((2, 3, 32, 32))
        self.dataset = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')
        self.dataset.pipeline = MagicMock(return_value=dict(imgs=imgs))

        self.repeat_times = 5
        # test init
        self.repeat_datasets = RepeatDataset(
            dataset=self.dataset, times=self.repeat_times)

    def test_full_init(self):
        self._init_dataset()

        self.repeat_datasets.full_init()
        assert len(self.repeat_datasets) == self.repeat_times * len(
            self.dataset)
        self.repeat_datasets.full_init()
        self.repeat_datasets._fully_initialized = False
        self.repeat_datasets[1]
        assert len(self.repeat_datasets) == \
               self.repeat_times * len(self.dataset)

        with pytest.raises(NotImplementedError):
            self.repeat_datasets.get_subset_(1)

        with pytest.raises(NotImplementedError):
            self.repeat_datasets.get_subset(1)

    def test_metainfo(self):
        self._init_dataset()
        assert self.repeat_datasets.meta == self.dataset.meta

    def test_length(self):
        self._init_dataset()
        assert len(
            self.repeat_datasets) == len(self.dataset) * self.repeat_times

    def test_getitem(self):
        self._init_dataset()
        for i in range(self.repeat_times):
            assert self.repeat_datasets[len(self.dataset) *
                                        i] == self.dataset[0]

    def test_get_data_info(self):
        self._init_dataset()
        for i in range(self.repeat_times):
            assert self.repeat_datasets.get_data_info(
                len(self.dataset) * i) == self.dataset.get_data_info(0)


class TestClassBalancedDataset:

    def _init_dataset(self):
        dataset = BaseDataset
        data_info = dict(filename='test_img.jpg', height=604, width=640)
        dataset.parse_data_info = MagicMock(return_value=data_info)
        imgs = torch.rand((2, 3, 32, 32))
        dataset.get_cat_ids = MagicMock(return_value=[0])
        self.dataset = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')
        self.dataset.pipeline = MagicMock(return_value=dict(imgs=imgs))

        self.repeat_indices = [0, 0, 1, 1, 1]
        # test init
        self.cls_banlanced_datasets = ClassBalancedDataset(
            dataset=self.dataset, oversample_thr=1e-3)
        self.cls_banlanced_datasets.repeat_indices = self.repeat_indices

    def test_full_init(self):
        self._init_dataset()

        self.cls_banlanced_datasets.full_init()
        self.cls_banlanced_datasets.repeat_indices = self.repeat_indices
        assert len(self.cls_banlanced_datasets) == len(self.repeat_indices)
        self.cls_banlanced_datasets.full_init()
        self.cls_banlanced_datasets._fully_initialized = False
        self.cls_banlanced_datasets[1]
        self.cls_banlanced_datasets.repeat_indices = self.repeat_indices
        assert len(self.cls_banlanced_datasets) == len(self.repeat_indices)

        with pytest.raises(NotImplementedError):
            self.cls_banlanced_datasets.get_subset_(1)

        with pytest.raises(NotImplementedError):
            self.cls_banlanced_datasets.get_subset(1)

    def test_metainfo(self):
        self._init_dataset()
        assert self.cls_banlanced_datasets.meta == self.dataset.meta

    def test_length(self):
        self._init_dataset()
        assert len(self.cls_banlanced_datasets) == len(self.repeat_indices)

    def test_getitem(self):
        self._init_dataset()
        for i in range(len(self.repeat_indices)):
            assert self.cls_banlanced_datasets[i] == self.dataset[
                self.repeat_indices[i]]

    def test_get_data_info(self):
        self._init_dataset()
        for i in range(len(self.repeat_indices)):
            assert self.cls_banlanced_datasets.get_data_info(
                i) == self.dataset.get_data_info(self.repeat_indices[i])

    def test_get_cat_ids(self):
        self._init_dataset()
        for i in range(len(self.repeat_indices)):
            assert self.cls_banlanced_datasets.get_cat_ids(
                i) == self.dataset.get_cat_ids(self.repeat_indices[i])
