# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from unittest.mock import MagicMock

import pytest
import torch

from mmengine.config import Config, ConfigDict
from mmengine.dataset import (BaseDataset, ClassBalancedDataset, Compose,
                              ConcatDataset, RepeatDataset, force_full_init)
from mmengine.registry import DATASETS, TRANSFORMS


def function_pipeline(data_info):
    return data_info


@TRANSFORMS.register_module()
class CallableTransform:

    def __call__(self, data_info):
        return data_info


@TRANSFORMS.register_module()
class NotCallableTransform:
    pass


@DATASETS.register_module()
class CustomDataset(BaseDataset):
    pass


class TestBaseDataset:

    def setup_method(self):
        self.data_info = dict(
            filename='test_img.jpg', height=604, width=640, sample_idx=0)
        self.imgs = torch.rand((2, 3, 32, 32))
        self.ori_meta = BaseDataset.METAINFO
        self.ori_parse_data_info = BaseDataset.parse_data_info
        BaseDataset.parse_data_info = MagicMock(return_value=self.data_info)
        self.pipeline = MagicMock(return_value=dict(imgs=self.imgs))

    def teardown_method(self):
        BaseDataset.METAINFO = self.ori_meta
        BaseDataset.parse_data_info = self.ori_parse_data_info

    def test_init(self):
        # test the instantiation of self.base_dataset
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json')
        assert dataset._fully_initialized
        assert hasattr(dataset, 'data_list')
        assert hasattr(dataset, 'data_address')
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path=''),
            ann_file='annotations/dummy_annotation.json')
        assert dataset._fully_initialized
        assert hasattr(dataset, 'data_list')
        assert hasattr(dataset, 'data_address')

        # test the instantiation of self.base_dataset with
        # `serialize_data=False`
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json',
            serialize_data=False)
        assert dataset._fully_initialized
        assert hasattr(dataset, 'data_list')
        assert not hasattr(dataset, 'data_address')
        assert len(dataset) == 3
        assert dataset.get_data_info(0) == self.data_info

        # test the instantiation of self.base_dataset with lazy init
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=True)
        assert not dataset._fully_initialized
        assert not dataset.data_list

        # test the instantiation of self.base_dataset if ann_file is not
        # existed.
        with pytest.raises(FileNotFoundError):
            BaseDataset(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img_path='imgs'),
                ann_file='annotations/not_existed_annotation.json')
        # Use the default value of ann_file, i.e., ''
        with pytest.raises(TypeError):
            BaseDataset(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img_path='imgs'))

        # test the instantiation of self.base_dataset when the ann_file is
        # wrong
        with pytest.raises(ValueError):
            BaseDataset(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img_path='imgs'),
                ann_file='annotations/annotation_wrong_keys.json')
        with pytest.raises(TypeError):
            BaseDataset(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img_path='imgs'),
                ann_file='annotations/annotation_wrong_format.json')
        with pytest.raises(TypeError):
            BaseDataset(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img_path=['img']),
                ann_file='annotations/annotation_wrong_format.json')

        # test the instantiation of self.base_dataset when `parse_data_info`
        # return `list[dict]`
        BaseDataset.parse_data_info = MagicMock(
            return_value=[self.data_info,
                          self.data_info.copy()])
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json')
        dataset.pipeline = self.pipeline
        assert dataset._fully_initialized
        assert hasattr(dataset, 'data_list')
        assert hasattr(dataset, 'data_address')
        assert len(dataset) == 6
        assert dataset[0] == dict(imgs=self.imgs)
        assert dataset.get_data_info(0) == self.data_info

        # test the instantiation of self.base_dataset when `parse_data_info`
        # return unsupported data.
        with pytest.raises(TypeError):
            BaseDataset.parse_data_info = MagicMock(return_value='xxx')
            dataset = BaseDataset(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img_path='imgs'),
                ann_file='annotations/dummy_annotation.json')
        with pytest.raises(TypeError):
            BaseDataset.parse_data_info = MagicMock(
                return_value=[self.data_info, 'xxx'])
            BaseDataset(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img_path='imgs'),
                ann_file='annotations/dummy_annotation.json')
        # test the instantiation of self.base_dataset without `ann_file`
        BaseDataset.parse_data_info = self.ori_parse_data_info
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='',
            serialize_data=False,
            lazy_init=True)
        assert not dataset.ann_file

        # Test `ann_file` and `data_root` could be None.
        dataset = BaseDataset(ann_file=None, data_root=None, lazy_init=True)

    def test_meta(self):
        # test dataset.metainfo with setting the metainfo from annotation file
        # as the metainfo of self.base_dataset.
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json')

        assert dataset.metainfo == dict(
            dataset_type='test_dataset', task_name='test_task', empty_list=[])

        # test dataset.metainfo with setting METAINFO in self.base_dataset
        dataset_type = 'new_dataset'
        BaseDataset.METAINFO = dict(
            dataset_type=dataset_type, classes=('dog', 'cat'))

        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json')
        assert dataset.metainfo == dict(
            dataset_type=dataset_type,
            task_name='test_task',
            classes=('dog', 'cat'),
            empty_list=[])

        # test dataset.metainfo with passing metainfo into self.base_dataset
        metainfo = dict(classes=('dog', ), task_name='new_task')
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json',
            metainfo=metainfo)
        assert BaseDataset.METAINFO == dict(
            dataset_type=dataset_type, classes=('dog', 'cat'))
        assert dataset.metainfo == dict(
            dataset_type=dataset_type,
            task_name='new_task',
            classes=('dog', ),
            empty_list=[])

        # test dataset.metainfo with passing metainfo as Config into
        # self.base_dataset
        metainfo = Config(dict(classes=('dog', ), task_name='new_task'))
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json',
            metainfo=metainfo)
        assert BaseDataset.METAINFO == dict(
            dataset_type=dataset_type, classes=('dog', 'cat'))
        assert dataset.metainfo == dict(
            dataset_type=dataset_type,
            task_name='new_task',
            classes=('dog', ),
            empty_list=[])

        # test dataset.metainfo with passing metainfo as ConfigDict (Mapping)
        # into self.base_dataset
        metainfo = ConfigDict(dict(classes=('dog', ), task_name='new_task'))
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json',
            metainfo=metainfo)
        assert BaseDataset.METAINFO == dict(
            dataset_type=dataset_type, classes=('dog', 'cat'))
        assert dataset.metainfo == dict(
            dataset_type=dataset_type,
            task_name='new_task',
            classes=('dog', ),
            empty_list=[])

        # reset `base_dataset.METAINFO`, the `dataset.metainfo` should not
        # change
        BaseDataset.METAINFO['classes'] = ('dog', 'cat', 'fish')
        assert BaseDataset.METAINFO == dict(
            dataset_type=dataset_type, classes=('dog', 'cat', 'fish'))
        assert dataset.metainfo == dict(
            dataset_type=dataset_type,
            task_name='new_task',
            classes=('dog', ),
            empty_list=[])

        # test dataset.metainfo with passing metainfo containing a file into
        # self.base_dataset
        metainfo = dict(
            classes=osp.join(
                osp.dirname(__file__), '../data/meta/classes.txt'))
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json',
            metainfo=metainfo)
        assert dataset.metainfo == dict(
            dataset_type=dataset_type,
            task_name='test_task',
            classes=['dog'],
            empty_list=[])

        # test dataset.metainfo with passing unsupported metainfo into
        # self.base_dataset
        with pytest.raises(TypeError):
            metainfo = 'dog'
            dataset = BaseDataset(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img_path='imgs'),
                ann_file='annotations/dummy_annotation.json',
                metainfo=metainfo)

        # test dataset.metainfo with passing metainfo into self.base_dataset
        # and lazy_init is True
        metainfo = dict(classes=('dog', ))
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json',
            metainfo=metainfo,
            lazy_init=True)
        # 'task_name' and 'empty_list' not in dataset.metainfo
        assert dataset.metainfo == dict(
            dataset_type=dataset_type, classes=('dog', ))

        # test whether self.base_dataset.METAINFO is changed when a customize
        # dataset inherit self.base_dataset
        # test reset METAINFO in ToyDataset.
        class ToyDataset(BaseDataset):
            METAINFO = dict(xxx='xxx')

        assert ToyDataset.METAINFO == dict(xxx='xxx')
        assert BaseDataset.METAINFO == dict(
            dataset_type=dataset_type, classes=('dog', 'cat', 'fish'))

        # test update METAINFO in ToyDataset.
        class ToyDataset(BaseDataset):
            METAINFO = copy.deepcopy(BaseDataset.METAINFO)
            METAINFO['classes'] = ('bird', )

        assert ToyDataset.METAINFO == dict(
            dataset_type=dataset_type, classes=('bird', ))
        assert BaseDataset.METAINFO == dict(
            dataset_type=dataset_type, classes=('dog', 'cat', 'fish'))

    @pytest.mark.parametrize('lazy_init', [True, False])
    def test_length(self, lazy_init):
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=lazy_init)
        if not lazy_init:
            assert dataset._fully_initialized
            assert hasattr(dataset, 'data_list')
            assert len(dataset) == 3
        else:
            # test `__len__()` when lazy_init is True
            assert not dataset._fully_initialized
            assert not dataset.data_list
            # call `full_init()` automatically
            assert len(dataset) == 3
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

        # when the input transform is None, do nothing
        compose = Compose(None)
        assert (compose(dict(img=self.imgs))['img'] == self.imgs).all()

        compose = Compose([])
        assert (compose(dict(img=self.imgs))['img'] == self.imgs).all()

    @pytest.mark.parametrize('lazy_init', [True, False])
    def test_getitem(self, lazy_init):
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=lazy_init)
        dataset.pipeline = self.pipeline
        if not lazy_init:
            assert dataset._fully_initialized
            assert hasattr(dataset, 'data_list')
            assert dataset[0] == dict(imgs=self.imgs)
        else:
            # Test `__getitem__()` when lazy_init is True
            assert not dataset._fully_initialized
            assert not dataset.data_list
            # Call `full_init()` automatically
            assert dataset[0] == dict(imgs=self.imgs)
            assert dataset._fully_initialized
            assert hasattr(dataset, 'data_list')

        # Test with test mode
        dataset.test_mode = False
        assert dataset[0] == dict(imgs=self.imgs)
        # Test cannot get a valid image.
        dataset.prepare_data = MagicMock(return_value=None)
        with pytest.raises(Exception):
            dataset[0]
        # Test get valid image by `_rand_another`

        def fake_prepare_data(idx):
            if idx == 0:
                return None
            else:
                return 1

        dataset.prepare_data = fake_prepare_data
        dataset[0]
        dataset.test_mode = True
        with pytest.raises(Exception):
            dataset[0]

    @pytest.mark.parametrize('lazy_init', [True, False])
    def test_get_data_info(self, lazy_init):
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
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
        # Test parse_data_info with `data_prefix`
        BaseDataset.parse_data_info = self.ori_parse_data_info
        data_root = osp.join(osp.dirname(__file__), '../data/')
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json')
        data_info = dataset.get_data_info(0)
        assert data_info['img_path'] == osp.join(data_root, 'imgs',
                                                 'test_img.jpg')

    def test_force_full_init(self):
        with pytest.raises(AttributeError):

            class ClassWithoutFullInit:

                @force_full_init
                def foo(self):
                    pass

            class_without_full_init = ClassWithoutFullInit()
            class_without_full_init.foo()

    def test_full_init(self):
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
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
        assert len(dataset) == 3
        assert dataset[0] == dict(imgs=self.imgs)
        assert dataset.get_data_info(0) == self.data_info

        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=False)

        dataset.pipeline = self.pipeline
        assert dataset._fully_initialized
        assert hasattr(dataset, 'data_list')
        assert len(dataset) == 3
        assert dataset[0] == dict(imgs=self.imgs)
        assert dataset.get_data_info(0) == self.data_info

        # test the instantiation of self.base_dataset when passing indices
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path=''),
            ann_file='annotations/dummy_annotation.json')
        dataset_sliced = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path=''),
            ann_file='annotations/dummy_annotation.json',
            indices=1)
        assert dataset_sliced[0] == dataset[0]
        assert len(dataset_sliced) == 1

    @pytest.mark.parametrize(
        'lazy_init, serialize_data',
        ([True, False], [False, True], [True, True], [False, False]))
    def test_get_subset_(self, lazy_init, serialize_data):
        # Test positive int indices.
        indices = 2
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path=''),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=lazy_init,
            serialize_data=serialize_data)

        dataset_copy = copy.deepcopy(dataset)
        dataset_copy.get_subset_(indices)
        assert len(dataset_copy) == 2
        for i in range(len(dataset_copy)):
            ori_data = dataset[i]
            assert dataset_copy[i] == ori_data

        # Test negative int indices.
        indices = -2
        dataset_copy = copy.deepcopy(dataset)
        dataset_copy.get_subset_(indices)
        assert len(dataset_copy) == 2
        for i in range(len(dataset_copy)):
            ori_data = dataset[i + 1]
            ori_data['sample_idx'] = i
            assert dataset_copy[i] == ori_data

        # If indices is 0, return empty dataset.
        dataset_copy = copy.deepcopy(dataset)
        dataset_copy.get_subset_(0)
        assert len(dataset_copy) == 0

        # Test list indices with positive element.
        indices = [1]
        dataset_copy = copy.deepcopy(dataset)
        ori_data = dataset[1]
        ori_data['sample_idx'] = 0
        dataset_copy.get_subset_(indices)
        assert len(dataset_copy) == 1
        assert dataset_copy[0] == ori_data

        # Test list indices with negative element.
        indices = [-1]
        dataset_copy = copy.deepcopy(dataset)
        ori_data = dataset[2]
        ori_data['sample_idx'] = 0
        dataset_copy.get_subset_(indices)
        assert len(dataset_copy) == 1
        assert dataset_copy[0] == ori_data

        # Test empty list.
        indices = []
        dataset_copy = copy.deepcopy(dataset)
        dataset_copy.get_subset_(indices)
        assert len(dataset_copy) == 0
        # Test list with multiple positive indices.
        indices = [0, 1, 2]
        dataset_copy = copy.deepcopy(dataset)
        dataset_copy.get_subset_(indices)
        for i in range(len(dataset_copy)):
            ori_data = dataset[i]
            ori_data['sample_idx'] = i
            assert dataset_copy[i] == ori_data
        # Test list with multiple negative indices.
        indices = [-1, -2, 0]
        dataset_copy = copy.deepcopy(dataset)
        dataset_copy.get_subset_(indices)
        for i in range(len(dataset_copy)):
            ori_data = dataset[len(dataset) - i - 1]
            ori_data['sample_idx'] = i
            assert dataset_copy[i] == ori_data

        with pytest.raises(TypeError):
            dataset.get_subset_(dict())

    @pytest.mark.parametrize(
        'lazy_init, serialize_data',
        ([True, False], [False, True], [True, True], [False, False]))
    def test_get_subset(self, lazy_init, serialize_data):
        # Test positive indices.
        indices = 2
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path=''),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=lazy_init,
            serialize_data=serialize_data)
        dataset_sliced = dataset.get_subset(indices)
        assert len(dataset_sliced) == 2
        assert dataset_sliced[0] == dataset[0]
        for i in range(len(dataset_sliced)):
            assert dataset_sliced[i] == dataset[i]
        # Test negative indices.
        indices = -2
        dataset_sliced = dataset.get_subset(indices)
        assert len(dataset_sliced) == 2
        for i in range(len(dataset_sliced)):
            ori_data = dataset[i + 1]
            ori_data['sample_idx'] = i
            assert dataset_sliced[i] == ori_data
        # If indices is 0 or empty list, return empty dataset.
        assert len(dataset.get_subset(0)) == 0
        assert len(dataset.get_subset([])) == 0
        # test list indices.
        indices = [1]
        dataset_sliced = dataset.get_subset(indices)
        ori_data = dataset[1]
        ori_data['sample_idx'] = 0
        assert len(dataset_sliced) == 1
        assert dataset_sliced[0] == ori_data
        # Test list with multiple positive index.
        indices = [0, 1, 2]
        dataset_sliced = dataset.get_subset(indices)
        for i in range(len(dataset_sliced)):
            ori_data = dataset[i]
            ori_data['sample_idx'] = i
            assert dataset_sliced[i] == ori_data
        # Test list with multiple negative index.
        indices = [-1, -2, 0]
        dataset_sliced = dataset.get_subset(indices)
        for i in range(len(dataset_sliced)):
            ori_data = dataset[len(dataset) - i - 1]
            ori_data['sample_idx'] = i
            assert dataset_sliced[i] == ori_data

    def test_rand_another(self):
        # test the instantiation of self.base_dataset when passing num_samples
        dataset = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path=''),
            ann_file='annotations/dummy_annotation.json',
            indices=1)
        assert dataset._rand_another() >= 0
        assert dataset._rand_another() < len(dataset)


class TestConcatDataset:

    def setup_method(self):
        dataset = BaseDataset

        # create dataset_a
        data_info = dict(filename='test_img.jpg', height=604, width=640)
        dataset.parse_data_info = MagicMock(return_value=data_info)
        imgs = torch.rand((2, 3, 32, 32))

        self.dataset_a = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json')
        self.dataset_a.pipeline = MagicMock(return_value=dict(imgs=imgs))

        # create dataset_b
        data_info = dict(filename='gray.jpg', height=288, width=512)
        dataset.parse_data_info = MagicMock(return_value=data_info)
        imgs = torch.rand((2, 3, 32, 32))
        self.dataset_b = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json')
        self.dataset_b.pipeline = MagicMock(return_value=dict(imgs=imgs))
        # test init
        self.cat_datasets = ConcatDataset(
            datasets=[self.dataset_a, self.dataset_b])

    def test_init(self):
        # Test build dataset from cfg.
        dataset_cfg_b = dict(
            type=CustomDataset,
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json')
        cat_datasets = ConcatDataset(datasets=[self.dataset_a, dataset_cfg_b])
        cat_datasets.datasets[1].pipeline = self.dataset_b.pipeline
        assert len(cat_datasets) == len(self.cat_datasets)
        for i in range(len(cat_datasets)):
            assert (cat_datasets.get_data_info(i) ==
                    self.cat_datasets.get_data_info(i))
            assert (cat_datasets[i] == self.cat_datasets[i])

        with pytest.raises(TypeError):
            ConcatDataset(datasets=[0])

        with pytest.raises(TypeError):
            ConcatDataset(
                datasets=[self.dataset_a, dataset_cfg_b], ignore_keys=1)

    def test_full_init(self):
        # test init with lazy_init=True
        self.cat_datasets.full_init()
        assert len(self.cat_datasets) == 6
        self.cat_datasets.full_init()
        self.cat_datasets._fully_initialized = False
        self.cat_datasets[1]
        assert len(self.cat_datasets) == 6

        with pytest.raises(NotImplementedError):
            self.cat_datasets.get_subset_(1)

        with pytest.raises(NotImplementedError):
            self.cat_datasets.get_subset(1)

        dataset_b = BaseDataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json',
            metainfo=dict(classes=('cat')))
        # Regardless of order, different meta information without
        # `ignore_keys` will raise error.
        with pytest.raises(ValueError):
            ConcatDataset(datasets=[self.dataset_a, dataset_b])
        with pytest.raises(ValueError):
            ConcatDataset(datasets=[dataset_b, self.dataset_a])
        # `ignore_keys` does not contain different meta information keys will
        # raise error.
        with pytest.raises(ValueError):
            ConcatDataset(
                datasets=[self.dataset_a, dataset_b], ignore_keys=['a'])
        # Different meta information with `ignore_keys` will not raise error.
        cat_datasets = ConcatDataset(
            datasets=[self.dataset_a, dataset_b], ignore_keys='classes')
        cat_datasets.full_init()
        assert len(cat_datasets) == 6
        cat_datasets.full_init()
        cat_datasets._fully_initialized = False
        cat_datasets[1]
        assert len(cat_datasets.metainfo) == 3
        assert len(cat_datasets) == 6

    def test_metainfo(self):
        assert self.cat_datasets.metainfo == self.dataset_a.metainfo

    def test_length(self):
        assert len(self.cat_datasets) == (
            len(self.dataset_a) + len(self.dataset_b))

    def test_getitem(self):
        assert (
            self.cat_datasets[0]['imgs'] == self.dataset_a[0]['imgs']).all()
        assert (self.cat_datasets[0]['imgs'] !=
                self.dataset_b[0]['imgs']).all()

        assert (
            self.cat_datasets[-1]['imgs'] == self.dataset_b[-1]['imgs']).all()
        assert (self.cat_datasets[-1]['imgs'] !=
                self.dataset_a[-1]['imgs']).all()

    def test_get_data_info(self):
        assert self.cat_datasets.get_data_info(
            0) == self.dataset_a.get_data_info(0)
        assert self.cat_datasets.get_data_info(
            0) != self.dataset_b.get_data_info(0)

        assert self.cat_datasets.get_data_info(
            -1) == self.dataset_b.get_data_info(-1)
        assert self.cat_datasets.get_data_info(
            -1) != self.dataset_a.get_data_info(-1)

    def test_get_ori_dataset_idx(self):
        assert self.cat_datasets._get_ori_dataset_idx(3) == (
            1, 3 - len(self.dataset_a))
        assert self.cat_datasets._get_ori_dataset_idx(-1) == (
            1, len(self.dataset_b) - 1)
        with pytest.raises(ValueError):
            assert self.cat_datasets._get_ori_dataset_idx(-10)


class TestRepeatDataset:

    def setup_method(self):
        dataset = BaseDataset
        data_info = dict(filename='test_img.jpg', height=604, width=640)
        dataset.parse_data_info = MagicMock(return_value=data_info)
        imgs = torch.rand((2, 3, 32, 32))
        self.dataset = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json')
        self.dataset.pipeline = MagicMock(return_value=dict(imgs=imgs))

        self.repeat_times = 5
        # test init
        self.repeat_datasets = RepeatDataset(
            dataset=self.dataset, times=self.repeat_times)

    def test_init(self):
        # Test build dataset from cfg.
        dataset_cfg = dict(
            type=CustomDataset,
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json')
        repeat_dataset = RepeatDataset(
            dataset=dataset_cfg, times=self.repeat_times)
        repeat_dataset.dataset.pipeline = self.dataset.pipeline
        assert len(repeat_dataset) == len(self.repeat_datasets)
        for i in range(len(repeat_dataset)):
            assert (repeat_dataset.get_data_info(i) ==
                    self.repeat_datasets.get_data_info(i))
            assert (repeat_dataset[i] == self.repeat_datasets[i])

        with pytest.raises(TypeError):
            RepeatDataset(dataset=[0], times=5)

    def test_full_init(self):
        self.repeat_datasets.full_init()
        assert len(
            self.repeat_datasets) == self.repeat_times * len(self.dataset)
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
        assert self.repeat_datasets.metainfo == self.dataset.metainfo

    def test_length(self):
        assert len(
            self.repeat_datasets) == len(self.dataset) * self.repeat_times

    def test_getitem(self):
        for i in range(self.repeat_times):
            assert self.repeat_datasets[len(self.dataset) *
                                        i] == self.dataset[0]

    def test_get_data_info(self):
        for i in range(self.repeat_times):
            assert self.repeat_datasets.get_data_info(
                len(self.dataset) * i) == self.dataset.get_data_info(0)


class TestClassBalancedDataset:

    def setup_method(self):
        dataset = BaseDataset
        data_info = dict(filename='test_img.jpg', height=604, width=640)
        dataset.parse_data_info = MagicMock(return_value=data_info)
        imgs = torch.rand((2, 3, 32, 32))
        dataset.get_cat_ids = MagicMock(return_value=[0])
        self.dataset = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json')
        self.dataset.pipeline = MagicMock(return_value=dict(imgs=imgs))

        self.repeat_indices = [0, 0, 1, 1, 1]
        # test init
        self.cls_banlanced_datasets = ClassBalancedDataset(
            dataset=self.dataset, oversample_thr=1e-3)
        self.cls_banlanced_datasets.repeat_indices = self.repeat_indices

    def test_init(self):
        # Test build dataset from cfg.
        dataset_cfg = dict(
            type=CustomDataset,
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img_path='imgs'),
            ann_file='annotations/dummy_annotation.json')
        cls_banlanced_datasets = ClassBalancedDataset(
            dataset=dataset_cfg, oversample_thr=1e-3)
        cls_banlanced_datasets.repeat_indices = self.repeat_indices
        cls_banlanced_datasets.dataset.pipeline = self.dataset.pipeline
        assert len(cls_banlanced_datasets) == len(self.cls_banlanced_datasets)
        for i in range(len(cls_banlanced_datasets)):
            assert (cls_banlanced_datasets.get_data_info(i) ==
                    self.cls_banlanced_datasets.get_data_info(i))
            assert (
                cls_banlanced_datasets[i] == self.cls_banlanced_datasets[i])

        with pytest.raises(TypeError):
            ClassBalancedDataset(dataset=[0], times=5)

    def test_full_init(self):
        self.cls_banlanced_datasets.full_init()
        self.cls_banlanced_datasets.repeat_indices = self.repeat_indices
        assert len(self.cls_banlanced_datasets) == len(self.repeat_indices)
        # Reinit `repeat_indices`.
        self.cls_banlanced_datasets._fully_initialized = False
        self.cls_banlanced_datasets.repeat_indices = self.repeat_indices
        assert len(self.cls_banlanced_datasets) != len(self.repeat_indices)

        with pytest.raises(NotImplementedError):
            self.cls_banlanced_datasets.get_subset_(1)

        with pytest.raises(NotImplementedError):
            self.cls_banlanced_datasets.get_subset(1)

    def test_metainfo(self):
        assert self.cls_banlanced_datasets.metainfo == self.dataset.metainfo

    def test_length(self):
        assert len(self.cls_banlanced_datasets) == len(self.repeat_indices)

    def test_getitem(self):
        for i in range(len(self.repeat_indices)):
            assert self.cls_banlanced_datasets[i] == self.dataset[
                self.repeat_indices[i]]

    def test_get_data_info(self):
        for i in range(len(self.repeat_indices)):
            assert self.cls_banlanced_datasets.get_data_info(
                i) == self.dataset.get_data_info(self.repeat_indices[i])

    def test_get_cat_ids(self):
        for i in range(len(self.repeat_indices)):
            assert self.cls_banlanced_datasets.get_cat_ids(
                i) == self.dataset.get_cat_ids(self.repeat_indices[i])
