import os.path as osp
from unittest.mock import MagicMock

import pytest
import torch

from mmengine.data import (BaseDataset, ClassBalancedDataset, ConcatDataset,
                           RepeatDataset)


class TestBaseDataset:

    def __init__(self):
        self.base_dataset = BaseDataset

        self.data_info = dict(filename='test_img.jpg', height=604, width=640)
        self.base_dataset.parse_annotations = MagicMock(
            return_value=self.data_info)

        self.imgs = torch.rand((2, 3, 32, 32))
        self.base_dataset.pipeline = MagicMock(
            return_value=dict(imgs=self.imgs))

    def test_init(self):
        # test the instantiation of self.base_dataset
        dataset = self.base_dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')
        assert dataset._fully_initialized is True
        assert hasattr(dataset, 'data_infos')
        assert hasattr(dataset, 'data_address')

        # test the instantiation of self.base_dataset with
        # `serialize_data=False`
        dataset = self.base_dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            serialize_data=False)
        assert dataset._fully_initialized is True
        assert hasattr(dataset, 'data_infos')
        assert not hasattr(dataset, 'data_address')

        # test the instantiation of self.base_dataset with lazy init
        dataset = self.base_dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=True)
        assert dataset._fully_initialized is False
        assert not hasattr(dataset, 'data_infos')

        # test the instantiation of self.base_dataset when the ann_file is
        # wrong
        with pytest.raises(ValueError):
            self.base_dataset(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img='imgs'),
                ann_file='annotations/wrong_annotation.json')

        # test the instantiation of self.base_dataset when `parse_annotations`
        # return `list[dict]`
        self.base_dataset.parse_annotations = MagicMock(
            return_value=[self.data_info,
                          self.data_info.copy()])
        dataset = self.base_dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')
        assert dataset._fully_initialized is True
        assert hasattr(dataset, 'data_infos')
        assert hasattr(dataset, 'data_address')
        assert len(dataset) == 4
        assert dataset[0] == dict(imgs=self.imgs)
        assert dataset.get_data_info(0) == self.data_info

        # set self.base_dataset to initial state
        self.__init__()

    def test_meta(self):
        # test dataset.meta with setting the meta from annotation file as the
        # meta of self.base_dataset
        dataset = self.base_dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')
        for key in ['dataset_type', 'task_name']:
            assert key in dataset.meta

        # test dataset.meta with setting META in self.base_dataset
        dataset_type = 'new_dataset'
        self.base_dataset.META = dict(
            dataset_type=dataset_type, classes=('dog', 'cat'))

        dataset = self.base_dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')
        for key in self.base_dataset.META.keys():
            assert key in dataset.meta
        assert dataset.meta['dataset_type'] == dataset_type

        # test dataset.meta with passing meta into self.base_dataset
        meta = dict(classes=('dog', ))
        dataset = self.base_dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            meta=meta)
        for key in meta:
            assert key in dataset.meta
        assert dataset.meta['classes'] == meta['classes']
        # reset `base_dataset.META`, the `dataset.meta` should not change
        self.base_dataset.META['classes'] = ('dog', 'cat')
        assert dataset.meta['classes'] == meta['classes']

        # test dataset.meta with passing meta into self.base_dataset and
        # lazy_init is True
        meta = dict(classes=('dog', ))
        dataset = self.base_dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            meta=meta,
            lazy_init=True)
        for key in meta:
            assert key in dataset.meta
        assert dataset.meta['classes'] == meta['classes']
        assert 'task_name' not in dataset.meta

        # set self.base_dataset to initial state
        self.__init__()

    @pytest.mark.parametrize('lazy_init', [True, False])
    def test_length(self, lazy_init):
        dataset = self.base_dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=lazy_init)

        if lazy_init is False:
            assert dataset._fully_initialized is True
            assert hasattr(dataset, 'data_infos')
            assert len(dataset) == 2
        else:
            # test `__len__()` when lazy_init is True
            assert dataset._fully_initialized is False
            assert not hasattr(dataset, 'data_infos')
            # call `full_init()` automatically
            assert len(dataset) == 2
            assert dataset._fully_initialized is True
            assert hasattr(dataset, 'data_infos')

    @pytest.mark.parametrize('lazy_init', [True, False])
    def test_getitem(self, lazy_init):
        dataset = self.base_dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=lazy_init)

        if lazy_init is False:
            assert dataset._fully_initialized is True
            assert hasattr(dataset, 'data_infos')
            assert dataset[0] == dict(imgs=self.imgs)
        else:
            # test `__getitem__()` when lazy_init is True
            assert dataset._fully_initialized is False
            assert not hasattr(dataset, 'data_infos')
            # call `full_init()` automatically
            assert dataset[0] == dict(imgs=self.imgs)
            assert dataset._fully_initialized is True
            assert hasattr(dataset, 'data_infos')

    @pytest.mark.parametrize('lazy_init', [True, False])
    def test_get_data_info(self, lazy_init):
        dataset = self.base_dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=lazy_init)

        if lazy_init is False:
            assert dataset._fully_initialized is True
            assert hasattr(dataset, 'data_infos')
            assert dataset.get_data_info(0) == self.data_info
        else:
            # test `get_data_info()` when lazy_init is True
            assert dataset._fully_initialized is False
            assert not hasattr(dataset, 'data_infos')
            # call `full_init()` automatically
            assert dataset.get_data_info(0) == self.data_info
            assert dataset._fully_initialized is True
            assert hasattr(dataset, 'data_infos')

    @pytest.mark.parametrize('lazy_init', [True, False])
    def test_full_init(self, lazy_init):
        dataset = self.base_dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=lazy_init)

        if lazy_init is False:
            assert dataset._fully_initialized is True
            assert hasattr(dataset, 'data_infos')
            assert len(dataset) == 2
            assert dataset[0] == dict(imgs=self.imgs)
            assert dataset.get_data_info(0) == self.data_info
        else:
            # test `full_init()` when lazy_init is True
            assert dataset._fully_initialized is False
            assert not hasattr(dataset, 'data_infos')
            # call `full_init()` manually
            dataset.full_init()
            assert dataset._fully_initialized is True
            assert hasattr(dataset, 'data_infos')
            assert len(dataset) == 2
            assert dataset[0] == dict(imgs=self.imgs)
            assert dataset.get_data_info(0) == self.data_info


class TestConcatDataset:

    def __init__(self):
        dataset = BaseDataset

        # create dataset_a
        data_info = dict(filename='test_img.jpg', height=604, width=640)
        dataset.parse_annotations = MagicMock(return_value=data_info)
        imgs = torch.rand((2, 3, 32, 32))
        dataset.pipeline = MagicMock(return_value=dict(imgs=imgs))
        self.dataset_a = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')

        # create dataset_b
        data_info = dict(filename='gray.jpg', height=288, width=512)
        dataset.parse_annotations = MagicMock(return_value=data_info)
        imgs = torch.rand((2, 3, 32, 32))
        dataset.pipeline = MagicMock(return_value=dict(imgs=imgs))
        self.dataset_b = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            meta=dict(classes=('dog', 'cat')))

        # test init
        self.cat_datasets = ConcatDataset(
            datasets=[self.dataset_a, self.dataset_b])

    def test_meta(self):
        assert self.cat_datasets.meta == self.dataset_a.meta
        assert self.cat_datasets.meta != self.dataset_b.meta

    def test_length(self):
        assert len(self.cat_datasets) == (
            len(self.dataset_a) + len(self.dataset_b))

    def test_getitem(self):
        assert self.cat_datasets[0] == self.dataset_a[0]
        assert self.cat_datasets[0] != self.dataset_b[0]

        assert self.cat_datasets[-1] == self.dataset_b[-1]
        assert self.cat_datasets[-1] != self.dataset_a[-1]

    def test_get_data_info(self):
        assert self.cat_datasets.get_data_info(
            0) == self.dataset_a.get_data_info(0)
        assert self.cat_datasets.get_data_info(
            0) != self.dataset_b.get_data_info(0)

        assert self.cat_datasets.get_data_info(
            -1) == self.dataset_b.get_data_info(-1)
        assert self.cat_datasets.get_data_info(
            -1) != self.dataset_a[-1].get_data_info(-1)


class TestRepeatDataset:

    def __init__(self):
        dataset = BaseDataset
        data_info = dict(filename='test_img.jpg', height=604, width=640)
        dataset.parse_annotations = MagicMock(return_value=data_info)
        imgs = torch.rand((2, 3, 32, 32))
        dataset.pipeline = MagicMock(return_value=dict(imgs=imgs))
        self.dataset = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')

        self.repeat_times = 5
        # test init
        self.repeat_datasets = RepeatDataset(
            dataset=self.dataset, times=self.repeat_times)

    def test_meta(self):
        assert self.repeat_datasets.meta == self.dataset.meta

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

    def __init__(self):
        dataset = BaseDataset
        data_info = dict(filename='test_img.jpg', height=604, width=640)
        dataset.parse_annotations = MagicMock(return_value=data_info)
        imgs = torch.rand((2, 3, 32, 32))
        dataset.pipeline = MagicMock(return_value=dict(imgs=imgs))
        dataset.get_cat_ids = MagicMock(return_value=[0])
        self.dataset = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')

        self.repeat_indices = [0, 0, 1, 1, 1]
        # test init
        self.cls_banlanced_datasets = ClassBalancedDataset(
            dataset=self.dataset, oversample_thr=1e-3)
        self.cls_banlanced_datasets.repeat_indices = self.repeat_indices

    def test_meta(self):
        assert self.cls_banlanced_datasets.meta == self.dataset.meta

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
