# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from unittest.mock import MagicMock

import pytest
import torch

from mmengine.dataset import (BaseDataset, ClassBalancedDataset, ConcatDataset,
                              RepeatDataset)


class ToyDataset(BaseDataset):

    def parse_annotations(self, raw_data_info: dict):
        pass


class TestBaseDataset:
    dataset_type = ToyDataset
    data_info = dict(
        filename='test_img.jpg', height=604, width=640, sample_idx=0)
    imgs = torch.rand((2, 3, 32, 32))
    pipeline = MagicMock(return_value=dict(imgs=imgs))
    META: dict = dict()
    parse_annotations = MagicMock(return_value=data_info)

    def _init_dataset(self):
        self.dataset_type.META = self.META
        self.dataset_type.parse_annotations = self.parse_annotations

    def test_init(self):
        self._init_dataset()
        # test the instantiation of self.base_dataset
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')
        assert dataset._fully_initialized
        assert hasattr(dataset, 'data_infos')
        assert hasattr(dataset, 'data_address')
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img=None),
            ann_file='annotations/dummy_annotation.json')
        assert dataset._fully_initialized
        assert hasattr(dataset, 'data_infos')
        assert hasattr(dataset, 'data_address')

        # test the instantiation of self.base_dataset with
        # `serialize_data=False`
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            serialize_data=False)
        assert dataset._fully_initialized
        assert hasattr(dataset, 'data_infos')
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
        assert not dataset.data_infos

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

        # test the instantiation of self.base_dataset when `parse_annotations`
        # return `list[dict]`
        self.dataset_type.parse_annotations = MagicMock(
            return_value=[self.data_info,
                          self.data_info.copy()])
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')
        dataset.pipeline = self.pipeline
        assert dataset._fully_initialized
        assert hasattr(dataset, 'data_infos')
        assert hasattr(dataset, 'data_address')
        assert len(dataset) == 4
        assert dataset[0] == dict(imgs=self.imgs)
        assert dataset.get_data_info(0) == self.data_info

        # test the instantiation of self.base_dataset when `parse_annotations`
        # return unsupported data.
        with pytest.raises(TypeError):
            self.dataset_type.parse_annotations = MagicMock(return_value='xxx')
            dataset = self.dataset_type(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img='imgs'),
                ann_file='annotations/dummy_annotation.json')
        with pytest.raises(TypeError):
            self.dataset_type.parse_annotations = MagicMock(
                return_value=[self.data_info, 'xxx'])
            dataset = self.dataset_type(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img='imgs'),
                ann_file='annotations/dummy_annotation.json')

    def test_meta(self):
        self._init_dataset()
        # test dataset.meta with setting the meta from annotation file as the
        # meta of self.base_dataset
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')

        assert dataset.meta == dict(
            dataset_type='test_dataset', task_name='test_task', empty_list=[])

        # test dataset.meta with setting META in self.base_dataset
        dataset_type = 'new_dataset'
        self.dataset_type.META = dict(
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

        # test dataset.meta with passing meta into self.base_dataset
        meta = dict(classes=('dog', ), task_name='new_task')
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            meta=meta)
        assert self.dataset_type.META == dict(
            dataset_type=dataset_type, classes=('dog', 'cat'))
        assert dataset.meta == dict(
            dataset_type=dataset_type,
            task_name='new_task',
            classes=('dog', ),
            empty_list=[])
        # reset `base_dataset.META`, the `dataset.meta` should not change
        self.dataset_type.META['classes'] = ('dog', 'cat', 'fish')
        assert self.dataset_type.META == dict(
            dataset_type=dataset_type, classes=('dog', 'cat', 'fish'))
        assert dataset.meta == dict(
            dataset_type=dataset_type,
            task_name='new_task',
            classes=('dog', ),
            empty_list=[])

        # test dataset.meta with passing meta containing a file into
        # self.base_dataset
        meta = dict(
            classes=osp.join(
                osp.dirname(__file__), '../data/meta/classes.txt'))
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            meta=meta)
        assert dataset.meta == dict(
            dataset_type=dataset_type,
            task_name='test_task',
            classes=['dog'],
            empty_list=[])

        # test dataset.meta with passing unsupported meta into
        # self.base_dataset
        with pytest.raises(TypeError):
            meta = 'dog'
            dataset = self.dataset_type(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img='imgs'),
                ann_file='annotations/dummy_annotation.json',
                meta=meta)

        # test dataset.meta with passing meta into self.base_dataset and
        # lazy_init is True
        meta = dict(classes=('dog', ))
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            meta=meta,
            lazy_init=True)
        # 'task_name' and 'empty_list' not in dataset.meta
        assert dataset.meta == dict(
            dataset_type=dataset_type, classes=('dog', ))

        # test whether self.base_dataset.META is changed when a customize
        # dataset inherit self.base_dataset
        # test reset META in ToyDataset.
        class ToyDataset(self.dataset_type):
            META = dict(xxx='xxx')

        assert ToyDataset.META == dict(xxx='xxx')
        assert self.dataset_type.META == dict(
            dataset_type=dataset_type, classes=('dog', 'cat', 'fish'))

        # test update META in ToyDataset.
        class ToyDataset(self.dataset_type):
            META = copy.deepcopy(self.dataset_type.META)
            META['classes'] = ('bird', )

        assert ToyDataset.META == dict(
            dataset_type=dataset_type, classes=('bird', ))
        assert self.dataset_type.META == dict(
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
            assert hasattr(dataset, 'data_infos')
            assert len(dataset) == 2
        else:
            # test `__len__()` when lazy_init is True
            assert not dataset._fully_initialized
            assert not dataset.data_infos
            # call `full_init()` automatically
            assert len(dataset) == 2
            assert dataset._fully_initialized
            assert hasattr(dataset, 'data_infos')

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
            assert hasattr(dataset, 'data_infos')
            assert dataset[0] == dict(imgs=self.imgs)
        else:
            # test `__getitem__()` when lazy_init is True
            assert not dataset._fully_initialized
            assert not dataset.data_infos
            # call `full_init()` automatically
            assert dataset[0] == dict(imgs=self.imgs)
            assert dataset._fully_initialized
            assert hasattr(dataset, 'data_infos')

        # Test with test mode
        dataset.test_mode = True
        assert dataset[0] == dict(imgs=self.imgs)

    @pytest.mark.parametrize('lazy_init', [True, False])
    def test_get_data_info(self, lazy_init):
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=lazy_init)

        if not lazy_init:
            assert dataset._fully_initialized
            assert hasattr(dataset, 'data_infos')
            assert dataset.get_data_info(0) == self.data_info
        else:
            # test `get_data_info()` when lazy_init is True
            assert not dataset._fully_initialized
            assert not dataset.data_infos
            # call `full_init()` automatically
            assert dataset.get_data_info(0) == self.data_info
            assert dataset._fully_initialized
            assert hasattr(dataset, 'data_infos')

    @pytest.mark.parametrize('lazy_init', [True, False])
    def test_full_init(self, lazy_init):
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            lazy_init=lazy_init)
        dataset.pipeline = self.pipeline
        if not lazy_init:
            assert dataset._fully_initialized
            assert hasattr(dataset, 'data_infos')
            assert len(dataset) == 2
            assert dataset[0] == dict(imgs=self.imgs)
            assert dataset.get_data_info(0) == self.data_info
        else:
            # test `full_init()` when lazy_init is True
            assert not dataset._fully_initialized
            assert not dataset.data_infos
            # call `full_init()` manually
            dataset.full_init()
            assert dataset._fully_initialized
            assert hasattr(dataset, 'data_infos')
            assert len(dataset) == 2
            assert dataset[0] == dict(imgs=self.imgs)
            assert dataset.get_data_info(0) == self.data_info

    def test_slice_data(self):
        # test the instantiation of self.base_dataset when passing num_samples
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img=None),
            ann_file='annotations/dummy_annotation.json',
            num_samples=1)
        assert len(dataset) == 1

    def test_rand_another(self):
        # test the instantiation of self.base_dataset when passing num_samples
        dataset = self.dataset_type(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img=None),
            ann_file='annotations/dummy_annotation.json',
            num_samples=1)
        assert dataset._rand_another() >= 0
        assert dataset._rand_another() < len(dataset)


class TestConcatDataset:

    def _init_dataset(self):
        dataset = ToyDataset

        # create dataset_a
        data_info = dict(filename='test_img.jpg', height=604, width=640)
        dataset.parse_annotations = MagicMock(return_value=data_info)
        imgs = torch.rand((2, 3, 32, 32))

        self.dataset_a = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')
        self.dataset_a.pipeline = MagicMock(return_value=dict(imgs=imgs))

        # create dataset_b
        data_info = dict(filename='gray.jpg', height=288, width=512)
        dataset.parse_annotations = MagicMock(return_value=data_info)
        imgs = torch.rand((2, 3, 32, 32))
        self.dataset_b = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            meta=dict(classes=('dog', 'cat')))
        self.dataset_b.pipeline = MagicMock(return_value=dict(imgs=imgs))
        # test init
        self.cat_datasets = ConcatDataset(
            datasets=[self.dataset_a, self.dataset_b])

    def test_full_init(self):
        dataset = ToyDataset

        # create dataset_a
        data_info = dict(filename='test_img.jpg', height=604, width=640)
        dataset.parse_annotations = MagicMock(return_value=data_info)
        imgs = torch.rand((2, 3, 32, 32))

        dataset_a = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')
        dataset_a.pipeline = MagicMock(return_value=dict(imgs=imgs))

        # create dataset_b
        data_info = dict(filename='gray.jpg', height=288, width=512)
        dataset.parse_annotations = MagicMock(return_value=data_info)
        imgs = torch.rand((2, 3, 32, 32))
        dataset_b = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json',
            meta=dict(classes=('dog', 'cat')))
        dataset_b.pipeline = MagicMock(return_value=dict(imgs=imgs))
        # test init with lazy_init=True
        cat_datasets = ConcatDataset(
            datasets=[dataset_a, dataset_b], lazy_init=True)
        cat_datasets.full_init()
        assert len(cat_datasets) == 4
        cat_datasets.full_init()
        cat_datasets._fully_initialized = False
        cat_datasets[1]
        assert len(cat_datasets) == 4

    def test_meta(self):
        self._init_dataset()
        assert self.cat_datasets.meta == self.dataset_a.meta
        # meta of self.cat_datasets is from the first dataset when
        # concatnating datasets with different metas.
        assert self.cat_datasets.meta != self.dataset_b.meta

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
        dataset = ToyDataset
        data_info = dict(filename='test_img.jpg', height=604, width=640)
        dataset.parse_annotations = MagicMock(return_value=data_info)
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
        dataset = ToyDataset
        data_info = dict(filename='test_img.jpg', height=604, width=640)
        dataset.parse_annotations = MagicMock(return_value=data_info)
        imgs = torch.rand((2, 3, 32, 32))
        dataset = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')
        dataset.pipeline = MagicMock(return_value=dict(imgs=imgs))

        repeat_times = 5
        # test init
        repeat_datasets = RepeatDataset(
            dataset=dataset, times=repeat_times, lazy_init=True)

        repeat_datasets.full_init()
        assert len(repeat_datasets) == repeat_times * len(dataset)
        repeat_datasets.full_init()
        repeat_datasets._fully_initialized = False
        repeat_datasets[1]
        assert len(repeat_datasets) == repeat_times * len(dataset)

    def test_meta(self):
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
        dataset = ToyDataset
        data_info = dict(filename='test_img.jpg', height=604, width=640)
        dataset.parse_annotations = MagicMock(return_value=data_info)
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
        dataset = ToyDataset
        data_info = dict(filename='test_img.jpg', height=604, width=640)
        dataset.parse_annotations = MagicMock(return_value=data_info)
        imgs = torch.rand((2, 3, 32, 32))
        dataset.get_cat_ids = MagicMock(return_value=[0])
        dataset = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img='imgs'),
            ann_file='annotations/dummy_annotation.json')
        dataset.pipeline = MagicMock(return_value=dict(imgs=imgs))

        repeat_indices = [0, 0, 1, 1, 1]
        # test init
        cls_banlanced_datasets = ClassBalancedDataset(
            dataset=dataset, oversample_thr=1e-3, lazy_init=True)

        cls_banlanced_datasets.full_init()
        cls_banlanced_datasets.repeat_indices = repeat_indices
        assert len(cls_banlanced_datasets) == len(repeat_indices)
        cls_banlanced_datasets.full_init()
        cls_banlanced_datasets._fully_initialized = False
        cls_banlanced_datasets[1]
        cls_banlanced_datasets.repeat_indices = repeat_indices
        assert len(cls_banlanced_datasets) == len(repeat_indices)

    def test_meta(self):
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
