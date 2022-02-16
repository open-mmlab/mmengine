# Copyright (c) OpenMMLab. All rights reserved.
import random
from unittest import TestCase

import numpy as np
import pytest
import torch

from mmengine.data import BaseDataElement


class TestBaseDataElement(TestCase):

    def setup_data(self):
        metainfo = dict(
            img_id=random.randint(0, 100),
            img_shape=(random.randint(400, 600), random.randint(400, 600)))
        data = dict(bboxes=torch.rand((5, 4)), scores=torch.rand((5, )))
        return metainfo, data

    def check_key_value(self, instances, metainfo=None, data=None):
        # check the existence of keys in metainfo, data, and instances
        if metainfo:
            for k, v in metainfo.items():
                assert k in instances
                assert k in instances.keys()
                assert k in instances.metainfo_keys()
                assert k not in instances.data_keys()
                assert instances.get(k) == v
                assert getattr(instances, k) == v
        if data:
            for k, v in data.items():
                assert k in instances
                assert k in instances.keys()
                assert k not in instances.metainfo_keys()
                assert k in instances.data_keys()
                assert instances.get(k) == v
                assert getattr(instances, k) == v

    def check_data_device(self, instances, device):
        assert instances.device == device
        for v in instances.data_values():
            if isinstance(v, torch.Tensor):
                assert v.device == device

    def check_data_dtype(self, instances, dtype):
        for v in instances.data_values():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                assert v.dtype == dtype

    def test_init(self):
        # initialization with no data and metainfo
        metainfo, data = self.setup_data()
        instances = BaseDataElement()
        for k in metainfo:
            assert k not in instances
            assert instances.get(k, None) is None
        for k in data:
            assert k not in instances
            assert instances.get(k, 'abc') == 'abc'

        # initialization with kwargs
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo=metainfo, data=data)
        self.check_key_value(instances, metainfo, data)

        # initialization with args
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo, data)
        self.check_key_value(instances, metainfo, data)

        # initialization with args
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo=metainfo)
        self.check_key_value(instances, metainfo)
        instances = BaseDataElement(data=data)
        self.check_key_value(instances, data=data)

    def test_new(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo=metainfo, data=data)

        # test new() with no arguments
        new_instances = instances.new()
        assert type(new_instances) == type(instances)
        assert id(new_instances.bboxes) != id(instances.bboxes)
        assert id(new_instances.bboxes) != id(data['bboxes'])
        self.check_key_value(new_instances, metainfo, data)

        # test new() with arguments
        metainfo, data = self.setup_data()
        new_instances = instances.new(metainfo=metainfo, data=data)
        assert type(new_instances) == type(instances)
        assert id(new_instances.bboxes) != id(instances.bboxes)
        assert id(new_instances.bboxes) != id(data['bboxes'])
        self.check_key_value(new_instances, metainfo, data)

    def test_set_metainfo(self):
        metainfo, _ = self.setup_data()
        instances = BaseDataElement()
        instances.set_metainfo(metainfo)
        self.check_key_value(instances, metainfo=metainfo)

        # test setting existing keys and new keys
        new_metainfo, _ = self.setup_data()
        new_metainfo.update(other=123)
        instances.set_metainfo(new_metainfo)
        self.check_key_value(instances, metainfo=new_metainfo)

    def test_set_data(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement()

        instances.bboxes = data['bboxes']
        instances.scores = data['scores']
        self.check_key_value(instances, data=data)

        # a.xx only set data rather than metainfo
        instances.img_shape = metainfo['img_shape']
        instances.img_id = metainfo['img_id']
        self.check_key_value(instances, data=metainfo)

        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo, data)
        with self.assertRaises(AssertionError):
            instances.img_shape = metainfo['img_shape']

    def test_delete_modify(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo, data)

        new_metainfo, new_data = self.setup_data()
        instances.bboxes = new_data['bboxes']
        instances.scores = new_data['scores']

        # a.xx only set data rather than metainfo
        instances.set_metainfo(new_metainfo)
        self.check_key_value(instances, new_metainfo, new_data)

        assert instances.bboxes != data['bboxes']
        assert instances.scores != data['scores']
        assert instances.img_id != metainfo['img_id']
        assert instances.img_shape != metainfo['img_shape']

        del instances.bboxes
        assert instances.pop('scores', None) == new_data['scores']
        with self.assertRaises(AttributeError):
            del instances.scores

        assert 'bboxes' not in instances
        assert 'scores' not in instances
        assert instances.pop('bboxes', None) is None
        assert instances.pop('scores', 'abcdef') == 'abcdef'

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='GPU is required!')
    def test_cuda(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo, data)

        cuda_instances = instances.cuda()
        self.check_data_device(instances, 'cuda:0')

        # here we further test to convert from cuda to cpu
        cpu_instances = cuda_instances.cpu()
        self.check_data_device(cpu_instances, 'cpu')
        del cuda_instances

        cuda_instances = instances.to('cuda:0')
        self.check_data_device(cuda_instances, 'cuda:0')

    def test_cpu(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo, data)
        self.check_data_device(instances, 'cpu')

        cpu_instances = instances.cpu()
        assert cpu_instances.device == 'cpu'
        assert cpu_instances.bboxes.device == 'cpu'
        assert cpu_instances.scores.device == 'cpu'

    def test_numpy_tensor(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo, data)

        np_instances = instances.numpy()
        self.check_data_dtype(np_instances, np.ndarray)

        tensor_instances = instances.to_tensor()
        self.check_data_dtype(tensor_instances, torch.Tensor)

    def test_repr(self):
        metainfo = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
        instances = BaseDataElement(metainfo=metainfo)
        instances.det_labels = torch.LongTensor([0, 1, 2, 3])
        instances.det_scores = torch.Tensor([0.01, 0.1, 0.2, 0.3])
        assert repr(instances) == ('<BaseDataElement(\n'
                                   '  META INFORMATION\n'
                                   'img_shape: (800, 1196, 3)\n'
                                   'pad_shape: (800, 1216, 3)\n'
                                   '  DATA FIELDS\n'
                                   'shape of det_labels: torch.Size([4])\n'
                                   'shape of det_scores: torch.Size([4])\n'
                                   ') at 0x7f84acd10f90>')
