# Copyright (c) OpenMMLab. All rights reserved.
import random
from functools import partial
from unittest import TestCase

import numpy as np
import pytest
import torch

from mmengine.data import BaseDataElement, BaseDataSample


class TestBaseDataSample(TestCase):

    def setup_data(self):
        metainfo = dict(
            img_id=random.randint(0, 100),
            img_shape=(random.randint(400, 600), random.randint(400, 600)))
        gt_instances = BaseDataElement(
            data=dict(bboxes=torch.rand((5, 4)), labels=torch.rand((5, ))))
        pred_instances = BaseDataElement(
            data=dict(bboxes=torch.rand((5, 4)), scores=torch.rand((5, ))))
        data = dict(gt_instances=gt_instances, pred_instances=pred_instances)
        return metainfo, data

    def is_equal(self, x, y):
        assert type(x) == type(y)
        if isinstance(
                x, (int, float, str, list, tuple, dict, set, BaseDataElement)):
            return x == y
        elif isinstance(x, (torch.Tensor, np.ndarray)):
            return (x == y).all()

    def check_key_value(self, instances, metainfo=None, data=None):
        # check the existence of keys in metainfo, data, and instances
        if metainfo:
            for k, v in metainfo.items():
                assert k in instances
                assert k in instances.keys()
                assert k in instances.metainfo_keys()
                assert k not in instances.data_keys()
                assert self.is_equal(instances.get(k), v)
                assert self.is_equal(getattr(instances, k), v)
        if data:
            for k, v in data.items():
                assert k in instances
                assert k in instances.keys()
                assert k not in instances.metainfo_keys()
                assert k in instances.data_keys()
                assert self.is_equal(instances.get(k), v)
                assert self.is_equal(getattr(instances, k), v)

    def check_data_device(self, instances, device):
        # assert instances.device == device
        for v in instances.data_values():
            if isinstance(v, torch.Tensor):
                assert v.device == torch.device(device)
            elif isinstance(v, (BaseDataSample, BaseDataElement)):
                self.check_data_device(v, device)

    def check_data_dtype(self, instances, dtype):
        for v in instances.data_values():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                assert isinstance(v, dtype)
            if isinstance(v, (BaseDataSample, BaseDataElement)):
                self.check_data_dtype(v, dtype)

    def check_requires_grad(self, instances):
        for v in instances.data_values():
            if isinstance(v, torch.Tensor):
                assert v.requires_grad is False
            if isinstance(v, (BaseDataSample, BaseDataElement)):
                self.check_requires_grad(v)

    def test_init(self):
        # initialization with no data and metainfo
        metainfo, data = self.setup_data()
        instances = BaseDataSample()
        for k in metainfo:
            assert k not in instances
            assert instances.get(k, None) is None
        for k in data:
            assert k not in instances
            assert instances.get(k, 'abc') == 'abc'

        # initialization with kwargs
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo=metainfo, data=data)
        self.check_key_value(instances, metainfo, data)

        # initialization with args
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo, data)
        self.check_key_value(instances, metainfo, data)

        # initialization with args
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo=metainfo)
        self.check_key_value(instances, metainfo)
        instances = BaseDataSample(data=data)
        self.check_key_value(instances, data=data)

    def test_new(self):
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo=metainfo, data=data)

        # test new() with no arguments
        new_instances = instances.new()
        assert type(new_instances) == type(instances)
        # After deepcopy, the address of new data'element will be same as
        # origin, but when change new data' element will not effect the origin
        # element and will have new address
        _, data = self.setup_data()
        new_instances.set_data(data)
        assert not self.is_equal(new_instances.gt_instances,
                                 instances.gt_instances)
        self.check_key_value(new_instances, metainfo, data)

        # test new() with arguments
        metainfo, data = self.setup_data()
        new_instances = instances.new(metainfo=metainfo, data=data)
        assert type(new_instances) == type(instances)
        assert id(new_instances.gt_instances) != id(instances.gt_instances)
        _, new_data = self.setup_data()
        new_instances.set_data(new_data)
        assert id(new_instances.gt_instances) != id(data['gt_instances'])
        self.check_key_value(new_instances, metainfo, new_data)

        metainfo, data = self.setup_data()
        new_instances = instances.new(metainfo=metainfo)

    def test_set_metainfo(self):
        metainfo, _ = self.setup_data()
        instances = BaseDataSample()
        instances.set_metainfo(metainfo)
        self.check_key_value(instances, metainfo=metainfo)

        # test setting existing keys and new keys
        new_metainfo, _ = self.setup_data()
        new_metainfo.update(other=123)
        instances.set_metainfo(new_metainfo)
        self.check_key_value(instances, metainfo=new_metainfo)

        # test have the same key in data
        # TODO
        _, data = self.setup_data()
        instances = BaseDataSample(data=data)
        _, data = self.setup_data()
        with self.assertRaises(AttributeError):
            instances.set_metainfo(data)

        with self.assertRaises(AssertionError):
            instances.set_metainfo(123)

    def test_set_data(self):
        metainfo, data = self.setup_data()
        instances = BaseDataSample()

        instances.gt_instances = data['gt_instances']
        instances.pred_instances = data['pred_instances']
        self.check_key_value(instances, data=data)

        # a.xx only set data rather than metainfo
        instances.img_shape = metainfo['img_shape']
        instances.img_id = metainfo['img_id']
        self.check_key_value(instances, data=metainfo)

        # test can not set metainfo with `.`
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo, data)
        with self.assertRaises(AttributeError):
            instances.img_shape = metainfo['img_shape']

        # test set '_metainfo_fields' or '_data_fields'
        with self.assertRaises(AttributeError):
            instances._metainfo_fields = 1
        with self.assertRaises(AttributeError):
            instances._data_fields = 1

        with self.assertRaises(AssertionError):
            instances.set_data(123)

    def test_delete_modify(self):
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo, data)

        new_metainfo, new_data = self.setup_data()
        # avoid generating same metainfo
        while True:
            if new_metainfo['img_id'] == metainfo['img_id'] or new_metainfo[
                    'img_shape'] == metainfo['img_shape']:
                new_metainfo, new_data = self.setup_data()
            else:
                break

        instances.gt_instances = new_data['gt_instances']
        instances.pred_instances = new_data['pred_instances']

        # a.xx only set data rather than metainfo
        instances.set_metainfo(new_metainfo)
        self.check_key_value(instances, new_metainfo, new_data)

        assert not self.is_equal(instances.gt_instances, data['gt_instances'])
        assert not self.is_equal(instances.pred_instances,
                                 data['pred_instances'])
        assert not self.is_equal(instances.img_id, metainfo['img_id'])
        assert not self.is_equal(instances.img_shape, metainfo['img_shape'])

        del instances.gt_instances
        del instances.img_id
        assert not self.is_equal(
            instances.pop('pred_instances', None), data['pred_instances'])
        with self.assertRaises(AttributeError):
            del instances.pred_instances

        assert 'gt_instances' not in instances
        assert 'pred_instances' not in instances
        assert 'img_id' not in instances
        assert instances.pop('gt_instances', None) is None
        # test pop not exist key without default
        with self.assertRaises(KeyError):
            instances.pop('gt_instances')
        assert instances.pop('pred_instances', 'abcdef') == 'abcdef'

        assert instances.pop('img_id', None) is None
        # test pop not exist key without default
        with self.assertRaises(KeyError):
            instances.pop('img_id')
        assert instances.pop('img_shape') == new_metainfo['img_shape']

        # test del '_metainfo_fields' or '_data_fields'
        with self.assertRaises(AttributeError):
            del instances._metainfo_fields
        with self.assertRaises(AttributeError):
            del instances._data_fields

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='GPU is required!')
    def test_cuda(self):
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo, data)

        cuda_instances = instances.cuda()
        self.check_data_device(cuda_instances, 'cuda:0')

        # here we further test to convert from cuda to cpu
        cpu_instances = cuda_instances.cpu()
        self.check_data_device(cpu_instances, 'cpu')
        del cuda_instances

        cuda_instances = instances.to('cuda:0')
        self.check_data_device(cuda_instances, 'cuda:0')

        _, data = self.setup_data()
        instances = BaseDataSample(metainfo=data)

        cuda_instances = instances.cuda()
        self.check_data_device(cuda_instances, 'cuda:0')

        # here we further test to convert from cuda to cpu
        cpu_instances = cuda_instances.cpu()
        self.check_data_device(cpu_instances, 'cpu')
        del cuda_instances

        cuda_instances = instances.to('cuda:0')
        self.check_data_device(cuda_instances, 'cuda:0')

    def test_cpu(self):
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo, data)
        self.check_data_device(instances, 'cpu')

        cpu_instances = instances.cpu()
        # assert cpu_instances.device == 'cpu'
        self.check_data_device(cpu_instances, 'cpu')

        _, data = self.setup_data()
        instances = BaseDataSample(metainfo=data)
        self.check_data_device(instances, 'cpu')

        cpu_instances = instances.cpu()
        # assert cpu_instances.device == 'cpu'
        self.check_data_device(cpu_instances, 'cpu')

    def test_numpy_tensor(self):
        metainfo, data = self.setup_data()
        data.update(bboxes=torch.rand((5, 4)))
        instances = BaseDataSample(metainfo, data)
        np_instances = instances.numpy()
        self.check_data_dtype(np_instances, np.ndarray)

        tensor_instances = np_instances.to_tensor()
        self.check_data_dtype(tensor_instances, torch.Tensor)

        _, data = self.setup_data()
        data.update(bboxes=torch.rand((5, 4)))
        instances = BaseDataSample(metainfo=data)

        np_instances = instances.numpy()
        self.check_data_dtype(np_instances, np.ndarray)

        tensor_instances = np_instances.to_tensor()
        self.check_data_dtype(tensor_instances, torch.Tensor)

    def test_detach(self):
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo, data)
        instances.detach()
        self.check_requires_grad(instances)

        _, data = self.setup_data()
        instances = BaseDataSample(metainfo=data)
        instances.detach()
        self.check_requires_grad(instances)

    def test_repr(self):
        metainfo = dict(img_shape=(800, 1196, 3))
        gt_instances = BaseDataElement(
            metainfo=metainfo,
            data=dict(det_labels=torch.LongTensor([0, 1, 2, 3])))

        data = dict(gt_instances=gt_instances)
        sample = BaseDataSample(metainfo=metainfo, data=data)
        address = hex(id(sample))
        address_gt_instances = hex(id(sample.gt_instances))
        assert repr(sample) == (f'<BaseDataSample('
                                f'\n  META INFORMATION \n'
                                f'img_shape: (800, 1196, 3) \n'
                                f'\n  DATA FIELDS \n'
                                f'gt_instances:<BaseDataElement('
                                f'\n  META INFORMATION \n'
                                f'img_shape: (800, 1196, 3) \n'
                                f'\n  DATA FIELDS \n'
                                f'shape of det_labels: torch.Size([4]) \n'
                                f'\n) at {address_gt_instances}>\n'
                                f'\n) at {address}>')

        sample = BaseDataSample(data=metainfo, metainfo=data)
        address = hex(id(sample))
        address_gt_instances = hex(id(sample.gt_instances))
        assert repr(sample) == (f'<BaseDataSample('
                                f'\n  META INFORMATION \n'
                                f'gt_instances:<BaseDataElement('
                                f'\n  META INFORMATION \n'
                                f'img_shape: (800, 1196, 3) \n'
                                f'\n  DATA FIELDS \n'
                                f'shape of det_labels: torch.Size([4]) \n'
                                f'\n) at {address_gt_instances}>\n'
                                f'\n  DATA FIELDS \n'
                                f'img_shape: (800, 1196, 3) \n'
                                f'\n) at {address}>')
        metainfo = dict(bboxes=torch.rand((5, 4)))
        sample = BaseDataSample(metainfo=metainfo)
        address = hex(id(sample))
        assert repr(sample) == (f'<BaseDataSample('
                                f'\n  META INFORMATION \n'
                                f'shape of bboxes: torch.Size([5, 4]) \n'
                                f'\n  DATA FIELDS \n'
                                f'\n) at {address}>')
        sample = BaseDataSample(data=metainfo)
        address = hex(id(sample))
        assert repr(sample) == (f'<BaseDataSample('
                                f'\n  META INFORMATION \n'
                                f'\n  DATA FIELDS \n'
                                f'shape of bboxes: torch.Size([5, 4]) \n'
                                f'\n) at {address}>')

    def test_set_get_fields(self):
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo)
        for key, value in data.items():
            instances.set_field(name=key, value=value, dtype=BaseDataElement)
        self.check_key_value(instances, data=data)

        # test type check
        _, data = self.setup_data()
        instances = BaseDataSample()
        for key, value in data.items():
            with self.assertRaises(AssertionError):
                instances.set_field(
                    name=key, value=value, dtype=BaseDataSample)

    def test_del_field(self):
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo)
        for key, value in data.items():
            instances.set_field(value=value, name=key, dtype=BaseDataElement)
        instances.del_field('gt_instances')
        instances.del_field('pred_instances')

        # gt_instance has been deleted, instances does not have the gt_instance
        with self.assertRaises(AttributeError):
            instances.del_field('gt_instances')
        assert 'gt_instances' not in instances
        assert 'pred_instances' not in instances

    def test_inheritance(self):

        class DetDataSample(BaseDataSample):
            proposals = property(
                fget=partial(BaseDataSample.get_field, name='_proposals'),
                fset=partial(
                    BaseDataSample.set_field,
                    name='_proposals',
                    dtype=BaseDataElement),
                fdel=partial(BaseDataSample.del_field, name='_proposals'),
                doc='Region proposals of an image')
            gt_instances = property(
                fget=partial(BaseDataSample.get_field, name='_gt_instances'),
                fset=partial(
                    BaseDataSample.set_field,
                    name='_gt_instances',
                    dtype=BaseDataElement),
                fdel=partial(BaseDataSample.del_field, name='_gt_instances'),
                doc='Ground truth instances of an image')
            pred_instances = property(
                fget=partial(BaseDataSample.get_field, name='_pred_instances'),
                fset=partial(
                    BaseDataSample.set_field,
                    name='_pred_instances',
                    dtype=BaseDataElement),
                fdel=partial(BaseDataSample.del_field, name='_pred_instances'),
                doc='Predicted instances of an image')

        det_sample = DetDataSample()

        # test set
        proposals = BaseDataElement(data=dict(bboxes=torch.rand((5, 4))))
        det_sample.proposals = proposals
        assert 'proposals' in det_sample

        # test get
        assert det_sample.proposals == proposals

        # test delete
        del det_sample.proposals
        assert 'proposals' not in det_sample

        # test the data whether meet the requirements
        with self.assertRaises(AssertionError):
            det_sample.proposals = torch.rand((5, 4))

    def test_values(self):
        # test_metainfo_values
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo, data)
        assert len(instances.metainfo_values()) == len(metainfo.values())
        # test_values
        assert len(
            instances.values()) == len(metainfo.values()) + len(data.values())

        # test_data_values
        assert len(instances.data_values()) == len(data.values())

    def test_keys(self):
        # test_metainfo_keys
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo, data)
        assert len(instances.metainfo_keys()) == len(metainfo.keys())

        # test_keys
        assert len(instances.keys()) == len(data.keys()) + len(metainfo.keys())

        # test_data_keys
        assert len(instances.data_keys()) == len(data.keys())

    def test_items(self):
        # test_metainfo_items
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo, data)
        assert len(dict(instances.metainfo_items())) == len(
            dict(metainfo.items()))
        # test_items
        assert len(dict(instances.items())) == len(dict(
            metainfo.items())) + len(dict(data.items()))

        # test_data_items
        assert len(dict(instances.data_items())) == len(dict(data.items()))
