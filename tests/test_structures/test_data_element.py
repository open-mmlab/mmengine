# Copyright (c) OpenMMLab. All rights reserved.
import random
from unittest import TestCase

import numpy as np
import pytest
import torch

from mmengine.structures import BaseDataElement


class DetDataSample(BaseDataElement):

    @property
    def proposals(self):
        return self._proposals

    @proposals.setter
    def proposals(self, value):
        self.set_field(value=value, name='_proposals', dtype=BaseDataElement)

    @proposals.deleter
    def proposals(self):
        del self._proposals

    @property
    def gt_instances(self):
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value):
        self.set_field(
            value=value, name='_gt_instances', dtype=BaseDataElement)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def pred_instances(self):
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value):
        self.set_field(
            value=value, name='_pred_instances', dtype=BaseDataElement)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances


class TestBaseDataElement(TestCase):

    def setup_data(self):
        metainfo = dict(
            img_id=random.randint(0, 100),
            img_shape=(random.randint(400, 600), random.randint(400, 600)))
        gt_instances = BaseDataElement(
            bboxes=torch.rand((5, 4)), labels=torch.rand((5, )))
        pred_instances = BaseDataElement(
            bboxes=torch.rand((5, 4)), scores=torch.rand((5, )))
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
                assert k in instances.all_keys()
                assert k in instances.metainfo_keys()
                assert k not in instances.keys()
                assert self.is_equal(instances.get(k), v)
                assert self.is_equal(getattr(instances, k), v)
        if data:
            for k, v in data.items():
                assert k in instances
                assert k in instances.keys()
                assert k not in instances.metainfo_keys()
                assert k in instances.all_keys()
                assert self.is_equal(instances.get(k), v)
                assert self.is_equal(getattr(instances, k), v)

    def check_data_device(self, instances, device):
        # assert instances.device == device
        for v in instances.values():
            if isinstance(v, torch.Tensor):
                assert v.device == torch.device(device)
            elif isinstance(v, BaseDataElement):
                self.check_data_device(v, device)

    def check_data_dtype(self, instances, dtype):
        for v in instances.values():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                assert isinstance(v, dtype)
            if isinstance(v, BaseDataElement):
                self.check_data_dtype(v, dtype)

    def check_requires_grad(self, instances):
        for v in instances.values():
            if isinstance(v, torch.Tensor):
                assert v.requires_grad is False
            if isinstance(v, BaseDataElement):
                self.check_requires_grad(v)

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
        instances = BaseDataElement(metainfo=metainfo, **data)
        self.check_key_value(instances, metainfo, data)

        # initialization with args
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo=metainfo)
        self.check_key_value(instances, metainfo)
        instances = BaseDataElement(**data)
        self.check_key_value(instances, data=data)

    def test_new(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo=metainfo, **data)

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
        new_instances = instances.new(metainfo=metainfo, **data)
        assert type(new_instances) == type(instances)
        assert id(new_instances.gt_instances) != id(instances.gt_instances)
        _, new_data = self.setup_data()
        new_instances.set_data(new_data)
        assert id(new_instances.gt_instances) != id(data['gt_instances'])
        self.check_key_value(new_instances, metainfo, new_data)

        metainfo, data = self.setup_data()
        new_instances = instances.new(metainfo=metainfo)

    def test_clone(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo=metainfo, **data)
        new_instances = instances.clone()
        assert type(new_instances) == type(instances)

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

        # test have the same key in data
        _, data = self.setup_data()
        instances = BaseDataElement(**data)
        _, data = self.setup_data()
        with self.assertRaises(AttributeError):
            instances.set_metainfo(data)

        with self.assertRaises(AssertionError):
            instances.set_metainfo(123)

    def test_set_data(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement()

        instances.gt_instances = data['gt_instances']
        instances.pred_instances = data['pred_instances']
        self.check_key_value(instances, data=data)

        metainfo, data = self.setup_data()
        instances = BaseDataElement()
        instances.set_data(data)
        self.check_key_value(instances, data=data)

        # a.xx only set data rather than metainfo
        instances.img_shape = metainfo['img_shape']
        instances.img_id = metainfo['img_id']
        self.check_key_value(instances, data=metainfo)

        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo=metainfo, **data)
        with self.assertRaises(AttributeError):
            instances.img_shape = metainfo['img_shape']

        # test set '_metainfo_fields' or '_data_fields'
        with self.assertRaises(AttributeError):
            instances._metainfo_fields = 1
        with self.assertRaises(AttributeError):
            instances._data_fields = 1

        with self.assertRaises(AssertionError):
            instances.set_data(123)

        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo=metainfo, **data)
        with pytest.raises(AttributeError):
            instances.set_data(dict(img_id=1))

    def test_update(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo=metainfo, **data)
        proposals = BaseDataElement(
            bboxes=torch.rand((5, 4)), scores=torch.rand((5, )))
        new_instances = BaseDataElement(proposals=proposals)
        instances.update(new_instances)
        self.check_key_value(instances, metainfo,
                             data.update(dict(proposals=proposals)))

    def test_delete_modify(self):
        random.seed(10)
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo=metainfo, **data)

        new_metainfo, new_data = self.setup_data()
        # avoid generating same metainfo, data
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
        instances = BaseDataElement(metainfo=metainfo, **data)

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
        instances = BaseDataElement(metainfo=metainfo, **data)
        self.check_data_device(instances, 'cpu')

        cpu_instances = instances.cpu()
        # assert cpu_instances.device == 'cpu'
        assert cpu_instances.gt_instances.bboxes.device == torch.device('cpu')
        assert cpu_instances.gt_instances.labels.device == torch.device('cpu')

    def test_numpy_tensor(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo=metainfo, **data)

        np_instances = instances.numpy()
        self.check_data_dtype(np_instances, np.ndarray)

        tensor_instances = np_instances.to_tensor()
        self.check_data_dtype(tensor_instances, torch.Tensor)

    def test_detach(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo=metainfo, **data)
        instances.detach()
        self.check_requires_grad(instances)

    def test_repr(self):
        metainfo = dict(img_shape=(800, 1196, 3))
        gt_instances = BaseDataElement(
            metainfo=metainfo, det_labels=torch.LongTensor([0, 1, 2, 3]))
        sample = BaseDataElement(metainfo=metainfo, gt_instances=gt_instances)
        address = hex(id(sample))
        address_gt_instances = hex(id(sample.gt_instances))
        assert repr(sample) == (
            '<BaseDataElement(\n\n'
            '    META INFORMATION\n'
            '    img_shape: (800, 1196, 3)\n\n'
            '    DATA FIELDS\n'
            '    gt_instances: <BaseDataElement(\n        \n'
            '            META INFORMATION\n'
            '            img_shape: (800, 1196, 3)\n        \n'
            '            DATA FIELDS\n'
            '            det_labels: tensor([0, 1, 2, 3])\n'
            f'        ) at {address_gt_instances}>\n'
            f') at {address}>')

    def test_set_fields(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo=metainfo)
        for key, value in data.items():
            instances.set_field(name=key, value=value, dtype=BaseDataElement)
        self.check_key_value(instances, data=data)

        # test type check
        _, data = self.setup_data()
        instances = BaseDataElement()
        for key, value in data.items():
            with self.assertRaises(AssertionError):
                instances.set_field(name=key, value=value, dtype=torch.Tensor)

    def test_inheritance(self):

        det_sample = DetDataSample()

        # test set
        proposals = BaseDataElement(bboxes=torch.rand((5, 4)))
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
        instances = BaseDataElement(metainfo=metainfo, **data)
        assert len(instances.metainfo_values()) == len(metainfo.values())
        # test_all_values
        assert len(instances.all_values()) == len(metainfo.values()) + len(
            data.values())

        # test_values
        assert len(instances.values()) == len(data.values())

    def test_keys(self):
        # test_metainfo_keys
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo=metainfo, **data)
        assert len(instances.metainfo_keys()) == len(metainfo.keys())

        # test_all_keys
        assert len(
            instances.all_keys()) == len(data.keys()) + len(metainfo.keys())

        # test_keys
        assert len(instances.keys()) == len(data.keys())

        det_sample = DetDataSample()
        proposals = BaseDataElement(bboxes=torch.rand((5, 4)))
        det_sample.proposals = proposals
        assert '_proposals' not in det_sample.keys()

    def test_items(self):
        # test_metainfo_items
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo=metainfo, **data)
        assert len(dict(instances.metainfo_items())) == len(
            dict(metainfo.items()))
        # test_all_items
        assert len(dict(instances.all_items())) == len(dict(
            metainfo.items())) + len(dict(data.items()))

        # test_items
        assert len(dict(instances.items())) == len(dict(data.items()))

    def test_to_dict(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo=metainfo, **data)
        dict_instances = instances.to_dict()
        # test convert BaseDataElement to dict
        for k in instances.all_keys():
            # all keys in instances should be in dict_instances
            assert k in dict_instances
        assert isinstance(dict_instances, dict)
        # sub data element should also be converted to dict
        assert isinstance(dict_instances['gt_instances'], dict)
        assert isinstance(dict_instances['pred_instances'], dict)

        det_sample = DetDataSample()
        proposals = BaseDataElement(bboxes=torch.rand((5, 4)))
        det_sample.proposals = proposals
        dict_sample = det_sample.to_dict()
        assert '_proposals' not in dict_sample
        assert 'proposals' in dict_sample

    def test_metainfo(self):
        # test metainfo property
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo=metainfo, **data)
        self.assertDictEqual(instances.metainfo, metainfo)
