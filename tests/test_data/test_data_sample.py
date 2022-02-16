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
            elif isinstance(v, BaseDataElement):
                self.check_data_device(v, device)

    def check_data_dtype(self, instances, dtype):
        for v in instances.data_values():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                assert v.dtype == dtype
            if isinstance(v, BaseDataElement):
                self.check_data_dtype(v, dtype)

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
        assert id(new_instances.data) != id(instances.data)
        assert id(new_instances.bboxes) != id(data)
        self.check_key_value(new_instances, metainfo, data)

        # test new() with arguments
        metainfo, data = self.setup_data()
        new_instances = instances.new(metainfo=metainfo, data=data)
        assert type(new_instances) == type(instances)
        assert id(new_instances.data) != id(instances.data)
        assert id(new_instances.data) != id(data)
        self.check_key_value(new_instances, metainfo, data)

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

    def test_delete_modify(self):
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo, data)

        new_metainfo, new_data = self.setup_data()
        instances.gt_instances = new_data['gt_instances']
        instances.pred_instances = new_data['pred_instances']

        # a.xx only set data rather than metainfo
        instances.set_metainfo(new_metainfo)
        self.check_key_value(instances, new_metainfo, new_data)

        assert instances.gt_instances != data['gt_instances']
        assert instances.pred_instances != data['pred_instances']
        assert instances.img_id != metainfo['img_id']
        assert instances.img_shape != metainfo['img_shape']

        del instances.gt_instances
        assert instances.pop('pred_instances',
                             None) == new_data['pred_instances']
        with self.assertRaises(AttributeError):
            del instances.pred_instances

        assert 'gt_instances' not in instances
        assert 'pred_instances' not in instances
        assert instances.pop('gt_instances', None) is None
        assert instances.pop('pred_instances', 'abcdef') == 'abcdef'

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='GPU is required!')
    def test_cuda(self):
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo, data)

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
        instances = BaseDataSample(metainfo, data)
        self.check_data_device(instances, 'cpu')

        cpu_instances = instances.cpu()
        assert cpu_instances.device == 'cpu'
        assert cpu_instances.bboxes.device == 'cpu'
        assert cpu_instances.scores.device == 'cpu'

    def test_numpy_tensor(self):
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo, data)

        np_instances = instances.numpy()
        self.check_data_dtype(np_instances, np.ndarray)

        tensor_instances = instances.to_tensor()
        self.check_data_dtype(tensor_instances, torch.Tensor)

    def test_repr(self):
        metainfo = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
        gt_instances = BaseDataElement(
            data=dict(
                det_labels=torch.LongTensor([0, 1, 2, 3],
                                            det_scores=torch.Tensor(
                                                [0.01, 0.1, 0.2, 0.3]))))
        data = dict(gt_instances=gt_instances)
        instances = BaseDataSample(metainfo=metainfo, data=data)
        assert repr(instances) == ('<BaseDataSample(\n'
                                   '  META INFORMATION\n'
                                   'img_shape: (800, 1196, 3)\n'
                                   'pad_shape: (800, 1216, 3)\n'
                                   '  DATA FIELDS\n'
                                   '\tgt_instances: <BaseDataElement(\n'
                                   '\t  META INFORMATION\n'
                                   '\timg_shape: (800, 1196, 3)\n'
                                   '\tpad_shape: (800, 1216, 3)\n'
                                   '\t  DATA FIELDS\n'
                                   '\tshape of det_labels: torch.Size([4])\n'
                                   '\tshape of det_scores: torch.Size([4])\n'
                                   '\t) at 0x7f84acd10f90>'
                                   ') at 0x7f84acd10f90>')

    def test_set_get_fields(self):
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo)
        for key, value in data.items():
            instances._set_field(value, key, BaseDataElement)
        self.check_key_value(instances, data=data)

        # test type check
        _, data = self.setup_data()
        instances = BaseDataSample()
        for key, value in data.items():
            with self.assertRaises(AssertionError):
                instances._set_field(value, key, BaseDataSample)

    def test_del_field(self):
        metainfo, data = self.setup_data()
        instances = BaseDataSample(metainfo)
        for key, value in data.items():
            instances._set_field(value, key, BaseDataElement)
        instances._del_field('gt_instances')
        instances._del_field('pred_instances')
        with self.assertRaises(AttributeError):
            instances._del_field('gt_instances')
        assert 'gt_instances' not in instances
        assert 'pred_instances' not in instances

    def test_inherence(self):

        class DetDataSample(BaseDataSample):
            proposals = property(
                fget=partial(BaseDataSample._get_field, name='_proposals'),
                fset=partial(
                    BaseDataSample._set_field,
                    name='_proposals',
                    dtype=BaseDataElement),
                fdel=partial(BaseDataSample._del_field, name='_proposals'),
                doc='Region proposals of an image')
            gt_instances = property(
                fget=partial(BaseDataSample._get_field, name='_gt_instances'),
                fset=partial(
                    BaseDataSample._set_field,
                    name='_gt_instances',
                    dtype=BaseDataElement),
                fdel=partial(BaseDataSample._del_field, name='_gt_instances'),
                doc='Ground truth instances of an image')
            pred_instances = property(
                fget=partial(
                    BaseDataSample._get_field, name='_pred_instances'),
                fset=partial(
                    BaseDataSample._set_field,
                    name='_pred_instances',
                    dtype=BaseDataElement),
                fdel=partial(
                    BaseDataSample._del_field, name='_pred_instances'),
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
        with self.assertRaises(AssertionError):
            det_sample.proposals = torch.rand((5, 4))
