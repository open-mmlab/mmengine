# Copyright (c) OpenMMLab. All rights reserved.
import random

import pytest
import torch

from mmengine.data import BaseDataElement


class TestBaseDataElement:

    def setup_data(self):
        metainfo = dict(
            img_id=random.randint(0, 100),
            img_shape=(random.randint(400, 600), random.randint(400, 600)))
        data = dict(bboxes=torch.rand((5, 4)), scores=torch.rand((5, )))
        return metainfo, data

    def check_key_existence(self, instances, metainfo=None, data=None):
        # check the existence of keys in metainfo, data, and instances
        if metainfo:
            for k, v in metainfo.items():
                assert k in instances
                assert k in instances.keys()
                assert k in instances.metainfo_keys()
                assert k not in instances.data_keys()
                assert instances.get(k) == v
        if data:
            for k, v in data.items():
                assert k in instances
                assert k in instances.keys()
                assert k not in instances.metainfo_keys()
                assert k in instances.data_keys()
                assert instances.get(k) == v

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
        assert instances.bboxes == data['bboxes']
        assert instances.scores == data['scores']
        assert instances.img_id == metainfo['img_id']
        assert instances.img_shape == metainfo['img_shape']
        self.check_key_existence(instances, metainfo, data)

        # initialization with args
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo, data)
        assert instances.bboxes == data['bboxes']
        assert instances.scores == data['scores']
        assert instances.img_id == metainfo['img_id']
        assert instances.img_shape == metainfo['img_shape']
        self.check_key_existence(instances, metainfo, data)

    def test_new(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo=metainfo, data=data)

        # test new() with no arguments
        new_instances = instances.new()
        assert new_instances.bboxes == data['bboxes']
        assert new_instances.scores == data['scores']
        assert new_instances.img_id == metainfo['img_id']
        assert new_instances.img_shape == metainfo['img_shape']

        # test new() with arguments
        metainfo, data = self.setup_data()
        new_instances = instances.new(metainfo=metainfo, data=data)
        assert new_instances.bboxes == data['bboxes']
        assert new_instances.scores == data['scores']
        assert new_instances.img_id == metainfo['img_id']
        assert new_instances.img_shape == metainfo['img_shape']

    def test_set_metainfo(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement()
        instances.set_metainfo(metainfo)
        self.check_key_existence(instances, metainfo=metainfo)

        assert instances.img_shape == metainfo['img_shape']
        assert instances.img_id == metainfo['img_id']

    def test_set_data(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement()

        instances.bboxes = data['bboxes']
        instances.scores = data['scores']
        self.check_key_existence(instances, data=data)

        # a.xx only set data rather than metainfo
        instances.img_shape = metainfo['img_shape']
        instances.img_id = metainfo['img_id']
        self.check_key_existence(instances, data=metainfo)

    def test_delete_modify(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo, data)

        new_metainfo, new_data = self.setup_data()
        instances.bboxes = new_data['bboxes']
        instances.scores = new_data['scores']
        instances.img_id = new_metainfo['img_id']
        instances.img_shape = new_metainfo['img_shape']

        assert instances.bboxes == new_data['bboxes']
        assert instances.scores == new_data['scores']
        assert instances.img_id == new_metainfo['img_id']
        assert instances.img_shape == new_metainfo['img_shape']

        assert instances.bboxes != data['bboxes']
        assert instances.scores != data['scores']
        assert instances.img_id != metainfo['img_id']
        assert instances.img_shape != metainfo['img_shape']

        del instances.bboxes
        del instances.scores
        assert 'bboxes' not in instances
        assert 'scores' not in instances
        assert instances.pop('bboxes', None) is None
        assert instances.pop('scores', 'abcdef') == 'abcdef'

        assert instances.pop('img_shape') == new_metainfo['img_shape']
        assert instances.pop('img_id') == new_metainfo['img_id']

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='GPU is required!')
    def test_cuda(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo, data)

        cuda_instances = instances.cuda()
        assert instances.device == 'cpu'
        assert instances.bboxes.device == 'cpu'
        assert instances.scores.device == 'cpu'
        assert cuda_instances.device == 'cuda:0'
        assert cuda_instances.bboxes.device == 'cuda:0'
        assert cuda_instances.scores.device == 'cuda:0'

        cpu_instances = cuda_instances.cpu()
        assert cpu_instances.device == 'cpu'
        assert cpu_instances.bboxes.device == 'cpu'
        assert cpu_instances.scores.device == 'cpu'
        del cuda_instances

        cuda_instances = instances.to('cuda:0')
        assert cuda_instances.device == 'cuda:0'
        assert cuda_instances.bboxes.device == 'cuda:0'
        assert cuda_instances.scores.device == 'cuda:0'

    def test_cpu(self):
        metainfo, data = self.setup_data()
        instances = BaseDataElement(metainfo, data)

        cpu_instances = instances.cpu()
        assert cpu_instances.device == 'cpu'
        assert cpu_instances.bboxes.device == 'cpu'
        assert cpu_instances.scores.device == 'cpu'
