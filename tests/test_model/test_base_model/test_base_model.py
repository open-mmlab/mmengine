# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
import torch.nn as nn
from torch.optim import SGD

from mmengine.model import BaseDataPreprocessor, BaseModel
from mmengine.optim import OptimWrapper
from mmengine.registry import MODELS
from mmengine.testing import assert_allclose


@MODELS.register_module()
class CustomDataPreprocessor(BaseDataPreprocessor):

    def forward(self, data, training=False):
        if training:
            return 1
        else:
            return 2


class ToyModel(BaseModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=None)
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, inputs, data_sample=None, mode='tensor'):
        if mode == 'loss':
            out = self.conv(inputs)
            return dict(loss=out)
        elif mode == 'predict':
            out = self.conv(inputs)
            return out
        elif mode == 'tensor':
            out = self.conv(inputs)
            return out


class NestedModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.toy_model = ToyModel()

    def forward(self):
        pass


class TestBaseModel(TestCase):

    def test_init(self):
        # initiate model without `data_preprocessor`
        model = ToyModel()
        self.assertIsInstance(model.data_preprocessor, BaseDataPreprocessor)
        data_preprocessor = dict(type='CustomDataPreprocessor')
        model = ToyModel(data_preprocessor=data_preprocessor)
        self.assertIsInstance(model.data_preprocessor, CustomDataPreprocessor)
        self.assertEqual(model.data_preprocessor(1, training=True), 1)
        self.assertEqual(model.data_preprocessor(1, training=False), 2)

        # initiate model with built `data_preprocessor`.
        data_preprocessor = CustomDataPreprocessor()
        model = ToyModel(data_preprocessor=data_preprocessor)
        self.assertIs(model.data_preprocessor, data_preprocessor)

        # initiate model with error type `data_preprocessor`.
        with self.assertRaisesRegex(TypeError, 'data_preprocessor should be'):
            ToyModel(data_preprocessor=[data_preprocessor])

    def test_parse_losses(self):
        model = ToyModel()
        loss_cls = torch.tensor(1, dtype=torch.float32)
        loss_list = [
            torch.tensor(2, dtype=torch.float32),
            torch.tensor(3, dtype=torch.float32)
        ]
        losses = dict(loss_cls=loss_cls, loss_list=loss_list)
        target_parsed_losses = torch.tensor(6, dtype=torch.float32)
        targe_log_vars = dict(
            loss=torch.tensor(6, dtype=torch.float32),
            loss_cls=torch.tensor(1, dtype=torch.float32),
            loss_list=torch.tensor(5, dtype=torch.float32))
        parse_losses, log_vars = model.parse_losses(losses)
        assert_allclose(parse_losses, target_parsed_losses)
        for key in log_vars:
            self.assertIn(key, targe_log_vars)
            assert_allclose(log_vars[key], targe_log_vars[key])

        with self.assertRaises(TypeError):
            losses['error_key'] = dict()
            model.parse_losses(losses)

    def test_train_step(self):
        model = ToyModel()
        optimizer = SGD(model.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        inputs = torch.randn(1, 3, 1, 1)
        data = dict(inputs=inputs, data_sample=None)
        # initiate grad.
        # model.conv.weight.grad = torch.randn(1, 3, 1, 1)
        log_vars = model.train_step(data, optim_wrapper)
        self.assertIsNotNone(model.conv.weight.grad)
        self.assertIsInstance(log_vars['loss'], torch.Tensor)

    def test_val_step(self):
        inputs = torch.randn(1, 3, 1, 1)
        data = dict(inputs=inputs, data_sample=None)
        model = ToyModel()
        out = model.val_step(data)
        self.assertIsInstance(out, torch.Tensor)

    def test_test_step(self):
        inputs = torch.randn(1, 3, 1, 1)
        data = dict(inputs=inputs, data_sample=None)
        model = ToyModel()
        out = model.val_step(data)
        self.assertIsInstance(out, torch.Tensor)

    @unittest.skipIf(not torch.cuda.is_available(), 'cuda should be available')
    def test_cuda(self):
        inputs = torch.randn(1, 3, 1, 1).cuda()
        data = dict(inputs=inputs, data_sample=None)
        model = ToyModel().cuda()
        out = model.val_step(data)
        self.assertEqual(out.device.type, 'cuda')

        model = NestedModel()
        self.assertEqual(model.data_preprocessor._device, torch.device('cpu'))
        self.assertEqual(model.toy_model.data_preprocessor._device,
                         torch.device('cpu'))
        model.cuda()
        self.assertEqual(model.data_preprocessor._device, torch.device('cuda'))
        self.assertEqual(model.toy_model.data_preprocessor._device,
                         torch.device('cuda'))

    @unittest.skipIf(not torch.cuda.is_available(), 'cuda should be available')
    def test_to(self):
        inputs = torch.randn(1, 3, 1, 1).to('cuda:0')
        data = dict(inputs=inputs, data_sample=None)
        model = ToyModel().to(torch.cuda.current_device())
        out = model.val_step(data)
        self.assertEqual(out.device.type, 'cuda')

        model = NestedModel()
        self.assertEqual(model.data_preprocessor._device, torch.device('cpu'))
        self.assertEqual(model.toy_model.data_preprocessor._device,
                         torch.device('cpu'))
        model.to('cuda')
        self.assertEqual(model.data_preprocessor._device, torch.device('cuda'))
        self.assertEqual(model.toy_model.data_preprocessor._device,
                         torch.device('cuda'))

        model.to()
        self.assertEqual(model.data_preprocessor._device, torch.device('cuda'))
        self.assertEqual(model.toy_model.data_preprocessor._device,
                         torch.device('cuda'))
