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
        super().__init__(data_preprocessor)
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, batch_inputs, data_samples=None, mode='feat'):
        if mode == 'loss':
            out = self.conv(batch_inputs)
            return dict(loss=out)
        elif mode == 'predict':
            out = self.conv(batch_inputs)
            return out
        elif mode == 'feat':
            out = self.conv(batch_inputs)
            return out


class TestBaseModel(TestCase):

    def test_init(self):
        # initiate model without `preprocess_cfg`
        model = ToyModel()
        self.assertIsInstance(model.data_preprocessor, BaseDataPreprocessor)
        data_preprocessor = dict(type='CustomDataPreprocessor')
        model = ToyModel(data_preprocessor)
        self.assertIsInstance(model.data_preprocessor, CustomDataPreprocessor)
        self.assertEqual(model.data_preprocessor(1, training=True), 1)
        self.assertEqual(model.data_preprocessor(1, training=False), 2)

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
        inputs = torch.randn(3, 1, 1)
        data = dict(inputs=inputs)
        # initiate grad.
        # model.conv.weight.grad = torch.randn(1, 3, 1, 1)
        log_vars = model.train_step([data], optim_wrapper)
        self.assertIsNotNone(model.conv.weight.grad)
        self.assertIsInstance(log_vars['loss'], torch.Tensor)

    def test_val_step(self):
        inputs = torch.randn(3, 1, 1)
        data = dict(inputs=inputs)
        model = ToyModel()
        out = model.val_step([data])
        self.assertIsInstance(out, torch.Tensor)

    def test_test_step(self):
        inputs = torch.randn(3, 1, 1)
        data = dict(inputs=inputs)
        model = ToyModel()
        out = model.val_step([data])
        self.assertIsInstance(out, torch.Tensor)

    @unittest.skipIf(not torch.cuda.is_available(), 'cuda should be available')
    def test_cuda(self):
        inputs = torch.randn(3, 1, 1).cuda()
        data = dict(inputs=inputs)
        model = ToyModel().cuda()
        model.val_step([data])

    @unittest.skipIf(not torch.cuda.is_available(), 'cuda should be available')
    def test_to(self):
        inputs = torch.randn(3, 1, 1).cuda()
        data = dict(inputs=inputs)
        model = ToyModel().to(torch.cuda.current_device())
        model.val_step([data])
