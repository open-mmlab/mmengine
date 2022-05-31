# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest
from unittest.mock import MagicMock

import torch
import torch.distributed as torch_dist
import torch.nn as nn
from torch.optim import SGD

from mmengine.model import (BaseModel, MMDistributedDataParallel,
                            MMSeparateDDPWrapper)
from mmengine.optim import OptimWrapper, OptimWrapperDict
from mmengine.testing import assert_allclose
from mmengine.testing._internal import MultiProcessTestCase


class ToyModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)

    def forward(self, x, data_samples=None, mode='feat'):
        if mode == 'loss':
            x = self.conv1(x)
            x = self.conv2(x)
            return dict(loss=x)
        elif mode == 'predict':
            return x
        else:
            return x


class ComplexModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, 1)
        self.conv2 = nn.Conv2d(3, 1, 1)

    def train_step(self, data, optimizer_wrapper):
        batch_inputs, _ = self.data_preprocessor(data)
        loss1 = self.conv1(batch_inputs)
        optimizer_wrapper['optimizer_wrapper1'].update_params(loss1)
        loss2 = self.conv2(batch_inputs)
        optimizer_wrapper['optimizer_wrapper2'].update_params(loss2)
        return dict(loss1=loss1, loss2=loss2)

    def val_step(self, data):
        return 1

    def test_step(self, data):
        return 2


class TestModelWrapper(MultiProcessTestCase):

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def test_train_step(self):
        self._init_dist_env(self.rank, self.world_size)
        # Test `optimizer_wrapper` is a instance of `OptimWrapper`
        model = ToyModel()
        ddp_model = MMDistributedDataParallel(module=model)
        optimizer = SGD(ddp_model.parameters(), lr=0)
        optimizer_wrapper = OptimWrapper(optimizer, accumulative_iters=1)
        inputs = torch.randn(3, 1, 1) * self.rank * 255
        data = dict(inputs=inputs, data_sample=MagicMock())
        ddp_model.train_step([data], optimizer_wrapper=optimizer_wrapper)
        grad = ddp_model.module.conv1.weight.grad
        assert_allclose(grad, torch.zeros_like(grad))

    def test_val_step(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ToyModel()
        ddp_model = MMDistributedDataParallel(module=model)
        inputs = torch.randn(3, 1, 1) * self.rank * 255
        data = dict(inputs=inputs, data_sample=MagicMock())
        # Test get predictions.
        predictions = ddp_model.val_step([data])
        self.assertIsInstance(predictions, torch.Tensor)

    def test_test_step(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ToyModel()
        ddp_model = MMDistributedDataParallel(module=model)
        inputs = torch.randn(3, 1, 1) * self.rank * 255
        data = dict(inputs=inputs, data_sample=MagicMock())
        predictions = ddp_model.test_step([data])
        self.assertIsInstance(predictions, torch.Tensor)

    def _init_dist_env(self, rank, world_size):
        """Initialize the distributed environment."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29510'
        os.environ['RANK'] = str(rank)
        torch_dist.init_process_group(
            backend='gloo', rank=rank, world_size=world_size)


@unittest.skipIf(
    not torch.cuda.is_available(), reason='cuda should be available')
class TestMMSeparateDDPWrapper(TestModelWrapper):

    def test_train_step(self):
        self._init_dist_env(self.rank, self.world_size)
        # Test `optimizer_wrapper` is a dict. In this case,
        # There will be two independently updated `DistributedDataParallel`
        # submodules.
        model = ComplexModel()
        ddp_model = MMSeparateDDPWrapper(model.cuda())
        optimizer1 = SGD(model.conv1.parameters(), lr=0.1)
        optimizer2 = SGD(model.conv1.parameters(), lr=0.2)
        optimizer_wrapper1 = OptimWrapper(optimizer1, 1)
        optimizer_wrapper2 = OptimWrapper(optimizer2, 1)
        optim_wrapper_dict = OptimWrapperDict(
            dict(
                optimizer_wrapper1=optimizer_wrapper1,
                optimizer_wrapper2=optimizer_wrapper2))
        inputs = torch.randn(3, 1, 1).cuda() * self.rank * 255
        data = dict(inputs=inputs)
        # Automatically sync grads of `optimizer_wrapper1` since
        # `cumulative_iters` = 1
        ddp_model.train_step([data], optimizer_wrapper=optim_wrapper_dict)

    def test_val_step(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ComplexModel()
        ddp_model = MMSeparateDDPWrapper(model)
        data = torch.randn(3, 1, 1)
        # Test get predictions.
        predictions = ddp_model.val_step([data])
        self.assertEqual(predictions, 1)

    def test_test_step(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ComplexModel()
        ddp_model = MMSeparateDDPWrapper(model)
        data = torch.randn(3, 1, 1)
        # Test get predictions.
        predictions = ddp_model.test_step(data)
        self.assertEqual(predictions, 2)

    def _init_dist_env(self, rank, world_size):
        """Initialize the distributed environment."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29515'
        os.environ['RANK'] = str(rank)
        torch_dist.init_process_group(
            backend='gloo', rank=rank, world_size=world_size)
