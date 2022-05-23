# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest
from unittest.mock import MagicMock

import torch
import torch.distributed as torch_dist
import torch.nn as nn
from torch.optim import SGD

from mmengine.dist import all_gather
from mmengine.logging import MessageHub
from mmengine.model import (BaseModel, MMDistributedDataParallel,
                            MMSeporateDDPWrapper)
from mmengine.optim import OptimizerWrapper
from mmengine.testing import assert_allclose
from mmengine.testing._internal import MultiProcessTestCase


class ToyModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)

    def forward(self, x, data_samples=None, return_loss=False):
        if return_loss:
            x = self.conv1(x)
            x = self.conv2(x)
            return dict(loss=x)
        else:
            return x


class ComplexModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, 1)
        self.conv2 = nn.Conv2d(3, 1, 1)

    def train_step(self, x, optimizer_wrapper):
        loss = self.conv1(x)
        optimizer_wrapper['optimizer_wrapper1'].optimizer_step(loss)
        loss = self.conv2(x)
        optimizer_wrapper['optimizer_wrapper2'].optimizer_step(loss)

    def val_step(self, x, return_loss=False):
        if return_loss:
            return 1
        else:
            return 2

    def test_step(self, x):
        return 3


class TestModelWrapper(MultiProcessTestCase):

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def test_train_step(self):
        self._init_dist_env(self.rank, self.world_size)
        message_hub = MessageHub.get_instance('TestModelWrapper')
        message_hub.update_info('iter', 0)
        message_hub.update_info('max_iters', 100)
        # Test `optimizer_wrapper` is a instance of `OptimizerWrapper`
        model = ToyModel()
        ddp_model = MMDistributedDataParallel(model)
        optimizer = SGD(ddp_model.parameters(), lr=0)
        optimizer_wrapper = OptimizerWrapper(
            ddp_model, optimizer, cumulative_iters=1)
        inputs = torch.randn(3, 1, 1) * self.rank * 255
        data = dict(inputs=inputs, data_sample=MagicMock())
        # Automatically sync grads if `cumulative_iters` = 1
        ddp_model.train_step([data], optimizer_wrapper=optimizer_wrapper)
        grad = ddp_model.module.conv1.weight.grad
        assert_allclose(grad, torch.zeros_like(grad))

        # Do not sync grads when `optimizer_wrapper.cur_iter` cannot be
        # divided by `optimizer_wrapper.cumulative_iters`
        optimizer_wrapper.cumulative_iters = 3
        with self.assertRaises(AssertionError):
            ddp_model.train_step([data], optimizer_wrapper=optimizer_wrapper)
            all_grads = all_gather(model.conv1.weight.grad)
            assert_allclose(all_grads[0], all_grads[1])

        # sync grads if `optimizer_wrapper.cur_iter` cannot be divided by
        # `optimizer_wrapper.cumulative_iters`
        message_hub.update_info('iter', 2)
        ddp_model.train_step([data], optimizer_wrapper=optimizer_wrapper)
        grad = ddp_model.module.conv1.weight.grad
        assert_allclose(grad, torch.zeros_like(grad))

    def test_val_step(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ToyModel()
        ddp_model = MMDistributedDataParallel(model)
        inputs = torch.randn(3, 1, 1) * self.rank * 255
        data = dict(inputs=inputs, data_sample=MagicMock())
        # Test get predictions.
        predictions = ddp_model.val_step([data], return_loss=False)
        self.assertIsInstance(predictions, torch.Tensor)
        # Test get losses.
        predictions = ddp_model.val_step([data], return_loss=True)
        self.assertIsInstance(predictions, dict)

    def test_test_step(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ToyModel()
        ddp_model = MMDistributedDataParallel(model)
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
class TestDynamicDDP(TestModelWrapper):

    def test_train_step(self):
        self._init_dist_env(self.rank, self.world_size)
        message_hub = MessageHub.get_instance('TestDynamicDDP')
        message_hub.update_info('iter', 0)
        message_hub.update_info('max_iters', 100)
        # Test `optimizer_wrapper` is a dict. In this case,
        # There will be two independently updated `DistributedDataParallel`
        # submodules.
        model = ComplexModel()
        ddp_model = MMSeporateDDPWrapper(model)
        optimizer1 = SGD(model.conv1.parameters(), lr=0.1)
        optimizer2 = SGD(model.conv1.parameters(), lr=0.2)
        optimizer_wrapper1 = OptimizerWrapper(model.conv1, optimizer1, 1)
        optimizer_wrapper2 = OptimizerWrapper(model.conv2, optimizer2, 2)
        optimizer_wrapper = dict(
            optimizer_wrapper1=optimizer_wrapper1,
            optimizer_wrapper2=optimizer_wrapper2)
        data = torch.randn(1, 3, 1, 1).cuda() * self.rank * 255

        # Automatically sync grads of `optimizer_wrapper1` since
        # `cumulative_iters` = 1
        ddp_model.train_step(data, optimizer_wrapper=optimizer_wrapper)
        grad1 = model.conv1.module.weight.grad
        assert_allclose(grad1, torch.zeros_like(grad1))

        # Do not sync grads of optimizer_wrapper2 since
        # `cumulative_iters` = 2
        grad2 = model.conv2.module.weight.grad
        all_grads2 = all_gather(grad2)
        with self.assertRaises(AssertionError):
            assert_allclose(all_grads2[0], all_grads2[1])

        # Automatically sync grads of `optimizer_wrapper1` and
        # `optimizer_wrapper2` since iter can be divided by
        # `cumulative_iters`.
        message_hub.update_info('iter', 1)
        ddp_model.train_step(data, optimizer_wrapper=optimizer_wrapper)
        grad1 = model.conv1.module.weight.grad
        assert_allclose(grad1, torch.zeros_like(grad1))
        grad2 = model.conv2.module.weight.grad
        all_grads2 = all_gather(grad2)
        assert_allclose(all_grads2[0], all_grads2[1])

    def test_val_step(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ComplexModel()
        ddp_model = MMSeporateDDPWrapper(model)
        data = torch.randn(3, 1, 1)
        # Test get predictions.
        predictions = ddp_model.val_step(data, return_loss=False)
        self.assertEqual(predictions, 2)
        # Test get losses.
        predictions = ddp_model.val_step(data, return_loss=True)
        self.assertEqual(predictions, 1)

    def test_test_step(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ComplexModel()
        ddp_model = MMSeporateDDPWrapper(model)
        data = torch.randn(3, 1, 1)
        # Test get predictions.
        predictions = ddp_model.test_step(data)
        self.assertEqual(predictions, 3)

    def _init_dist_env(self, rank, world_size):
        """Initialize the distributed environment."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29515'
        os.environ['RANK'] = str(rank)
        torch_dist.init_process_group(
            backend='gloo', rank=rank, world_size=world_size)
