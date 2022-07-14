# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest
from unittest.mock import MagicMock

import torch
import torch.distributed as torch_dist
import torch.nn as nn
from torch.optim import SGD

from mmengine.dist import all_gather
from mmengine.model import (BaseModel, MMDistributedDataParallel,
                            MMSeparateDistributedDataParallel)
from mmengine.model.averaged_model import ExponentialMovingAverage
from mmengine.optim import AmpOptimWrapper, OptimWrapper, OptimWrapperDict
from mmengine.testing import assert_allclose
from mmengine.testing._internal import MultiProcessTestCase
from mmengine.utils.parrots_wrapper import TORCH_VERSION
from mmengine.utils.version_utils import digit_version

if digit_version(TORCH_VERSION) >= digit_version('1.11.0'):
    from mmengine.model import MMFullyShardedDataParallel  # noqa: F401


class ToyModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)

    def forward(self, x, data_samples=None, mode='tensor'):
        x = self.conv1(x)
        x = self.conv2(x)
        if mode == 'loss':
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

    def train_step(self, data, optim_wrapper):
        batch_inputs, _ = self.data_preprocessor(data)
        loss1 = self.conv1(batch_inputs)
        optim_wrapper['optim_wrapper1'].update_params(loss1)
        loss2 = self.conv2(batch_inputs)
        optim_wrapper['optim_wrapper2'].update_params(loss2)
        return dict(loss1=loss1, loss2=loss2)

    def val_step(self, data):
        return 1

    def test_step(self, data):
        return 2

    def forward(self):
        pass


class TestDistributedDataParallel(MultiProcessTestCase):

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @unittest.skipIf(
        not torch.cuda.is_available(), reason='cuda should be available')
    def test_train_step(self):
        self._init_dist_env(self.rank, self.world_size)
        # Mixed precision training and gradient asynchronous should be valid at
        # the same time
        model = ToyModel().cuda()
        ddp_model = MMDistributedDataParallel(module=model)
        optimizer = SGD(ddp_model.parameters(), lr=0)
        optim_wrapper = AmpOptimWrapper(
            optimizer=optimizer, accumulative_counts=3)
        inputs = torch.randn(3, 1, 1).cuda() * self.rank * 255
        data = dict(inputs=[inputs], data_sample=None)
        res = ddp_model.train_step(data, optim_wrapper=optim_wrapper)['loss']
        self.assertIs(res.dtype, torch.float16)
        grad = ddp_model.module.conv1.weight.grad
        all_grads = all_gather(grad)
        with self.assertRaises(AssertionError):
            assert_allclose(all_grads[0], all_grads[1])

        # Gradient accumulation
        ddp_model.train_step(data, optim_wrapper=optim_wrapper)

        # Test update params and clean grads.
        ddp_model.train_step(data, optim_wrapper=optim_wrapper)
        grad = ddp_model.module.conv1.weight.grad
        all_grads = all_gather(grad)
        assert_allclose(all_grads[0], torch.zeros_like(all_grads[0]))
        assert_allclose(all_grads[1], torch.zeros_like(all_grads[0]))

    def test_val_step(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ToyModel()
        ddp_model = MMDistributedDataParallel(module=model)
        inputs = torch.randn(3, 1, 1) * self.rank * 255
        data = dict(inputs=[inputs], data_sample=None)
        # Test get predictions.
        predictions = ddp_model.val_step(data)
        self.assertIsInstance(predictions, torch.Tensor)

    def test_test_step(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ToyModel()
        ddp_model = MMDistributedDataParallel(module=model)
        inputs = torch.randn(3, 1, 1) * self.rank * 255
        data = dict(inputs=[inputs], data_sample=None)
        predictions = ddp_model.test_step(data)
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
class TestMMSeparateDistributedDataParallel(TestDistributedDataParallel):

    def test_init(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ComplexModel()
        model.ema = ExponentialMovingAverage(nn.Conv2d(1, 1, 1))
        model.act = nn.ReLU()
        ddp_model = MMSeparateDistributedDataParallel(model.cuda())
        self.assertIsInstance(ddp_model.module.ema, ExponentialMovingAverage)
        self.assertIsInstance(ddp_model.module.conv1,
                              MMDistributedDataParallel)
        self.assertIsInstance(ddp_model.module.act, nn.ReLU)

    def test_train_step(self):
        self._init_dist_env(self.rank, self.world_size)
        # Test `optim_wrapper` is a dict. In this case,
        # There will be two independently updated `DistributedDataParallel`
        # submodules.
        model = ComplexModel()
        ddp_model = MMSeparateDistributedDataParallel(model.cuda())
        optimizer1 = SGD(model.conv1.parameters(), lr=0.1)
        optimizer2 = SGD(model.conv1.parameters(), lr=0.2)
        optim_wrapper1 = OptimWrapper(optimizer1, 1)
        optim_wrapper2 = OptimWrapper(optimizer2, 1)
        optim_wrapper_dict = OptimWrapperDict(
            optim_wrapper1=optim_wrapper1, optim_wrapper2=optim_wrapper2)
        inputs = torch.randn(3, 1, 1).cuda() * self.rank * 255
        data = dict(inputs=[inputs], data_sample=None)
        # Automatically sync grads of `optim_wrapper1` since
        # `cumulative_iters` = 1
        ddp_model.train()
        self.assertTrue(ddp_model.training)
        ddp_model.train_step(data, optim_wrapper=optim_wrapper_dict)

    def test_val_step(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ComplexModel()
        ddp_model = MMSeparateDistributedDataParallel(model)
        data = torch.randn(3, 1, 1)
        # Test get predictions.
        ddp_model.eval()
        self.assertFalse(ddp_model.training)
        predictions = ddp_model.val_step(data)
        self.assertEqual(predictions, 1)

    def test_test_step(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ComplexModel()
        ddp_model = MMSeparateDistributedDataParallel(model)
        data = torch.randn(3, 1, 1)
        # Test get predictions.
        ddp_model.eval()
        self.assertFalse(ddp_model.training)
        predictions = ddp_model.test_step(data)
        self.assertEqual(predictions, 2)

    def _init_dist_env(self, rank, world_size):
        """Initialize the distributed environment."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29515'
        os.environ['RANK'] = str(rank)
        torch_dist.init_process_group(
            backend='gloo', rank=rank, world_size=world_size)


@unittest.skipIf(
    torch.cuda.device_count() < 2, reason='need 2 gpu to test fsdp')
@unittest.skipIf(
    digit_version(TORCH_VERSION) < digit_version('1.11.0'),
    reason='fsdp needs Pytorch 1.11 or higher')
class TestMMFullyShardedDataParallel(MultiProcessTestCase):

    def _init_dist_env(self, rank, world_size):
        """Initialize the distributed environment."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29520'
        os.environ['RANK'] = str(rank)

        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        torch_dist.init_process_group(
            backend='nccl', rank=rank, world_size=world_size)

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def test_train_step(self):
        self._init_dist_env(self.rank, self.world_size)
        # Test `optim_wrapper` is a instance of `OptimWrapper`
        model = ToyModel()
        fsdp_model = MMFullyShardedDataParallel(module=model.cuda())
        optimizer = SGD(fsdp_model.parameters(), lr=0)
        optim_wrapper = OptimWrapper(optimizer, accumulative_iters=1)
        inputs = torch.randn(3, 1, 1) * self.rank * 255
        data = dict(inputs=[inputs], data_sample=MagicMock())
        fsdp_model.train()
        self.assertTrue(fsdp_model.training)
        fsdp_model.train_step(data, optim_wrapper=optim_wrapper)

    def test_val_step(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ToyModel()
        fsdp_model = MMFullyShardedDataParallel(module=model.cuda())
        inputs = torch.randn(3, 1, 1) * self.rank * 255
        data = dict(inputs=[inputs], data_sample=MagicMock())
        # Test get predictions.
        predictions = fsdp_model.val_step(data)
        self.assertIsInstance(predictions, torch.Tensor)

    def test_test_step(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ToyModel()
        fsdp_model = MMFullyShardedDataParallel(module=model.cuda())
        inputs = torch.randn(3, 1, 1) * self.rank * 255
        data = dict(inputs=[inputs], data_sample=MagicMock())
        predictions = fsdp_model.test_step(data)
        self.assertIsInstance(predictions, torch.Tensor)
