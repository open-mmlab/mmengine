# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest
from unittest import TestCase
from unittest.mock import MagicMock

import torch
import torch.distributed as torch_dist
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import SGD, Adam, Optimizer

from mmengine.dist import all_gather
from mmengine.logging import MessageHub, MMLogger
from mmengine.optim import AmpOptimWrapper, OptimWrapper
from mmengine.testing import assert_allclose
from mmengine.testing._internal import MultiProcessTestCase
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)
        self.conv3 = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ToyModel2(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class TestOptimWrapper(MultiProcessTestCase):
    # Test `OptimWrapper.optim_context` will block the gradient
    # synchronization when using gradient accumulation strategy in distributed
    # data parallel training.
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def run_test(self, test_name: str, parent_pipe) -> None:
        self.model = ToyModel()
        self.optimizer = SGD(self.model.parameters(), lr=0.1)
        self.logger = MMLogger.get_instance('test_optim_wrapper')
        self.message_hub = MessageHub.get_instance('test_optim_wrapper_init')
        super().run_test(test_name, parent_pipe)

    def test_init(self):
        optim_wrapper = OptimWrapper(self.optimizer)
        self.assertIs(optim_wrapper.optimizer, self.optimizer)
        self.assertIsNone(optim_wrapper.clip_grad_kwargs)
        self.assertEqual(optim_wrapper._accumulative_counts, 1)
        self.assertIs(optim_wrapper.message_hub, self.message_hub)
        self.assertEqual(optim_wrapper._inner_count, 0)
        self.assertEqual(optim_wrapper._max_counts, -1)
        self.assertEqual(optim_wrapper._remainder_counts, -1)

        with self.assertRaisesRegex(AssertionError,
                                    'If `clip_grad` is not None'):
            OptimWrapper(self.optimizer, clip_grad=[])

    def test_update_params(self):
        # Test update params every iteration.
        optim_wrapper = OptimWrapper(self.optimizer, accumulative_counts=1)
        self._mock_method(optim_wrapper)
        loss = torch.tensor(1.)
        optim_wrapper.update_params(loss)
        self.assertEqual(optim_wrapper.scaled_loss, torch.tensor(1.))
        optim_wrapper.step.assert_called_with()
        optim_wrapper.zero_grad.assert_called_with()

        # Test gradient accumulation.
        optim_wrapper = OptimWrapper(self.optimizer, accumulative_counts=3)
        self._mock_method(optim_wrapper)
        # `iter=0`, accumulate gradient and do not update params.
        loss = torch.tensor(1.)
        optim_wrapper.update_params(loss)
        self.assertEqual(optim_wrapper.scaled_loss, torch.tensor(1.) / 3.)
        optim_wrapper.step.assert_not_called()
        optim_wrapper.zero_grad.assert_not_called()

        # gradient accumulate
        optim_wrapper.update_params(loss)
        self.assertEqual(optim_wrapper._inner_count, 2.)

        # `iter=2`, update params.
        optim_wrapper.update_params(loss)
        optim_wrapper.step.assert_called()
        optim_wrapper.zero_grad.assert_called()
        self._mock_method(optim_wrapper)

        # Test end of training without calling `initialize_iter_status`
        optim_wrapper._inner_count = 99
        optim_wrapper.update_params(loss)
        optim_wrapper.step.assert_not_called()
        optim_wrapper.zero_grad.assert_not_called()
        self.assertEqual(optim_wrapper.scaled_loss, torch.tensor(1.) / 3.)
        self._mock_method(optim_wrapper)

        # After calling `initialize_iter_status`, params will be updated at the
        # last iteration, and the `loss_scaler` will be adjusted.
        optim_wrapper.initialize_count_status(self.model, 99, 100)
        optim_wrapper.update_params(loss)
        optim_wrapper.step.assert_called()
        optim_wrapper.zero_grad.assert_called()
        self.assertEqual(optim_wrapper.scaled_loss, torch.tensor(1.))
        self._mock_method(optim_wrapper)

        # optim_wrapper.step should not be called at iteration 97 98, and the
        # loss factor should be 3 at iteration 99.
        optim_wrapper.initialize_count_status(self.model, 96, 100)
        for _ in range(2):
            optim_wrapper.update_params(loss)
            optim_wrapper.step.assert_not_called()
            optim_wrapper.zero_grad.assert_not_called()
        self.assertEqual(optim_wrapper.scaled_loss, torch.tensor(1.) / 3)

    def test_initialize_iter_status(self):
        optim_wrapper = OptimWrapper(self.optimizer, accumulative_counts=3)
        optim_wrapper.initialize_count_status(self.model, 0, 100)
        self.assertEqual(optim_wrapper._remainder_counts, 1)

        # Indivisible cur_iter will output warning.
        optim_wrapper = OptimWrapper(self.optimizer, accumulative_counts=3)
        with self.assertLogs(self.logger) as cm:
            optim_wrapper.initialize_count_status(self.model, 2, 100)
            self.assertEqual(len(cm.output), 1)
            self.assertRegex(cm.records[0].msg, 'Resumed iteration number')

        # Model with batch norm will output warning.
        optim_wrapper = OptimWrapper(self.optimizer, accumulative_counts=3)
        model = nn.BatchNorm2d(1)
        with self.assertLogs(self.logger) as cm:
            optim_wrapper.initialize_count_status(model, 0, 99)
            self.assertEqual(len(cm.output), 1)
            self.assertRegex(cm.records[0].msg, 'Gradient accumulative')

    def test_ger_lr(self):
        model = ToyModel()
        optim = SGD(model.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optim)
        self.assertEqual(optim_wrapper.get_lr(), dict(lr=[0.1]))

    def test_get_momentum(self):
        # Get momentum from SGD
        model = ToyModel()
        optim = SGD(model.parameters(), lr=0., momentum=0.8)
        optim_wrapper = OptimWrapper(optim)
        self.assertEqual(optim_wrapper.get_momentum(), dict(momentum=[0.8]))
        # Get momentum from Adam
        optim = Adam(model.parameters(), lr=0., betas=(0.9, 0.9))
        optim_wrapper = OptimWrapper(optim)
        self.assertEqual(optim_wrapper.get_momentum(), dict(momentum=[0.9]))

    def test_backward(self):
        loss = MagicMock()
        optim_wrapper = OptimWrapper(self.optimizer)
        optim_wrapper.backward(loss)
        loss.backward.assert_called()

    def test_zero_grad(self):
        optimizer = MagicMock(spec=Optimizer)
        optim_wrapper = OptimWrapper(optimizer)
        optim_wrapper.zero_grad()
        optimizer.zero_grad.assert_called()

    def test_step(self):
        optimizer = MagicMock(spec=Optimizer)
        optim_wrapper = OptimWrapper(optimizer)
        optim_wrapper.step()
        optimizer.step.assert_called()

    def test_clip_grads(self):
        optim_wrapper = OptimWrapper(
            self.optimizer, clip_grad=dict(max_norm=35))
        loss = self.model(torch.Tensor(1, 1, 1, 1))
        loss.backward()
        optim_wrapper._clip_grad()
        log_scalars = self.message_hub.log_scalars
        self.assertIn('train/grad_norm', log_scalars)

    def test_state_dict(self):
        optim_wrapper = OptimWrapper(self.optimizer)
        self.assertEqual(optim_wrapper.state_dict(),
                         self.optimizer.state_dict())

    def test_load_state_dict(self):
        optim_wrapper = OptimWrapper(self.optimizer)
        model = ToyModel()
        optimizer = SGD(model.parameters(), lr=0.1)
        optim_wrapper.load_state_dict(optimizer.state_dict())

        self.assertEqual(optim_wrapper.state_dict(), optimizer.state_dict())

    def test_param_groups(self):
        optim_wrapper = OptimWrapper(self.optimizer)
        self.assertEqual(optim_wrapper.param_groups,
                         self.optimizer.param_groups)

    def test_optim_context(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ToyModel2()
        ddp_model = DistributedDataParallel(model)
        optimizer = SGD(ddp_model.parameters(), lr=0.01)
        optim_wrapper = OptimWrapper(optimizer, accumulative_counts=1)
        optim_wrapper.zero_grad()

        # Automatically sync grads if `accumulative_counts` = 1
        optim_wrapper.initialize_count_status(model, 0, 100)
        inputs = torch.randn(1, 1, 1, 1) * self.rank
        ddp_model(inputs).sum().backward()
        grad = model.conv.weight.grad
        all_grads = all_gather(grad)
        assert_allclose(all_grads[0], all_grads[1])

        # Do not sync grads when `optim_wrapper.cur_iter` cannot be
        # divided by `optim_wrapper._accumulative_counts`
        optim_wrapper = OptimWrapper(optimizer, accumulative_counts=3)
        optim_wrapper.initialize_count_status(model, 0, 100)
        with optim_wrapper.optim_context(ddp_model):
            loss = ddp_model(inputs).sum()
        loss.backward()
        all_grads = all_gather(model.conv.weight.grad)
        with self.assertRaises(AssertionError):
            assert_allclose(all_grads[0], all_grads[1])

        # sync grads if `cur_iter == 2`
        optim_wrapper.initialize_count_status(model, 2, 100)
        with optim_wrapper.optim_context(ddp_model):
            loss = ddp_model(inputs).sum()
        loss.backward()
        all_grads = all_gather(model.conv.weight.grad)
        assert_allclose(all_grads[0], all_grads[1])

    def _init_dist_env(self, rank, world_size):
        """Initialize the distributed environment."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29515'
        os.environ['RANK'] = str(rank)
        torch_dist.init_process_group(
            backend='gloo', rank=rank, world_size=world_size)

    # TODO Test the real interface after add testing tool function which can
    #  test the function or method is read called.
    def _mock_method(self, optim_wrapper):

        def mock_methd(loss):
            optim_wrapper._inner_count += 1
            optim_wrapper.scaled_loss = loss

        optim_wrapper.backward = mock_methd
        optim_wrapper.step = MagicMock()
        optim_wrapper.zero_grad = MagicMock()


class TestAmpOptimWrapper(TestCase):

    def setUp(self) -> None:
        self.model = ToyModel()
        self.optimizer = SGD(self.model.parameters(), lr=0.1)

    @unittest.skipIf(
        not torch.cuda.is_available()
        and (digit_version(TORCH_VERSION) >= digit_version('1.6.0')),
        reason='`torch.cuda.amp` is only available when pytorch-gpu version '
        '>= 1.6')
    def test_init(self):
        # Test with default arguments.
        amp_optim_wrapper = AmpOptimWrapper(optimizer=self.optimizer)
        self.assertIsInstance(amp_optim_wrapper.loss_scaler, GradScaler)

        # Test with dynamic.
        amp_optim_wrapper = AmpOptimWrapper(
            'dynamic', optimizer=self.optimizer)
        self.assertIsNone(amp_optim_wrapper._scale_update_param)
        self.assertIsInstance(amp_optim_wrapper.loss_scaler, GradScaler)

        # Test with dict loss_scale.
        amp_optim_wrapper = AmpOptimWrapper(
            dict(init_scale=1, growth_factor=2), optimizer=self.optimizer)
        self.assertIsInstance(amp_optim_wrapper.loss_scaler, GradScaler)
        self.assertIsNone(amp_optim_wrapper._scale_update_param)
        with self.assertRaisesRegex(TypeError,
                                    'loss_scale must be of type float'):
            AmpOptimWrapper(optimizer=self.optimizer, loss_scale='unknown')

    @unittest.skipIf(
        not torch.cuda.is_available()
        and (digit_version(TORCH_VERSION) >= digit_version('1.6.0')),
        reason='`torch.cuda.amp` is only available when pytorch-gpu version '
        '>= 1.6')
    def test_step(self):
        optimizer = MagicMock(spec=Optimizer)
        amp_optim_wrapper = AmpOptimWrapper(optimizer=optimizer)
        amp_optim_wrapper.loss_scaler = MagicMock()
        amp_optim_wrapper.step()
        amp_optim_wrapper.loss_scaler.step.assert_called_with(
            amp_optim_wrapper.optimizer)
        amp_optim_wrapper.loss_scaler.update.assert_called_with(
            amp_optim_wrapper._scale_update_param)

    @unittest.skipIf(
        not torch.cuda.is_available()
        and (digit_version(TORCH_VERSION) >= digit_version('1.6.0')),
        reason='`torch.cuda.amp` is only available when pytorch-gpu version '
        '>= 1.6')
    def test_backward(self):
        amp_optim_wrapper = AmpOptimWrapper(optimizer=self.optimizer)
        loss_scaler = MagicMock()
        scale_return = MagicMock()
        scale_fn = MagicMock(return_value=scale_return)
        loss_scaler.scale = scale_fn
        amp_optim_wrapper.loss_scaler = loss_scaler

        amp_optim_wrapper.backward(1)
        loss_scaler.scale.assert_called_with(1)
        scale_return.backward.assert_called_with()

    @unittest.skipIf(
        not torch.cuda.is_available()
        and (digit_version(TORCH_VERSION) >= digit_version('1.6.0')),
        reason='`torch.cuda.amp` is only available when pytorch-gpu version '
        '>= 1.6')
    def test_state_dict(self):
        self.model = self.model.cuda()
        amp_optim_wrapper = AmpOptimWrapper(optimizer=self.optimizer)
        loss = self.model(torch.Tensor(1, 1, 1, 1).cuda())
        amp_optim_wrapper.update_params(loss)
        state_dict = amp_optim_wrapper.state_dict()
        scalar_state_dict = state_dict.pop('loss_scaler')
        optim_state_dict = state_dict

        self.assertDictEqual(optim_state_dict,
                             amp_optim_wrapper.optimizer.state_dict())
        self.assertDictEqual(scalar_state_dict,
                             amp_optim_wrapper.loss_scaler.state_dict())

    @unittest.skipIf(
        not torch.cuda.is_available()
        and (digit_version(TORCH_VERSION) >= digit_version('1.6.0')),
        reason='`torch.cuda.amp` is only available when pytorch-gpu version '
        '>= 1.6')
    def test_load_state_dict(self):
        amp_optim_wrapper = AmpOptimWrapper(optimizer=self.optimizer)
        self.model = self.model.cuda()
        # Test load from optimizer
        optimizer = SGD(self.model.parameters(), lr=0.1)
        amp_optim_wrapper.load_state_dict(optimizer.state_dict())

        self.assertDictEqual(optimizer.state_dict(),
                             amp_optim_wrapper.optimizer.state_dict())
        # Test load from optim_wrapper
        amp_optim_wrapper = AmpOptimWrapper(optimizer=self.optimizer)
        amp_optim_wrapper_ = AmpOptimWrapper(
            optimizer=SGD(self.model.parameters(), lr=0.1))
        amp_optim_wrapper_.load_state_dict(amp_optim_wrapper.state_dict())
        self.assertDictEqual(amp_optim_wrapper.optimizer.state_dict(),
                             amp_optim_wrapper_.optimizer.state_dict())
        self.assertDictEqual(amp_optim_wrapper.loss_scaler.state_dict(),
                             amp_optim_wrapper_.loss_scaler.state_dict())

    @unittest.skipIf(
        not torch.cuda.is_available()
        and (digit_version(TORCH_VERSION) >= digit_version('1.6.0')),
        reason='`torch.cuda.amp` is only available when pytorch-gpu version '
        '>= 1.6')
    def test_optim_context(self):
        amp_optim_wrapper = AmpOptimWrapper(optimizer=self.optimizer)
        with amp_optim_wrapper.optim_context(self.model):
            x = torch.randn(1, 1, 1, 1).cuda()
            y = nn.Conv2d(1, 1, 1).cuda()(x)
            self.assertEqual(y.dtype, torch.float16)
