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
from torch.optim import SGD, Optimizer

from mmengine import MessageHub, MMLogger
from mmengine.dist import all_gather
from mmengine.optim import AmpOptimWrapper, OptimWrapper
from mmengine.testing import assert_allclose
from mmengine.testing._internal import MultiProcessTestCase
from mmengine.utils import TORCH_VERSION, digit_version


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
    # Test `OptimWrapper.accumulate_grad` will block the gradient
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
        self.assertEqual(optim_wrapper.optimizer, self.optimizer)
        self.assertIsNone(optim_wrapper.clip_grad_kwargs)
        self.assertEqual(optim_wrapper.accumulative_iters, 1)
        self.assertIs(optim_wrapper.logger, self.logger)
        self.assertIs(optim_wrapper.message_hub, self.message_hub)

        with self.assertRaisesRegex(AssertionError,
                                    'If `clip_grad_kwargs` is not None'):
            OptimWrapper(self.optimizer, clip_grad=[])

    def test_update_params(self):
        # Test update params every iteration.
        optim_wrapper = OptimWrapper(self.optimizer, accumulative_iters=1)
        self._mock_method(optim_wrapper)
        loss = torch.tensor(1)
        optim_wrapper.update_params(loss)
        optim_wrapper.backward.assert_called_with(torch.tensor(1))
        optim_wrapper.step.assert_called_with()
        optim_wrapper.zero_grad.assert_called_with()

        with optim_wrapper.accumulate_grad(self.model, 2, 100):
            optim_wrapper.update_params(torch.tensor(1))
            optim_wrapper.backward.assert_called_with(torch.tensor(1))
            optim_wrapper.step.assert_called_with()
            optim_wrapper.zero_grad.assert_called_with()

        # It will raise an error if `accumulative_iters > 1` and
        # `accumulate_grad` is not enabled.
        optim_wrapper = OptimWrapper(self.optimizer, accumulative_iters=3)
        self._mock_method(optim_wrapper)
        with self.assertRaisesRegex(AssertionError,
                                    'gradient accumulation must be'):
            optim_wrapper.update_params(loss)

        # `iter=0`, Call `optimizer_step` first time.
        with optim_wrapper.accumulate_grad(
                self.model, cur_iter=0, max_iters=100):
            loss = torch.tensor(1)
            optim_wrapper.update_params(loss)
            optim_wrapper.backward.assert_called_with(torch.tensor(1) / 3)
            optim_wrapper.step.assert_not_called()
            optim_wrapper.zero_grad.assert_not_called()

        # `iter=2`, Call `optimizer_step` first time.
        with optim_wrapper.accumulate_grad(
                self.model, cur_iter=2, max_iters=100):
            optim_wrapper.update_params(loss)
            optim_wrapper.step.assert_called()
            optim_wrapper.zero_grad.assert_called()
        self._mock_method(optim_wrapper)
        # Test end of training.
        with optim_wrapper.accumulate_grad(
                self.model, cur_iter=99, max_iters=100):
            optim_wrapper.update_params(loss)
            optim_wrapper.step.assert_called()
            optim_wrapper.zero_grad.assert_called()
            optim_wrapper.backward.assert_called_with(1)

        # If ``accumulative_iters > 1``, call ``update_params`` with
        # non-accumulate_grad context will raise an Assertion error
        optim_wrapper = OptimWrapper(self.optimizer, accumulative_iters=1)
        optim_wrapper.accumulative_iters = 2
        with self.assertRaisesRegex(AssertionError,
                                    'gradient accumulation must be performed'):
            optim_wrapper.update_params(loss)

    def test_initilize_iter_status(self):
        optim_wrapper = OptimWrapper(self.optimizer, accumulative_iters=3)
        optim_wrapper._initilize_iter_status(self.model)
        self.assertEqual(optim_wrapper.divisible_iters, 0)
        self.assertEqual(optim_wrapper.remainder_iters, 0)

        # Indivisible cur_iter will output warning.
        optim_wrapper = OptimWrapper(self.optimizer, accumulative_iters=3)
        optim_wrapper.cur_iter = 0
        optim_wrapper.max_iters = 100
        with self.assertLogs(self.logger) as cm:
            optim_wrapper._initilize_iter_status(self.model)
            self.assertEqual(len(cm.output), 1)
            self.assertRegex(cm.records[0].msg, 'Resume iter number is not')

        # Model with batch norm will output warning.
        optim_wrapper = OptimWrapper(self.optimizer, accumulative_iters=3)
        optim_wrapper.cur_iter = 0
        optim_wrapper.max_iters = 99
        model = nn.BatchNorm2d(1)
        with self.assertLogs(self.logger) as cm:
            optim_wrapper._initilize_iter_status(model)
            self.assertEqual(len(cm.output), 1)
            self.assertRegex(cm.records[0].msg, 'Gradient accumulative')

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

    def test_accumulate_grad(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ToyModel2()
        ddp_model = DistributedDataParallel(model)
        optimizer = SGD(ddp_model.parameters(), lr=0.01)
        optim_wrapper = OptimWrapper(optimizer, accumulative_iters=1)
        optim_wrapper.zero_grad()
        with optim_wrapper.accumulate_grad(ddp_model, 0, 100):
            # Automatically sync grads if `accumulative_iters` = 1
            inputs = torch.randn(1, 1, 1, 1) * self.rank
            ddp_model(inputs).sum().backward()
            grad = model.conv.weight.grad
            all_grads = all_gather(grad)
            assert_allclose(all_grads[0], all_grads[1])

        # Do not sync grads when `optim_wrapper.cur_iter` cannot be
        # divided by `optim_wrapper.accumulative_iters`
        optim_wrapper = OptimWrapper(optimizer, accumulative_iters=3)
        with optim_wrapper.accumulate_grad(ddp_model, 0, 100):
            ddp_model(inputs).sum().backward()
            all_grads = all_gather(model.conv.weight.grad)
            with self.assertRaises(AssertionError):
                assert_allclose(all_grads[0], all_grads[1])

        # sync grads if `cur_iter == 2`
        with optim_wrapper.accumulate_grad(ddp_model, 2, 100):
            ddp_model(inputs).sum().backward()
            all_grads = all_gather(model.conv.weight.grad)
            assert_allclose(all_grads[0], all_grads[1])

    def test_precision_context(self):
        optim_wrapper = OptimWrapper(self.optimizer)
        with optim_wrapper.precision_context():
            pass

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
        optim_wrapper.backward = MagicMock()
        optim_wrapper.step = MagicMock()
        optim_wrapper.zero_grad = MagicMock()


class TestAmpOptimWrapper(TestCase):

    def setUp(self) -> None:
        self.model = ToyModel()
        self.optimizer = SGD(self.model.parameters(), lr=0.1)

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

    def test_step(self):
        optimizer = MagicMock(spec=Optimizer)
        amp_optim_wrapper = AmpOptimWrapper(optimizer=optimizer)
        amp_optim_wrapper.loss_scaler = MagicMock()
        amp_optim_wrapper.step()
        amp_optim_wrapper.loss_scaler.step.assert_called_with(
            amp_optim_wrapper.optimizer)
        amp_optim_wrapper.loss_scaler.update.assert_called_with(
            amp_optim_wrapper._scale_update_param)

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
        not torch.cuda.is_available() and torch,
        reason='at lest need 1 gpu to test')
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
        not torch.cuda.is_available() and torch,
        reason='at lest need 1 gpu to test')
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
    def test_precision_context(self):
        amp_optim_wrapper = AmpOptimWrapper(optimizer=self.optimizer)
        with amp_optim_wrapper.precision_context():
            x = torch.randn(1, 1, 1, 1).cuda()
            y = nn.Conv2d(1, 1, 1).cuda()(x)
            self.assertEqual(y.dtype, torch.float16)
