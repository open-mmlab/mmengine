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
from torch.optim import SGD

from mmengine import MessageHub, MMLogger
from mmengine.dist import all_gather
from mmengine.optim import (AmpOptimizerWrapper, OptimizerWrapper,
                            multi_optims_gradient_accumulation)
from mmengine.testing import assert_allclose
from mmengine.testing._internal import MultiProcessTestCase


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


class TestOptimizerWrapper(MultiProcessTestCase):

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def run_test(self, test_name: str, parent_pipe) -> None:
        self.model = ToyModel()
        self.optimizer = SGD(self.model.parameters(), lr=0.1)
        self.logger = MMLogger.get_instance('test_optimizer_wrapper')
        self.message_hub = MessageHub.get_instance('test_optim_wrapper_init')
        super().run_test(test_name, parent_pipe)

    def test_init(self):
        optimizer_wrapper = OptimizerWrapper(self.optimizer)
        self.assertEqual(optimizer_wrapper.optimizer, self.optimizer)
        self.assertIsNone(optimizer_wrapper.grad_clip)
        self.assertFalse(optimizer_wrapper.detect_anomalous_params)
        self.assertEqual(optimizer_wrapper.cumulative_iters, 1)
        self.assertEqual(optimizer_wrapper.logger, self.logger)
        self.assertEqual(optimizer_wrapper.message_hub, self.message_hub)

    def test_update_params(self):
        # Test update params every iteration.
        optimizer_wrapper = OptimizerWrapper(
            self.optimizer, cumulative_iters=1)
        self._mock_method(optimizer_wrapper)
        loss = torch.tensor(1)
        optimizer_wrapper.update_params(loss)
        optimizer_wrapper.backward.assert_called_with(torch.tensor(1), None)
        optimizer_wrapper.step.assert_called_with()
        optimizer_wrapper.zero_grad.assert_called_with()

        # Test update params under gradient_accumulation context
        with optimizer_wrapper.gradient_accumulation(self.model, 2, 100):
            optimizer_wrapper.update_params(torch.tensor(1))
            optimizer_wrapper.backward.assert_called_with(
                torch.tensor(1), None)
            optimizer_wrapper.step.assert_called_with()
            optimizer_wrapper.zero_grad.assert_called_with()

        # It will raise an error if `cumulative_iters > 1` and
        # `gradient_accumulation` is not enabled.
        optimizer_wrapper = OptimizerWrapper(
            self.optimizer, cumulative_iters=3)
        self._mock_method(optimizer_wrapper)
        with self.assertRaises(AssertionError):
            optimizer_wrapper.update_params(loss)

        # `iter=0`, Call `optimizer_step` first time.
        with optimizer_wrapper.gradient_accumulation(
                self.model, cur_iter=0, max_iters=100):
            loss = torch.tensor(1)
            optimizer_wrapper.update_params(loss)
            optimizer_wrapper.backward.assert_called_with(
                torch.tensor(1) / 3, None)
            optimizer_wrapper.step.assert_not_called()
            optimizer_wrapper.zero_grad.assert_not_called()

        # `iter=2`, Call `optimizer_step` first time.
        with optimizer_wrapper.gradient_accumulation(
                self.model, cur_iter=2, max_iters=100):
            optimizer_wrapper.update_params(loss)
            optimizer_wrapper.step.assert_called()
            optimizer_wrapper.zero_grad.assert_called()
        self._mock_method(optimizer_wrapper)
        # Test end of training.
        with optimizer_wrapper.gradient_accumulation(
                self.model, cur_iter=99, max_iters=100):
            optimizer_wrapper.update_params(loss)
            optimizer_wrapper.step.assert_called()
            optimizer_wrapper.zero_grad.assert_called()
            optimizer_wrapper.backward.assert_called_with(1, None)

    def test_iter_status_initialized(self):
        optimizer = MagicMock()
        optimizer_wrapper = OptimizerWrapper(optimizer, cumulative_iters=3)
        optimizer_wrapper._iter_status_initialized(self.model)
        self.assertEqual(optimizer_wrapper.divisible_iters, 0)
        self.assertEqual(optimizer_wrapper.remainder_iters, 0)

        # Indivisible cur_iter will output warning.
        optimizer_wrapper = OptimizerWrapper(optimizer, cumulative_iters=3)
        optimizer_wrapper.cur_iter = 0
        optimizer_wrapper.max_iters = 100
        with self.assertLogs(self.logger) as cm:
            optimizer_wrapper._iter_status_initialized(self.model)
            self.assertEqual(len(cm.output), 1)
            self.assertRegex(cm.records[0].msg, 'Resume iter number is not')

        # Model with batch norm will output warning.
        optimizer_wrapper = OptimizerWrapper(optimizer, cumulative_iters=3)
        optimizer_wrapper.cur_iter = 0
        optimizer_wrapper.max_iters = 99
        model = nn.BatchNorm2d(1)
        with self.assertLogs(self.logger) as cm:
            optimizer_wrapper._iter_status_initialized(model)
            self.assertEqual(len(cm.output), 1)
            self.assertRegex(cm.records[0].msg, 'Gradient accumulative')

    def test_has_batch_norm(self):
        optimizer_wrapper = OptimizerWrapper(self.optimizer)
        self.assertFalse(optimizer_wrapper._has_batch_norm(self.model))
        self.assertTrue(optimizer_wrapper._has_batch_norm(nn.BatchNorm2d(1)))

    def test_backward(self):
        loss = MagicMock()
        optimizer_wrapper = OptimizerWrapper(self.optimizer)
        optimizer_wrapper.backward(loss)
        loss.backward.assert_called()

    def test_zero_grad(self):
        optimizer = MagicMock()
        optimizer_wrapper = OptimizerWrapper(optimizer)
        optimizer_wrapper.zero_grad()
        optimizer.zero_grad.assert_called()

    def test_step(self):
        optimizer = MagicMock()
        optimizer_wrapper = OptimizerWrapper(optimizer)
        optimizer_wrapper.step()
        optimizer.step.assert_called()

    def test_detect_anomalous_parameters(self):
        optimizer_wrapper = OptimizerWrapper(self.optimizer)
        loss = self.model(torch.Tensor(1, 1, 1, 1))

        with self.assertLogs(self.logger) as cm:
            optimizer_wrapper._detect_anomalous_params(loss, self.model)
            self.assertEqual(len(cm.output), 2)
            self.assertRegex(cm.records[0].msg, r'conv3.weight')
            self.assertRegex(cm.records[1].msg, r'conv3.bias')

    def test_clip_grads(self):
        optimizer_wrapper = OptimizerWrapper(
            self.optimizer, grad_clip=dict(max_norm=35))
        loss = self.model(torch.Tensor(1, 1, 1, 1))
        loss.backward()
        optimizer_wrapper._clip_grads()
        log_scalars = self.message_hub.log_scalars
        self.assertIn('train/grad_norm', log_scalars)

    def test_state_dict(self):
        optimizer_wrapper = OptimizerWrapper(self.optimizer)
        self.assertEqual(optimizer_wrapper.state_dict(),
                         self.optimizer.state_dict())

    def test_load_state_dict(self):
        optimizer_wrapper = OptimizerWrapper(self.optimizer)
        model = ToyModel()
        optimizer = SGD(model.parameters(), lr=0.1)
        optimizer_wrapper.load_state_dict(optimizer.state_dict())

        self.assertEqual(optimizer_wrapper.state_dict(),
                         optimizer.state_dict())

    def test_param_groups(self):
        optimizer_wrapper = OptimizerWrapper(self.optimizer)
        self.assertEqual(optimizer_wrapper.param_groups,
                         self.optimizer.param_groups)

    def test_gradient_accumulative(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ToyModel2()
        ddp_model = DistributedDataParallel(model)
        optimizer = SGD(ddp_model.parameters(), lr=0.01)
        optimizer_wrapper = OptimizerWrapper(optimizer, cumulative_iters=1)
        optimizer_wrapper.zero_grad()
        with optimizer_wrapper.gradient_accumulation(ddp_model, 0, 100):
            # Automatically sync grads if `cumulative_iters` = 1
            inputs = torch.randn(1, 1, 1, 1) * self.rank
            ddp_model(inputs).sum().backward()
            grad = model.conv.weight.grad
            all_grads = all_gather(grad)
            assert_allclose(all_grads[0], all_grads[1])

        # Do not sync grads when `optimizer_wrapper.cur_iter` cannot be
        # divided by `optimizer_wrapper.cumulative_iters`
        optimizer_wrapper = OptimizerWrapper(optimizer, cumulative_iters=3)
        with optimizer_wrapper.gradient_accumulation(ddp_model, 0, 100):
            ddp_model(inputs).sum().backward()
            all_grads = all_gather(model.conv.weight.grad)
            with self.assertRaises(AssertionError):
                assert_allclose(all_grads[0], all_grads[1])

        # sync grads if `cur_iter == 2`
        with optimizer_wrapper.gradient_accumulation(ddp_model, 2, 100):
            ddp_model(inputs).sum().backward()
            all_grads = all_gather(model.conv.weight.grad)
            assert_allclose(all_grads[0], all_grads[1])

    def test_precision_context(self):
        optimizer_wrapper = OptimizerWrapper(self.optimizer)
        with optimizer_wrapper.precision_context():
            pass

    def test_multi_optims_gradient_accumulation(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ToyModel2()
        ddp_model = DistributedDataParallel(model)
        optimizer = SGD(model.parameters(), lr=0.1)
        optimizer_wrapper = OptimizerWrapper(optimizer, cumulative_iters=3)
        data = torch.randn(1, 1, 1, 1) * self.rank
        # Test do not sync grads
        with multi_optims_gradient_accumulation(
                optimizer_wrapper, ddp_model, 0, 100):
            loss = ddp_model(data)
            optimizer_wrapper.update_params(loss)
            all_grads = all_gather(model.conv.weight.grad)
            with self.assertRaises(AssertionError):
                assert_allclose(all_grads[0], all_grads[1])

        # Test sync grads
        with multi_optims_gradient_accumulation(
                optimizer_wrapper, ddp_model, 2, 100):
            loss = ddp_model(data)
            optimizer_wrapper.update_params(loss)
            all_grads = all_gather(model.conv.weight.grad)
            assert_allclose(all_grads[0], torch.zeros_like(all_grads[0]))

        # Test sync grads if optimizer_wrapper is a dict
        with multi_optims_gradient_accumulation(
                dict(optim=optimizer_wrapper), ddp_model, 2, 100):
            loss = ddp_model(data)
            optimizer_wrapper.update_params(loss)
            all_grads = all_gather(model.conv.weight.grad)
            assert_allclose(all_grads[0], torch.zeros_like(all_grads[0]))

    def _init_dist_env(self, rank, world_size):
        """Initialize the distributed environment."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29515'
        os.environ['RANK'] = str(rank)
        torch_dist.init_process_group(
            backend='gloo', rank=rank, world_size=world_size)

    def _mock_method(self, optimizer_wrapper):
        optimizer_wrapper.backward = MagicMock()
        optimizer_wrapper.step = MagicMock()
        optimizer_wrapper.zero_grad = MagicMock()


class TestAmpOptimizerWrapper(TestCase):

    def setUp(self) -> None:
        self.model = ToyModel()
        self.optimizer = SGD(self.model.parameters(), lr=0.1)

    def test_init(self):
        # Test with default arguments.
        amp_optim_wrapper = AmpOptimizerWrapper(optimizer=self.optimizer)
        self.assertIsInstance(amp_optim_wrapper.loss_scalar, GradScaler)

        # Test with dynamic.
        amp_optim_wrapper = AmpOptimizerWrapper(
            'dynamic', optimizer=self.optimizer)
        self.assertIsNone(amp_optim_wrapper._scale_update_param)
        self.assertIsInstance(amp_optim_wrapper.loss_scalar, GradScaler)

        # Test with dict loss_scale.
        amp_optim_wrapper = AmpOptimizerWrapper(
            dict(init_scale=1, growth_factor=2),
            optimizer=self.optimizer)
        self.assertIsInstance(amp_optim_wrapper.loss_scalar, GradScaler)
        self.assertIsNone(amp_optim_wrapper._scale_update_param)
        with self.assertRaises(TypeError):
            AmpOptimizerWrapper(
                optimizer=self.optimizer,
                loss_scale='unknown',
                detect_anomalous_params=False)

    def test_step(self):
        optimizer = MagicMock()
        amp_optim_wrapper = AmpOptimizerWrapper(optimizer=optimizer)
        amp_optim_wrapper.loss_scalar = MagicMock()
        amp_optim_wrapper.step()
        amp_optim_wrapper.loss_scalar.step.assert_called_with(
            amp_optim_wrapper.optimizer)
        amp_optim_wrapper.loss_scalar.update.assert_called_with(
            amp_optim_wrapper._scale_update_param)

    def test_backward(self):
        for detect_anomalous_params in (True, False):
            amp_optim_wrapper = AmpOptimizerWrapper(optimizer=self.optimizer)
            amp_optim_wrapper.detect_anomalous_params = detect_anomalous_params
            loss_scalar = MagicMock()
            scale_return = MagicMock()
            scale_fn = MagicMock(return_value=scale_return)
            loss_scalar.scale = scale_fn
            amp_optim_wrapper.loss_scalar = loss_scalar
            amp_optim_wrapper._detect_anomalous_params = MagicMock()
            if not detect_anomalous_params:
                amp_optim_wrapper.backward(1)
                loss_scalar.scale.assert_called_with(1)
                scale_return.backward.assert_called_with()
            else:
                with self.assertRaises(AssertionError):
                    amp_optim_wrapper.backward(1)

    @unittest.skipIf(
        not torch.cuda.is_available(), reason='at lest need 1 gpu to test')
    def test_state_dict(self):
        self.model = self.model.cuda()
        amp_optim_wrapper = AmpOptimizerWrapper(optimizer=self.optimizer)
        with amp_optim_wrapper.precision_context():
            loss = self.model(torch.Tensor(1, 1, 1, 1).cuda())
            amp_optim_wrapper.update_params(loss)
            state_dict = amp_optim_wrapper.state_dict()
            optim_state_dict = dict()
            scalar_state_dict = dict()
            for key, value in state_dict.items():
                if key.startswith('loss_scalar.'):
                    scalar_state_dict[key.replace('loss_scalar.', '')] = value
                else:
                    optim_state_dict[key] = value
            self.assertDictEqual(optim_state_dict,
                                 amp_optim_wrapper.optimizer.state_dict())
            self.assertDictEqual(scalar_state_dict,
                                 amp_optim_wrapper.loss_scalar.state_dict())

    @unittest.skipIf(
        not torch.cuda.is_available(), reason='at lest need 1 gpu to test')
    def test_load_state_dict(self):
        amp_optim_wrapper = AmpOptimizerWrapper(optimizer=self.optimizer)
        self.model = self.model.cuda()
        # Test load from optimizer
        optimizer = SGD(self.model.parameters(), lr=0.1)
        amp_optim_wrapper.load_state_dict(optimizer.state_dict())

        self.assertDictEqual(optimizer.state_dict(),
                             amp_optim_wrapper.optimizer.state_dict())
        # Test load from optimizer_wrapper
        amp_optim_wrapper = AmpOptimizerWrapper(optimizer=self.optimizer)
        amp_optim_wrapper_ = AmpOptimizerWrapper(
            optimizer=SGD(self.model.parameters(), lr=0.1))
        amp_optim_wrapper_.load_state_dict(amp_optim_wrapper.state_dict())
        self.assertDictEqual(amp_optim_wrapper.optimizer.state_dict(),
                             amp_optim_wrapper_.optimizer.state_dict())
        self.assertDictEqual(amp_optim_wrapper.loss_scalar.state_dict(),
                             amp_optim_wrapper_.loss_scalar.state_dict())

    @unittest.skipIf(
        not torch.cuda.is_available(), reason='at lest need 1 gpu to test')
    def test_precision_context(self):
        amp_optim_wrapper = AmpOptimizerWrapper(optimizer=self.optimizer)
        with amp_optim_wrapper.precision_context():
            x = torch.randn(1, 1, 1, 1).cuda()
            y = nn.Conv2d(1, 1, 1).cuda()(x)
            self.assertEqual(y.dtype, torch.float16)

