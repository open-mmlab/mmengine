# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock
import unittest
import os

import torch
import torch.nn as nn
from torch.optim import SGD
import torch.distributed as torch_dist
from torch.nn.parallel.distributed import DistributedDataParallel

from mmengine.testing._internal import MultiProcessTestCase
from mmengine import MessageHub, MMLogger
from mmengine.optim import OptimizerWrapper, AmpOptimizerWrapper
from mmengine.dist import all_gather
from mmengine.testing import assert_allclose


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
        self.model = ToyModel()
        self.optimizer = SGD(self.model.parameters(), lr=0.1)
        super().setUp()
        self._spawn_processes()

    def run_test(self, test_name: str, parent_pipe) -> None:
        self.logger = MMLogger.get_instance('test_optimizer_wrapper')
        self.message_hub = MessageHub.get_instance('test_optim_wrapper_init')
        self.message_hub.update_info('iter', 0)
        self.message_hub.update_info('max_iters', 100)
        super().run_test(test_name, parent_pipe)

    def test_init(self):
        optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)
        self.assertEqual(optimizer_wrapper.optimizer, self.optimizer)
        self.assertEqual(optimizer_wrapper.model, self.model)
        self.assertIsNone(optimizer_wrapper.grad_clip)
        self.assertFalse(optimizer_wrapper.detect_anomalous_params)
        self.assertFalse(optimizer_wrapper.initialized)
        self.assertEqual(optimizer_wrapper.cumulative_iters, 1)
        self.assertEqual(optimizer_wrapper.logger, self.logger)
        self.assertEqual(optimizer_wrapper.message_hub, self.message_hub)

    def test_optimizer_step(self):
        optimizer_wrapper = OptimizerWrapper(
            self.model, self.optimizer, cumulative_iters=3)
        # Calculate `divisible_iters` and `remainder_iters`.
        optimizer_wrapper._parse_cumulative_iters()
        optimizer_wrapper.initialized = False
        # Mock method
        optimizer_wrapper.backward = MagicMock()
        optimizer_wrapper.step = MagicMock()
        optimizer_wrapper.zero_grad = MagicMock()
        optimizer_wrapper._parse_cumulative_iters = MagicMock()
        # `iter=0`, Call `optimizer_step` first time.
        loss = torch.tensor(1)
        optimizer_wrapper.optimizer_step(loss)
        optimizer_wrapper._parse_cumulative_iters.assert_called()
        optimizer_wrapper.backward.assert_called_with(torch.tensor(1) / 3)
        optimizer_wrapper.step.assert_not_called()
        optimizer_wrapper.zero_grad.assert_not_called()
        optimizer_wrapper.initialized = True
        # reset MagicMock
        optimizer_wrapper._parse_cumulative_iters = MagicMock()
        # `iter=2`, Call `optimizer_step` first time.
        self.message_hub.update_info('iter', 2)
        optimizer_wrapper.optimizer_step(loss)
        optimizer_wrapper._parse_cumulative_iters.assert_not_called()
        optimizer_wrapper.step.assert_called()
        optimizer_wrapper.zero_grad.assert_called()
        # Reset MagicMock.
        optimizer_wrapper.step = MagicMock()
        optimizer_wrapper.zero_grad = MagicMock()
        # Test end of training.
        self.message_hub.update_info('iter', 99)
        optimizer_wrapper.optimizer_step(loss)
        optimizer_wrapper.step.assert_called()
        optimizer_wrapper.zero_grad.assert_called()
        optimizer_wrapper.backward.assert_called_with(1)

    def test_parse_cumulative_iters(self):
        optimizer = MagicMock()
        optimizer_wrapper = OptimizerWrapper(
            self.model, optimizer, cumulative_iters=3)

        self.assertFalse(optimizer_wrapper.initialized)
        optimizer_wrapper._parse_cumulative_iters()
        self.assertEqual(optimizer_wrapper.divisible_iters, 99)
        self.assertEqual(optimizer_wrapper.remainder_iters, 1)
        self.assertTrue(optimizer_wrapper.initialized)
        optimizer.zero_grad.assert_called()

        # Indivisible cur_iter will output warning.
        self.message_hub.update_info('iter', 1)
        self.message_hub.update_info('max_iters', 100)
        with self.assertLogs(self.logger) as cm:
            optimizer_wrapper._parse_cumulative_iters()
            self.assertEqual(len(cm.output), 1)
            self.assertRegex(cm.records[0].msg, 'Resume iter number is not')

        # Model with batch norm will output warning.
        self.message_hub.update_info('iter', 0)
        optimizer_wrapper.model = nn.BatchNorm2d(1)
        with self.assertLogs(self.logger) as cm:
            optimizer_wrapper._parse_cumulative_iters()
            self.assertEqual(len(cm.output), 1)
            self.assertRegex(cm.records[0].msg, 'Gradient accumulative')

    def test_has_batch_norm(self):
        optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)
        self.assertFalse(optimizer_wrapper._has_batch_norm(self.model))
        self.assertTrue(optimizer_wrapper._has_batch_norm(nn.BatchNorm2d(1)))

    def test_backward(self):
        loss = MagicMock()
        optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)
        optimizer_wrapper.backward(loss)
        loss.backward.assert_called()

    def test_zero_grad(self):
        optimizer = MagicMock()
        optimizer_wrapper = OptimizerWrapper(self.model, optimizer)
        optimizer_wrapper.zero_grad()
        optimizer.zero_grad.assert_called()

    def test_step(self):
        optimizer = MagicMock()
        optimizer_wrapper = OptimizerWrapper(self.model, optimizer)
        optimizer_wrapper.step()
        optimizer.step.assert_called()

    def test_detect_anomalous_parameters(self):
        optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)
        loss = self.model(torch.Tensor(1, 1, 1, 1))

        with self.assertLogs(self.logger) as cm:
            optimizer_wrapper._detect_anomalous_params(loss)
            self.assertEqual(len(cm.output), 2)
            self.assertRegex(cm.records[0].msg, r'conv3.weight')
            self.assertRegex(cm.records[1].msg, r'conv3.bias')

    def test_clip_grads(self):
        optimizer_wrapper = OptimizerWrapper(
            self.model, self.optimizer, grad_clip=dict(max_norm=35))
        loss = self.model(torch.Tensor(1, 1, 1, 1))
        loss.backward()
        optimizer_wrapper._clip_grads()
        log_scalars = self.message_hub.log_scalars
        self.assertIn('train/grad_norm', log_scalars)

    def test_state_dict(self):
        optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)
        self.assertEqual(optimizer_wrapper.state_dict(),
                         self.optimizer.state_dict())

    def test_load_state_dict(self):
        optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)
        model = ToyModel()
        optimizer = SGD(model.parameters(), lr=0.1)
        optimizer_wrapper.load_state_dict(optimizer.state_dict())

        self.assertEqual(optimizer_wrapper.state_dict(),
                         optimizer.state_dict())

    def test_param_groups(self):
        optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)
        self.assertEqual(optimizer_wrapper.param_groups,
                         self.optimizer.param_groups)

    def test_sync_grads(self):
        self._init_dist_env(self.rank, self.world_size)
        # Model with unused parameters sync grads automatically.
        model = ToyModel2()
        optimizer = SGD(model.parameters(), lr=0.1)
        optimizer_wrapper = OptimizerWrapper(model, optimizer)
        optimizer_wrapper.zero_grad()
        inputs = torch.randn(1, 1, 1, 1) * self.rank
        loss = model(inputs)
        optimizer_wrapper.backward(loss)
        grad = model.conv.weight.grad
        # `nn.Module` will not sync grad automatically. test `sync_grads`
        # method.
        optimizer_wrapper.sync_grads()
        all_grads = all_gather(grad)
        assert_allclose(all_grads[0], all_grads[1])

    def test_block_backward_sync(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ToyModel2()
        ddp_model = DistributedDataParallel(model)
        optimizer = SGD(ddp_model.parameters(), lr=0.1)
        optimizer_wrapper = OptimizerWrapper(ddp_model, optimizer)
        # Test ddp model will not sync grad under `block_backward_sync`.
        # with optimizer_wrapper.block_backward_sync():
        with self.assertRaises(AssertionError):
            inputs = torch.ones(1, 1, 1, 1) * self.rank
            ddp_model(inputs).sum().backward()
            with optimizer_wrapper.block_backward_sync():
                inputs = torch.ones(1, 1, 1, 1) * self.rank
                ddp_model(inputs).sum().backward()
                grad = model.conv.weight.grad
                all_grads = all_gather(grad)
                assert_allclose(all_grads[0], all_grads[1])

    def test_gradient_accumulative_context(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ToyModel2()
        ddp_model = DistributedDataParallel(model)
        optimizer = SGD(ddp_model.parameters(), lr=0.01)
        optimizer_wrapper = OptimizerWrapper(
            ddp_model, optimizer, cumulative_iters=1)
        optimizer_wrapper.zero_grad()
        with optimizer_wrapper.gradient_accumulative_context():
            # Automatically sync grads if `cumulative_iters` = 1
            inputs = torch.randn(1, 1, 1, 1) * self.rank
            ddp_model(inputs).sum().backward()
            grad = model.conv.weight.grad
            all_grads = all_gather(grad)
            assert_allclose(all_grads[0], all_grads[1])

        # Do not sync grads when `optimizer_wrapper.cur_iter` cannot be
        # divided by `optimizer_wrapper.cumulative_iters`
        optimizer_wrapper.cumulative_iters = 3
        with optimizer_wrapper.gradient_accumulative_context():
            ddp_model(inputs).sum().backward()
            all_grads = all_gather(model.conv.weight.grad)
            with self.assertRaises(AssertionError):
                assert_allclose(all_grads[0], all_grads[1])

        # sync grads if `cur_iter == 2`
        self.message_hub.update_info('iter', 2)
        with optimizer_wrapper.gradient_accumulative_context():
            ddp_model(inputs).sum().backward()
            all_grads = all_gather(model.conv.weight.grad)
            assert_allclose(all_grads[0], all_grads[1])

    def test_precision_context(self):
        optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)
        with optimizer_wrapper.precision_context():
            pass

    def _init_dist_env(self, rank, world_size):
        """Initialize the distributed environment."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29515'
        os.environ['RANK'] = str(rank)
        torch_dist.init_process_group(
            backend='gloo', rank=rank, world_size=world_size)


class TestAmpOptimizerWrapper(TestCase):

    def setUp(self) -> None:
        self.model = ToyModel()
        self.optimizer = SGD(self.model.parameters(), lr=0.1)

    def test_init(self):
        # Test with default arguments.
        amp_optim_wrapper = AmpOptimizerWrapper(
            model=self.model, optimizer=self.optimizer)
        self.assertEqual(amp_optim_wrapper._scale_update_param, 512.)
        self.assertEqual(amp_optim_wrapper.loss_scaler._init_scale, 512.)

        # Test with dynamic.
        amp_optim_wrapper = AmpOptimizerWrapper(
            'dynamic', model=self.model, optimizer=self.optimizer)
        self.assertIsNone(amp_optim_wrapper._scale_update_param)

        # Test with dict loss_scale.
        amp_optim_wrapper = AmpOptimizerWrapper(
            dict(init_scale=1, growth_factor=2), model=self.model,
            optimizer=self.optimizer)
        self.assertEqual(amp_optim_wrapper.loss_scaler._init_scale, 1)
        self.assertEqual(amp_optim_wrapper.loss_scaler._growth_factor, 2)
        self.assertIsNone(amp_optim_wrapper._scale_update_param)
        with self.assertRaises(TypeError):
            AmpOptimizerWrapper(
                model=self.model,
                optimizer=self.optimizer,
                loss_scale='unknown',
                detect_anomalous_params=False)

    def test_zero_grad(self):
        optimizer = MagicMock()
        amp_optim_wrapper = AmpOptimizerWrapper(
            model=self.model, optimizer=optimizer)
        amp_optim_wrapper.zero_grad()
        optimizer.zero_grad.assert_called()

    def test_step(self):
        optimizer = MagicMock()
        amp_optim_wrapper = AmpOptimizerWrapper(
            model=self.model, optimizer=optimizer)
        amp_optim_wrapper.loss_scaler = MagicMock()
        amp_optim_wrapper.step()
        amp_optim_wrapper.loss_scaler.step.assert_called_with(
            amp_optim_wrapper.optimizer)
        amp_optim_wrapper.loss_scaler.update.assert_called_with(
            amp_optim_wrapper._scale_update_param)

    def test_backward(self):
        for detect_anomalous_params in (True, False):
            amp_optim_wrapper = AmpOptimizerWrapper(
                model=self.model,
                optimizer=self.optimizer)
            amp_optim_wrapper.detect_anomalous_params = detect_anomalous_params
            loss_scaler = MagicMock()
            scale_return = MagicMock()
            scale_fn = MagicMock(return_value=scale_return)
            loss_scaler.scale = scale_fn
            amp_optim_wrapper.loss_scaler = loss_scaler
            amp_optim_wrapper._amp_optim_wrapper = MagicMock()
            amp_optim_wrapper._detect_anomalous_params = MagicMock()
            amp_optim_wrapper.backward(1)
            loss_scaler.scale.assert_called_with(1)
            scale_return.backward.assert_called_with()

    def test_state_dict(self):
        amp_optim_wrapper = AmpOptimizerWrapper(
            model=self.model,
            optimizer=self.optimizer)
        state_dict = amp_optim_wrapper.state_dict()
        self.assertEqual(MessageHub.get_current_instance().get_info(
            'loss_scalar'), amp_optim_wrapper.loss_scaler.state_dict())
        self.assertEqual(state_dict, self.optimizer.state_dict())

    def test_load_state_dict(self):
        amp_optim_wrapper = AmpOptimizerWrapper(
            model=self.model,
            optimizer=self.optimizer)
        amp_optim_wrapper.loss_scaler = MagicMock()
        optimizer = SGD(self.model.parameters(), lr=0.1)
        MessageHub.get_current_instance().update_info(
            'loss_scalar', dict(scale=1))
        amp_optim_wrapper.load_state_dict(optimizer.state_dict())

        amp_optim_wrapper.loss_scaler.load_state_dict.assert_called_with(
            dict(scale=1)
        )
        self.assertEqual(amp_optim_wrapper.optimizer.state_dict(),
                         optimizer.state_dict())

    @unittest.skipIf(
        not torch.cuda.is_available(), reason='at lest need 1 gpu to test')
    def test_precision_context_manager(self):
        with AmpOptimizerWrapper.precision_context():
            x = torch.randn(1, 1, 1, 1).cuda()
            y = nn.Conv2d(1, 1, 1).cuda()(x)
            self.assertEqual(y.dtype, torch.float16)



