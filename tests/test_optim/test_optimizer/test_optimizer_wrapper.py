import logging
from unittest import TestCase
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.optim import SGD

from mmengine.optim import OptimizerWrapper
from mmengine import MessageHub, MMLogger


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


class TestOptimizerWrapper(TestCase):
    def setUp(self) -> None:
        self.model = ToyModel()
        self.optimizer = SGD(self.model.parameters(), lr=0.1)
        self.message_hub = MessageHub.get_instance('test_optim_wrapper_init')
        self.logger = MMLogger.get_instance('test_optim_wrapper')
        self.message_hub.update_info('iter', 0)
        self.message_hub.update_info('max_iters', 100)

    def tearDown(self) -> None:
        self.message_hub.update_info('iter', 0)
        self.message_hub.update_info('max_iters', 100)

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
        optimizer_wrapper._init()
        optimizer_wrapper.initialized = False
        # Mock method
        optimizer_wrapper.backward = MagicMock()
        optimizer_wrapper.step = MagicMock()
        optimizer_wrapper.zero_grad = MagicMock()
        optimizer_wrapper._init = MagicMock()
        # `iter=0`, Call `optimizer_step` first time.
        loss = torch.tensor(1)
        optimizer_wrapper.optimizer_step(loss)
        optimizer_wrapper._init.assert_called()
        optimizer_wrapper.backward.assert_called_with(torch.tensor(1) / 3)
        optimizer_wrapper.step.assert_not_called()
        optimizer_wrapper.zero_grad.assert_not_called()
        optimizer_wrapper.initialized = True
        # reset MagicMock
        optimizer_wrapper._init = MagicMock()
        # `iter=2`, Call `optimizer_step` first time.
        self.message_hub.update_info('iter', 2)
        optimizer_wrapper.optimizer_step(loss)
        optimizer_wrapper._init.assert_not_called()
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

    def test__init(self):
        optimizer = MagicMock()
        optimizer_wrapper = OptimizerWrapper(
            self.model, optimizer, cumulative_iters=3)

        self.assertFalse(optimizer_wrapper.initialized)
        optimizer_wrapper._init()
        self.assertEqual(optimizer_wrapper.divisible_iters, 99)
        self.assertEqual(optimizer_wrapper.remainder_iters, 1)
        self.assertTrue(optimizer_wrapper.initialized)
        optimizer.zero_grad.assert_called()

        # Indivisible cur_iter will output warning.
        self.message_hub.update_info('iter', 1)
        self.message_hub.update_info('max_iters', 100)
        with self.assertLogs(self.logger) as cm:
            optimizer_wrapper._init()
            self.assertEqual(len(cm.output), 1)
            self.assertRegex(cm.records[0].msg, 'Resume iter number is not')

        # Model with batch norm will output warning.
        self.message_hub.update_info('iter', 0)
        optimizer_wrapper.model = nn.BatchNorm2d(1)
        with self.assertLogs(self.logger) as cm:
            optimizer_wrapper._init()
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
        self.assertEqual(
            optimizer_wrapper.state_dict(), self.optimizer.state_dict())

    def test_load_state_dict(self):
        optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)
        model = ToyModel()
        optimizer = SGD(model.parameters(), lr=0.1)
        optimizer_wrapper.load_state_dict(optimizer.state_dict())

        self.assertEqual(
            optimizer_wrapper.state_dict(), optimizer.state_dict())

    def test_param_groups(self):
        optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)
        self.assertEqual(
            optimizer_wrapper.param_groups, self.optimizer.param_groups)
