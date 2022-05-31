# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

from mmengine.hooks import RuntimeInfoHook
from mmengine.logging import MessageHub


class TestRuntimeInfoHook(TestCase):

    def test_before_run(self):
        message_hub = MessageHub.get_instance(
            'runtime_info_hook_test_before_run')
        runner = Mock()
        runner.epoch = 3
        runner.iter = 30
        runner.max_epochs = 4
        runner.max_iters = 40
        runner.message_hub = message_hub
        hook = RuntimeInfoHook()
        hook.before_run(runner)
        self.assertEqual(message_hub.get_info('epoch'), 3)
        self.assertEqual(message_hub.get_info('iter'), 30)
        self.assertEqual(message_hub.get_info('max_epochs'), 4)
        self.assertEqual(message_hub.get_info('max_iters'), 40)

    def test_before_train(self):
        message_hub = MessageHub.get_instance(
            'runtime_info_hook_test_before_train')
        runner = Mock()
        runner.epoch = 7
        runner.iter = 71
        runner.message_hub = message_hub
        hook = RuntimeInfoHook()
        hook.before_train(runner)
        self.assertEqual(message_hub.get_info('epoch'), 7)
        self.assertEqual(message_hub.get_info('iter'), 71)

    def test_before_train_epoch(self):
        message_hub = MessageHub.get_instance(
            'runtime_info_hook_test_before_train_epoch')
        runner = Mock()
        runner.epoch = 9
        runner.message_hub = message_hub
        hook = RuntimeInfoHook()
        hook.before_train_epoch(runner)
        self.assertEqual(message_hub.get_info('epoch'), 9)

    def test_before_train_iter(self):
        # single optimizer
        message_hub = MessageHub.get_instance(
            'runtime_info_hook_test_before_train_iter')
        runner = Mock()
        runner.iter = 9
        runner.optimizer.param_groups = [{'lr': 0.01}]
        runner.message_hub = message_hub
        hook = RuntimeInfoHook()
        hook.before_train_iter(runner, batch_idx=2, data_batch=None)
        self.assertEqual(message_hub.get_info('iter'), 9)
        self.assertEqual(message_hub.get_scalar('train/lr').current(), 0.01)

        # multiple optimizers
        message_hub = MessageHub.get_instance(
            'runtime_info_hook_test_before_train_iter')
        runner = Mock()
        runner.iter = 9
        optimizer1 = Mock()
        optimizer1.param_groups = [{'lr': 0.01}]
        optimizer2 = Mock()
        optimizer2.param_groups = [{'lr': 0.02}]
        runner.optimizer = dict(key1=optimizer1, key2=optimizer2)
        runner.message_hub = message_hub
        hook = RuntimeInfoHook()
        hook.before_train_iter(runner, batch_idx=2, data_batch=None)
        self.assertEqual(message_hub.get_info('iter'), 9)
        self.assertEqual(
            message_hub.get_scalar('train/key1.lr').current(), 0.01)
        self.assertEqual(
            message_hub.get_scalar('train/key2.lr').current(), 0.02)

    def test_after_train_iter(self):
        message_hub = MessageHub.get_instance(
            'runtime_info_hook_test_after_train_iter')
        runner = Mock()
        runner.message_hub = message_hub
        hook = RuntimeInfoHook()
        hook.after_train_iter(
            runner,
            batch_idx=2,
            data_batch=None,
            outputs={'log_vars': {
                'loss_cls': 1.111
            }})
        self.assertEqual(
            message_hub.get_scalar('train/loss_cls').current(), 1.111)

    def test_after_val_epoch(self):
        message_hub = MessageHub.get_instance(
            'runtime_info_hook_test_after_val_epoch')
        runner = Mock()
        runner.message_hub = message_hub
        hook = RuntimeInfoHook()
        hook.after_val_epoch(runner, metrics={'acc': 0.8})
        self.assertEqual(message_hub.get_scalar('val/acc').current(), 0.8)

    def test_after_test_epoch(self):
        message_hub = MessageHub.get_instance(
            'runtime_info_hook_test_after_test_epoch')
        runner = Mock()
        runner.message_hub = message_hub
        hook = RuntimeInfoHook()
        hook.after_test_epoch(runner, metrics={'acc': 0.8})
        self.assertEqual(message_hub.get_scalar('test/acc').current(), 0.8)
