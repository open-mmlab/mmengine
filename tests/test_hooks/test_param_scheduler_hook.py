# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest.mock import Mock

from mmengine.hooks import ParamSchedulerHook
from mmengine.optim import _ParamScheduler
from mmengine.testing import RunnerTestCase


class TestParamSchedulerHook(RunnerTestCase):
    error_msg = ('runner.param_schedulers should be list of ParamScheduler or '
                 'a dict containing list of ParamScheduler')

    def test_after_train_iter(self):
        # runner.param_schedulers should be a list or dict
        with self.assertRaisesRegex(TypeError, self.error_msg):
            hook = ParamSchedulerHook()
            runner = Mock()
            scheduler = Mock()
            scheduler.step = Mock()
            scheduler.by_epoch = False
            runner.param_schedulers = scheduler
            hook.after_train_iter(runner, 0)
            scheduler.step.assert_called()

        # runner.param_schedulers is a list of schedulers
        hook = ParamSchedulerHook()
        runner = Mock()
        scheduler = Mock()
        scheduler.step = Mock()
        scheduler.by_epoch = False
        runner.param_schedulers = [scheduler]
        hook.after_train_iter(runner, 0)
        scheduler.step.assert_called()

        # runner.param_schedulers is a dict containing list of schedulers
        scheduler1 = Mock()
        scheduler1.step = Mock()
        scheduler1.by_epoch = False
        scheduler2 = Mock()
        scheduler2.step = Mock()
        scheduler2.by_epoch = False
        runner.param_schedulers = dict(key1=[scheduler1], key2=[scheduler2])
        hook.after_train_epoch(runner)
        hook.after_train_iter(runner, 0)
        scheduler1.step.assert_called()
        scheduler2.step.assert_called()

    def test_after_train_epoch(self):
        # runner.param_schedulers should be a list or dict
        with self.assertRaisesRegex(TypeError, self.error_msg):
            hook = ParamSchedulerHook()
            runner = Mock()
            scheduler = Mock()
            scheduler.step = Mock()
            scheduler.by_epoch = True
            runner.param_schedulers = scheduler
            hook.after_train_epoch(runner)
            scheduler.step.assert_called()

        # runner.param_schedulers is a list of schedulers
        hook = ParamSchedulerHook()
        runner = Mock()
        scheduler = Mock()
        scheduler.step = Mock()
        scheduler.by_epoch = True
        runner.param_schedulers = [scheduler]
        hook.after_train_epoch(runner)
        scheduler.step.assert_called()

        # runner.param_schedulers is a dict containing list of schedulers
        scheduler1 = Mock()
        scheduler1.step = Mock()
        scheduler1.by_epoch = True
        scheduler2 = Mock()
        scheduler2.step = Mock()
        scheduler2.by_epoch = True
        runner.param_schedulers = dict(key1=[scheduler1], key2=[scheduler2])
        hook.after_train_epoch(runner)
        scheduler1.step.assert_called()
        scheduler2.step.assert_called()

    def test_after_val_epoch(self):
        metrics = dict(loss=1.0)

        # mock super _ParamScheduler class
        class MockParamScheduler(_ParamScheduler):

            def __init__(self):
                pass

            def _get_value(self):
                pass

        # runner.param_schedulers should be a list or dict
        with self.assertRaisesRegex(TypeError, self.error_msg):
            hook = ParamSchedulerHook()
            runner = Mock()
            scheduler = Mock()
            scheduler.step = Mock()
            scheduler.by_epoch = True
            scheduler.need_val_args = True
            runner.param_schedulers = scheduler
            hook.after_val_epoch(runner, metrics)

        # runner.param_schedulers is a list of schedulers
        hook = ParamSchedulerHook()
        runner = Mock()
        scheduler = MockParamScheduler()
        scheduler.step = Mock()
        scheduler.by_epoch = True
        scheduler.need_val_args = True
        runner.param_schedulers = [scheduler]
        hook.after_val_epoch(runner, metrics)
        scheduler.step.assert_called_with(metrics)

        # runner.param_schedulers is a dict containing list of schedulers
        scheduler1 = MockParamScheduler()
        scheduler1.step = Mock()
        scheduler1.by_epoch = True
        scheduler1.need_val_args = True
        scheduler2 = MockParamScheduler()
        scheduler2.step = Mock()
        scheduler2.by_epoch = True
        scheduler2.need_val_args = True
        runner.param_schedulers = dict(key1=[scheduler1], key2=[scheduler2])
        hook.after_val_epoch(runner, metrics)
        scheduler1.step.assert_called_with(metrics)
        scheduler2.step.assert_called_with(metrics)

    def test_with_runner(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.train_cfg.max_epochs = 3
        cfg.param_scheduler = [
            dict(
                type='ConstantLR',
                factor=0.5,
                begin=0,
            ),
            dict(
                type='ConstantLR',
                factor=0.5,
                begin=1,
            )
        ]
        init_lr = cfg.optim_wrapper.optimizer.lr
        runner = self.build_runner(cfg)
        runner.train()

        # Length of train log is 4
        # Learning rate of the first epoch is init_lr*0.5
        # Learning rate of the second epoch is init_lr*0.5*0.5
        # Learning rate of the last epoch will be reset to 0.1
        train_lr = list(runner.message_hub.get_scalar('train/lr')._log_history)
        target_lr = [init_lr * 0.5] * 4 + \
                    [init_lr * 0.5 * 0.5] * 4 + \
                    [init_lr] * 4
        self.assertListEqual(train_lr, target_lr)

        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.param_scheduler = [
            dict(
                type='ConstantLR',
                factor=0.5,
                begin=0,
                by_epoch=False,
            ),
            dict(
                type='ConstantLR',
                factor=0.5,
                begin=4,
                by_epoch=False,
            )
        ]

        init_lr = cfg.optim_wrapper.optimizer.lr
        runner = self.build_runner(cfg)
        runner.train()

        # Learning rate of 1-4 iteration is init_lr*0.5
        # Learning rate of 5-11 iteration is init_lr*0.5*0.5
        train_lr = list(runner.message_hub.get_scalar('train/lr')._log_history)
        target_lr = [init_lr * 0.5] * 4 + \
                    [init_lr * 0.5 * 0.5] * 7 + \
                    [init_lr]
        self.assertListEqual(train_lr, target_lr)
