# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

from mmengine.hooks import ParamSchedulerHook
from mmengine.optim import OptimizerWrapper


class TestParamSchedulerHook:

    def test_after_iter(self):
        hook = ParamSchedulerHook()
        runner = MagicMock()
        runner.optimizer_wrapper = OptimizerWrapper(MagicMock(), MagicMock())
        scheduler = MagicMock()
        scheduler.step = MagicMock()
        scheduler.by_epoch = False
        runner.param_schedulers = [scheduler]
        hook.after_train_iter(runner, 0)
        scheduler.step.assert_called()
        runner.message_hub.update_scalar.assert_called()
        # Test with dict optimizer wrapper
        runner.optimizer_wrapper = dict(a=MagicMock())
        hook.after_train_iter(runner, 0)
        runner.message_hub.update_scalar.assert_called()

    def test_after_epoch(self):
        hook = ParamSchedulerHook()
        runner = MagicMock()
        runner.optimizer_wrapper = OptimizerWrapper(MagicMock(), MagicMock())
        scheduler = MagicMock()
        scheduler.step = MagicMock()
        scheduler.by_epoch = True
        runner.param_schedulers = [scheduler]
        hook.after_train_epoch(runner)
        scheduler.step.assert_called()
        runner.message_hub.update_scalar.assert_called()
        # Test with dict optimizer wrapper
        runner.optimizer_wrapper = dict(a=MagicMock())
        hook.after_train_epoch(runner)
        runner.message_hub.update_scalar.assert_called()
