# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

from mmengine.hooks import ParamSchedulerHook


class TestParamSchedulerHook:

    def test_after_iter(self):
        hook = ParamSchedulerHook()
        runner = Mock()
        scheduler = Mock()
        scheduler.step = Mock()
        scheduler.by_epoch = False
        runner.param_schedulers = [scheduler]
        hook.after_train_iter(runner, 0)
        scheduler.step.assert_called()

    def test_after_epoch(self):
        hook = ParamSchedulerHook()
        runner = Mock()
        scheduler = Mock()
        scheduler.step = Mock()
        scheduler.by_epoch = True
        runner.param_schedulers = [scheduler]
        hook.after_train_epoch(runner)
        scheduler.step.assert_called()
