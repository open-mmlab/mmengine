# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

from mmengine.hooks import ParamSchedulerHook


class TestParamSchedulerHook:

    def test_after_iter(self):
        Hook = ParamSchedulerHook()
        Runner = Mock()
        scheduler = Mock()
        scheduler.step = Mock()
        scheduler.by_epoch = False
        Runner.schedulers = [scheduler]
        Hook.after_iter(Runner)
        scheduler.step.assert_called()

    def test_after_epoch(self):
        Hook = ParamSchedulerHook()
        Runner = Mock()
        scheduler = Mock()
        scheduler.step = Mock()
        scheduler.by_epoch = True
        Runner.schedulers = [scheduler]
        Hook.after_epoch(Runner)
        scheduler.step.assert_called()
