# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

from mmengine.hooks import IterTimerHook


class TestIterTimerHook:

    def test_before_epoch(self):
        hook = IterTimerHook()
        runner = Mock()
        hook._before_epoch(runner)
        assert isinstance(hook.t, float)

    def test_before_iter(self):
        hook = IterTimerHook()
        runner = Mock()
        runner.log_buffer = dict()
        hook._before_epoch(runner)
        hook._before_iter(runner, 0)
        runner.message_hub.update_scalar.assert_called()

    def test_after_iter(self):
        hook = IterTimerHook()
        runner = Mock()
        runner.log_buffer = dict()
        hook._before_epoch(runner)
        hook._after_iter(runner, 0)
        runner.message_hub.update_scalar.assert_called()
