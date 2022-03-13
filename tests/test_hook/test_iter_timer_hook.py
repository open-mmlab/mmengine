# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

from mmengine.hooks import IterTimerHook


class TestIterTimerHook:

    def test_before_epoch(self):
        Hook = IterTimerHook()
        Runner = Mock()
        Hook._before_epoch(Runner)
        assert isinstance(Hook.t, float)

    def test_before_iter(self):
        Hook = IterTimerHook()
        Runner = Mock()
        Runner.log_buffer = dict()
        Hook._before_epoch(Runner)
        Hook._before_iter(Runner)
        Runner.message_hub.update_log.assert_called()

    def test_after_iter(self):
        Hook = IterTimerHook()
        Runner = Mock()
        Runner.log_buffer = dict()
        Hook._before_epoch(Runner)
        Hook._after_iter(Runner)
        Runner.message_hub.update_log.assert_called()
