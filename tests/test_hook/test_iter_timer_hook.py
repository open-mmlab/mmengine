# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

from mmengine.hooks import IterTimerHook


class TestIterTimerHook:

    def test_before_epoch(self):
        Hook = IterTimerHook()
        Runner = Mock()
        Hook.before_epoch(Runner)

    def test_before_iter(self):
        Hook = IterTimerHook()
        Runner = Mock()
        Hook.before_epoch(Runner)
        Hook.before_iter(Runner)

    def test_after_iter(self):
        Hook = IterTimerHook()
        Runner = Mock()
        Hook.before_epoch(Runner)
        Hook.after_iter(Runner)
