# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

from mmengine.hooks import EmptyCacheHook


class TestEmptyCacheHook:

    def test_emtpy_cache_hook(self):
        Hook = EmptyCacheHook(True, True, True)
        Runner = Mock()
        Hook.after_iter(Runner)
        Hook.before_epoch(Runner)
        Hook.after_epoch(Runner)
