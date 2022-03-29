# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

from mmengine.hooks import EmptyCacheHook


class TestEmptyCacheHook:

    def test_emtpy_cache_hook(self):
        hook = EmptyCacheHook(True, True, True)
        runner = Mock()
        hook._after_iter(runner, 0)
        hook._before_epoch(runner)
        hook._after_epoch(runner)
