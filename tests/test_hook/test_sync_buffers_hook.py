# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

from mmengine.hooks import SyncBuffersHook


class TestSyncBuffersHook:

    def test_sync_buffers_hook(self):
        runner = Mock()
        runner.model = Mock()
        hook = SyncBuffersHook()
        hook._after_epoch(runner)
