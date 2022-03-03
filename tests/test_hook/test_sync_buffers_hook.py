# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

from mmengine.hooks import SyncBuffersHook


class TestSyncBuffersHook:

    def test_sync_buffers_hook(self):
        Runner = Mock()
        Runner.model = Mock()
        Hook = SyncBuffersHook()
        Hook.after_epoch(Runner)
