# Copyright (c) OpenMMLab. All rights reserved.
from mock import Mock
from mmengine.hooks import SyncBuffersHook


class TestSyncBuffersHook:

    def test_sync_buffers_hook(self):
        Runner = Mock()
        Runner.model = Mock()
        Hook = SyncBuffersHook(distributed=True)
        Hook.after_epoch(Runner)
        Hook = SyncBuffersHook(distributed=False)
        Hook.after_epoch(Runner)
