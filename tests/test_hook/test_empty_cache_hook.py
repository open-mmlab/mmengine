# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

from mock import Mock

from mmengine.hooks import EmptyCacheHook


def mock(*args, **kwargs):
    pass


class TestEmptyCacheHook:

    @patch('torch.cuda.empty_cache', mock)
    def test_emtpy_cache_hook(self):
        Hook = EmptyCacheHook(True, True, True)
        Runner = Mock()
        Hook.after_iter(Runner)
        Hook.before_epoch(Runner)
        Hook.after_epoch(Runner)
