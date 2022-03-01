# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mock import Mock

from mmengine.hooks import EmptyCacheHook


class TestEmptyCacheHook:

    @pytest.mark.skipif(condition=not torch.cuda.is_available(), reason='requires CUDA support')
    def test_emtpy_cache_hook(self):
        Hook = EmptyCacheHook(True, True, True)
        Runner = Mock()
        Hook.after_iter(Runner)
        Hook.before_epoch(Runner)
        Hook.after_epoch(Runner)
