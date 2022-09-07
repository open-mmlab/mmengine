# Copyright (c) OpenMMLab. All rights reserved.
import sys
from unittest import TestCase

import torch.cuda

import mmengine
from mmengine.utils.dl_utils import collect_env
from mmengine.utils.dl_utils.parrots_wrapper import _get_cuda_home


class TestCollectEnv(TestCase):

    def test_get_cuda_home(self):
        CUDA_HOME = _get_cuda_home()
        if torch.version.cuda is not None:
            self.assertIsNotNone(CUDA_HOME)
        else:
            self.assertIsNone(CUDA_HOME)

    def test_collect_env(self):
        env_info = collect_env()
        expected_keys = [
            'sys.platform', 'Python', 'CUDA available', 'PyTorch',
            'PyTorch compiling details', 'OpenCV', 'MMEngine', 'GCC'
        ]
        for key in expected_keys:
            assert key in env_info

        if env_info['CUDA available']:
            for key in ['CUDA_HOME', 'NVCC']:
                assert key in env_info

        if sys.platform == 'win32':
            assert 'MSVC' in env_info

        assert env_info['sys.platform'] == sys.platform
        assert env_info['Python'] == sys.version.replace('\n', '')
        assert env_info['MMEngine'] == mmengine.__version__
