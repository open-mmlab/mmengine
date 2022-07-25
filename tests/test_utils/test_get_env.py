# Copyright (c) OpenMMLab. All rights reserved.
import sys
from unittest import TestCase

import pytest

import mmengine
from mmengine.utils.collect_env import _get_cuda_home, collect_env


class TestGetEnv(TestCase):

    def test_get_cuda_home(self):
        CUDA_HOME = _get_cuda_home()
        assert CUDA_HOME

    def test_collect_env(self):
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            pytest.skip('skipping tests that require PyTorch')

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
