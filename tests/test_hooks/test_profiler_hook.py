# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmengine import digit_version
from mmengine.hooks import ProfilerHook  # noqa
from mmengine.testing import RunnerTestCase


class TestProfilerHook(RunnerTestCase):

    @pytest.mark.skipif(
        digit_version(torch.__version__)[1] < 8,
        reason='torch required to 1.8')
    def test_setup(self):
        self.setUp()
        self.epoch_based_cfg['custom_hooks'] = [
            dict(type='ProfilerHook', priority='NORMAL')
        ]
        self._run()
        pass

    def test_print_log(self):
        self.setUp()
        self.epoch_based_cfg['custom_hooks'] = [
            dict(
                type='ProfilerHook',
                priority='NORMAL',
                on_trace_ready=dict(type='log_trace'),
            )
        ]
        self._run()
        pass

    def test_json(self):
        self.setUp()
        self.epoch_based_cfg['custom_hooks'] = [
            dict(
                type='ProfilerHook',
                priority='NORMAL',
                json_trace_path='/mnt/d/Experiment/mmengine/001.json')
        ]
        self._run()
        pass

    def test_tensorboard(self):
        self.setUp()
        self.epoch_based_cfg['custom_hooks'] = [
            dict(
                type='ProfilerHook',
                priority='NORMAL',
                on_trace_ready=dict(
                    type='tb_trace', dir_name='/mnt/d/Experiment/mmengine/tb'))
        ]
        self._run()
        pass

    def _run(self):
        runner = self.build_runner(self.epoch_based_cfg)  # noqa
        runner.train()
        runner.val()
        runner.test()

        runner = self.build_runner(self.iter_based_cfg)
        runner.train()
        runner.val()
        runner.test()
        pass
