# Copyright (c) OpenMMLab. All rights reserved.

import unittest

import torch

from mmengine.hooks import ProfilerHook  # noqa
from mmengine.testing import RunnerTestCase
from mmengine.utils import digit_version, is_installed


@unittest.skipIf(
    not digit_version(torch.__version__)[1] >= 8,
    reason='torch required to 1.8')
class TestProfilerHook(RunnerTestCase):

    def test_default(self):
        self.epoch_based_cfg['custom_hooks'] = [
            dict(type='ProfilerHook', priority='NORMAL')
        ]
        runner = self.build_runner(self.epoch_based_cfg)  # noqa
        runner.train()
        pass

    def test_activity(self):
        self.epoch_based_cfg['custom_hooks'] = [
            dict(
                type='ProfilerHook',
                priority='NORMAL',
                activity_with_cpu=False)
        ]
        runner = self.build_runner(self.epoch_based_cfg)  # noqa
        runner.train()
        pass

    def test_iter(self):
        self.epoch_based_cfg['custom_hooks'] = [
            dict(type='ProfilerHook', priority='NORMAL', by_epoch=False)
        ]
        runner = self.build_runner(self.epoch_based_cfg)  # noqa
        runner.train()
        pass

    def test_multiple_epoch(self):
        with self.assertWarns(Warning):
            self.epoch_based_cfg['custom_hooks'] = [
                dict(
                    type='ProfilerHook',
                    priority='NORMAL',
                    by_epoch=True,
                    profile_times=2)
            ]
            runner = self.build_runner(self.epoch_based_cfg)  # noqa
            runner.train()
        pass

    def test_multiple_iter(self):
        with self.assertRaises(ValueError):
            self.epoch_based_cfg['custom_hooks'] = [
                dict(
                    type='ProfilerHook',
                    priority='NORMAL',
                    by_epoch=False,
                    profile_times=20000)
            ]
            runner = self.build_runner(self.epoch_based_cfg)  # noqa
            runner.train()
        pass

    def test_profile_times(self):
        with self.assertRaises(ValueError):
            self.epoch_based_cfg['custom_hooks'] = [
                dict(type='ProfilerHook', priority='NORMAL', profile_times=0)
            ]
            runner = self.build_runner(self.epoch_based_cfg)  # noqa
            runner.train()
        pass

    @unittest.skipIf(
        not torch.cuda.is_available(), reason='required tensorboard')
    def test_cuda(self):
        self.epoch_based_cfg['custom_hooks'] = [
            dict(
                type='ProfilerHook',
                priority='NORMAL',
                activity_with_cuda=True)
        ]
        runner = self.build_runner(self.epoch_based_cfg)  # noqa
        runner.train()
        pass

    def test_schedule(self):
        self.epoch_based_cfg['custom_hooks'] = [
            dict(
                type='ProfilerHook',
                priority='NORMAL',
                schedule=dict(
                    wait=1, warmup=1, active=3, repeat=1, skip_first=0))
        ]
        runner = self.build_runner(self.epoch_based_cfg)  # noqa
        runner.train()
        pass

    def test_table(self):
        # torch.autograd.profiler_util.EventList -> table
        self.epoch_based_cfg['custom_hooks'] = [
            dict(
                type='ProfilerHook',
                priority='NORMAL',
                on_trace_ready=dict(
                    type='log_trace',
                    sort_by='self_cpu_time_total',
                    row_limit=10))
        ]
        runner = self.build_runner(self.epoch_based_cfg)  # noqa
        runner.train()
        pass

    def test_json(self):
        self.epoch_based_cfg['custom_hooks'] = [
            dict(
                type='ProfilerHook',
                priority='NORMAL',
                json_trace_path=f'{self.temp_dir}/demo.json')
        ]
        runner = self.build_runner(self.epoch_based_cfg)  # noqa
        runner.train()
        pass

    @unittest.skipIf(
        not is_installed('tensorboard'), reason='required tensorboard')
    def test_tensorboard(self):
        self.epoch_based_cfg['custom_hooks'] = [
            dict(
                type='ProfilerHook',
                priority='NORMAL',
                on_trace_ready=dict(
                    type='tb_trace', dir_name=f'{self.temp_dir}/tb'),
                json_trace_path='will warning')
        ]
        runner = self.build_runner(self.epoch_based_cfg)  # noqa
        runner.train()

        self.epoch_based_cfg['custom_hooks'] = [
            dict(
                type='ProfilerHook',
                priority='NORMAL',
                on_trace_ready=dict(type='tb_trace'))
        ]
        runner = self.build_runner(self.epoch_based_cfg)  # noqa
        runner.train()

        self.epoch_based_cfg['custom_hooks'] = [
            dict(
                type='ProfilerHook',
                priority='NORMAL',
                on_trace_ready=dict(type='tb_trace', dir_name='/tmp/tmp_tb'))
        ]
        runner = self.build_runner(self.epoch_based_cfg)  # noqa
        runner.train()
        pass

    def test_on_trace_ready(self):
        with self.assertRaises(ValueError):
            self.epoch_based_cfg['custom_hooks'] = [
                dict(type='ProfilerHook', priority='NORMAL', on_trace_ready=0)
            ]
            runner = self.build_runner(self.epoch_based_cfg)  # noqa
            runner.train()

        with self.assertRaises(ValueError):
            self.epoch_based_cfg['custom_hooks'] = [
                dict(
                    type='ProfilerHook',
                    priority='NORMAL',
                    on_trace_ready=dict(type='unknown'))
            ]
            runner = self.build_runner(self.epoch_based_cfg)  # noqa
            runner.train()
        pass
