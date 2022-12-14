# Copyright (c) OpenMMLab. All rights reserved.

import logging
import unittest
from unittest.mock import MagicMock

import mmengine.hooks
from mmengine.hooks import ProfilerHook
from mmengine.testing import RunnerTestCase

# import torch

# from mmengine.utils import is_installed


@unittest.skipIf(
    not mmengine.hooks.profiler_hook.check_kineto(),
    reason='Due to Kineto support issues, '
    'please upgrade pytorch above 1.8.1 (windows users above 1.9.1)')
class TestProfilerHook(RunnerTestCase):

    def test_init(self):
        # Test profile_times_args
        ProfilerHook(by_epoch=False, profile_times=1)
        with self.assertRaises(ValueError):
            ProfilerHook(profile_times=0)
        with self.assertRaises(ValueError):
            ProfilerHook(by_epoch=True, profile_times=2)

        # Test schedule_args
        ProfilerHook(schedule=dict(wait=1, warmup=1, active=3, repeat=1))
        with self.assertRaises(TypeError):
            ProfilerHook(schedule=dict())

    def test_parse_trace_config(self):
        # Test on_trace_ready_args
        runner = MagicMock()
        hook = ProfilerHook(on_trace_ready=None)

        hook.on_trace_ready = None
        hook._parse_trace_config(runner)

        def deal_profile(_profile):
            pass

        hook.on_trace_ready = deal_profile
        hook._parse_trace_config(runner)

        with self.assertRaises(ValueError):
            hook.on_trace_ready = dict(type='unknown')
            hook._parse_trace_config(runner)

        hook.on_trace_ready = dict(
            type='log_trace', sort_by='self_cpu_time_total', row_limit=10)
        hook._parse_trace_config(runner)

    # @unittest.skipIf(
    #     not is_installed('torch-tb-profiler'),
    #     reason='required torch-tb-profiler')
    # def test_parse_trace_config_tensorboard(self):
    #     # Test on_trace_ready_args
    #     runner = MagicMock()
    #     runner.work_dir = '/tmp/tb'
    #     runner.logger = logging
    #     hook = ProfilerHook(on_trace_ready=None)
    #
    #     hook.on_trace_ready = dict(type='tb_trace')
    #     hook._parse_trace_config(runner)
    #
    #     hook.on_trace_ready['dir_name'] = 'tb'
    #     hook._parse_trace_config(runner)
    #
    #     hook.on_trace_ready['dir_name'] = '/tmp/tb'
    #     hook._parse_trace_config(runner)
    #
    #     # with self.assertWarns(DeprecationWarning):
    #     hook = ProfilerHook(
    #         on_trace_ready=dict(type='tb_trace'),
    #         json_trace_path=f'{self.temp_dir}/demo.json')
    #     hook._parse_trace_config(runner)

    def test_before_run(self):
        runner = MagicMock()
        runner.max_epochs = 1000
        runner.max_iters = 10000
        runner.logger = logging

        hook = ProfilerHook()
        hook.before_run(runner)
        hook.profiler.__exit__(None, None, None)

        with self.assertRaises(ValueError):
            hook = ProfilerHook(by_epoch=False, profile_times=10001)
            hook.before_run(runner)
            hook.profiler.__exit__(None, None, None)

        with self.assertRaises(ValueError):
            hook = ProfilerHook(by_epoch=True, profile_times=1001)
            hook.before_run(runner)
            hook.profiler.__exit__(None, None, None)

    def test_export_chrome_trace(self):
        runner = MagicMock()
        runner.max_epochs = 1000
        runner.logger = logging

        hook = ProfilerHook(json_trace_path=f'{self.temp_dir}/demo.json')
        hook.before_run(runner)
        hook._export_chrome_trace(runner)

    def test_after_train_epoch(self):
        runner = MagicMock()
        runner.max_epochs = 1000
        runner.logger = logging

        runner.epoch = 0

        hook = ProfilerHook()
        hook.before_run(runner)

        hook.profiler = MagicMock()
        hook.after_train_epoch(runner)
        hook.profiler.__exit__.assert_called_once()

    def test_after_train_iter(self):
        runner = MagicMock()
        runner.max_iters = 10000
        runner.logger = logging

        runner.iter = 9

        hook = ProfilerHook(by_epoch=False, profile_times=10, schedule=None)
        hook.before_run(runner)

        hook.profiler = MagicMock()
        hook.after_train_iter(runner, 1, 1, 1)
        hook.profiler.__exit__.assert_called_once()
        hook.profiler.step.assert_called_once()

        hook = ProfilerHook(
            by_epoch=False,
            schedule=dict(wait=1, warmup=1, active=3, repeat=1))
        hook.before_run(runner)

        hook.profiler = MagicMock()
        hook.after_train_iter(runner, 1, 1, 1)
        hook.profiler.step.assert_not_called()

    def test_with_runner(self):
        configs: dict = {
            'activity':
            dict(
                type='ProfilerHook',
                activity_with_cpu=False,
                activity_with_cuda=False),
            'save_json':
            dict(
                type='ProfilerHook',
                json_trace_path=f'{self.temp_dir}/demo.json')
        }
        for name_config, custom_hooks in configs.items():
            self.epoch_based_cfg['custom_hooks'] = [custom_hooks]
            runner = self.build_runner(self.epoch_based_cfg)
            runner.train()
            del runner

        with self.assertRaises(ValueError):
            self.epoch_based_cfg['custom_hooks'] = [
                dict(type='ProfilerHook', on_trace_ready=0)
            ]
            runner = self.build_runner(self.epoch_based_cfg)
            runner.train()

    # @unittest.skipIf(not torch.cuda.is_available(), reason='required cuda')
    # def test_with_runner_cuda(self):
    #     self.epoch_based_cfg['custom_hooks'] = [
    #         dict(type='ProfilerHook', activity_with_cuda=True)
    #     ]
    #     runner = self.build_runner(self.epoch_based_cfg)  # noqa
    #     runner.train()