# Copyright (c) OpenMMLab. All rights reserved.

import os
import os.path as ops
import unittest
from unittest.mock import MagicMock

import torch

import mmengine.hooks
from mmengine.device import is_npu_available
from mmengine.hooks import NPUProfilerHook, ProfilerHook
from mmengine.logging import MMLogger
from mmengine.testing import RunnerTestCase
from mmengine.utils import is_installed


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

    @unittest.skipIf(
        not is_installed('torch-tb-profiler'),
        reason='required torch-tb-profiler')
    def test_parse_trace_config_tensorboard(self):
        # Test on_trace_ready_args
        runner = MagicMock()
        runner.log_dir = self.temp_dir.name
        runner.logger = MMLogger.get_instance('test_profiler')
        hook = ProfilerHook(on_trace_ready=None)

        hook.on_trace_ready = dict(type='tb_trace')
        hook._parse_trace_config(runner)

        hook.on_trace_ready['dir_name'] = 'tb'
        hook._parse_trace_config(runner)

        hook.on_trace_ready['dir_name'] = ops.join(self.temp_dir.name, 'tb')
        hook._parse_trace_config(runner)

        # with self.assertWarns(DeprecationWarning):
        hook = ProfilerHook(
            on_trace_ready=dict(type='tb_trace'),
            json_trace_path=ops.join(self.temp_dir.name, 'demo.json'))
        hook._parse_trace_config(runner)

        self.epoch_based_cfg['custom_hooks'] = [
            dict(
                type='ProfilerHook',
                on_trace_ready=dict(
                    type='tb_trace', dir_name=self.temp_dir.name))
        ]
        runner = self.build_runner(self.epoch_based_cfg)
        runner.train()

    def test_before_run(self):
        runner = MagicMock()
        runner.max_epochs = 1000
        runner.max_iters = 10000
        runner.logger = MMLogger.get_instance('test_profiler')

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
        runner.logger = MMLogger.get_instance('test_profiler')

        hook = ProfilerHook(
            json_trace_path=ops.join(self.temp_dir.name, 'demo.json'))
        hook.before_run(runner)
        hook._export_chrome_trace(runner)

    def test_after_train_epoch(self):
        runner = MagicMock()
        runner.max_epochs = 1000
        runner.logger = MMLogger.get_instance('test_profiler')

        runner.epoch = 0

        hook = ProfilerHook()
        hook.before_run(runner)
        hook.profiler.__exit__(None, None, None)

        hook.profiler = MagicMock()
        hook.after_train_epoch(runner)
        hook.profiler.__exit__.assert_called_once()

    def test_after_train_iter(self):
        runner = MagicMock()
        runner.max_iters = 10000
        runner.logger = MMLogger.get_instance('test_profiler')

        runner.iter = 9

        hook = ProfilerHook(by_epoch=False, profile_times=10, schedule=None)
        hook.profiler = MagicMock()
        hook.after_train_iter(runner, 1, 1, 1)
        hook.profiler.__exit__.assert_called_once()
        hook.profiler.step.assert_called_once()

        hook = ProfilerHook(
            by_epoch=False,
            schedule=dict(wait=1, warmup=1, active=3, repeat=1))
        hook.profiler = MagicMock()
        hook.after_train_iter(runner, 1, 1, 1)
        hook.profiler.step.assert_called_once()

    def test_with_runner(self):
        self.epoch_based_cfg['custom_hooks'] = [
            dict(
                type='ProfilerHook',
                activity_with_cpu=False,
                activity_with_cuda=False)
        ]
        runner = self.build_runner(self.epoch_based_cfg)
        runner.train()

        json_path = ops.join(self.temp_dir.name, 'demo.json')
        self.epoch_based_cfg['custom_hooks'] = [
            dict(type='ProfilerHook', json_trace_path=json_path)
        ]
        runner = self.build_runner(self.epoch_based_cfg)
        runner.train()
        self.assertTrue(
            ops.exists(json_path), 'ERROR::json file is not generated!')

        self.epoch_based_cfg['custom_hooks'] = [
            dict(
                type='ProfilerHook',
                on_trace_ready=dict(
                    type='log_trace',
                    sort_by='self_cpu_time_total',
                    row_limit=10))
        ]
        runner = self.build_runner(self.epoch_based_cfg)
        runner.train()

        with self.assertRaises(ValueError):
            self.epoch_based_cfg['custom_hooks'] = [
                dict(type='ProfilerHook', on_trace_ready=0)
            ]
            runner = self.build_runner(self.epoch_based_cfg)
            runner.train()

        if torch.cuda.is_available():
            self.epoch_based_cfg['custom_hooks'] = [
                dict(type='ProfilerHook', activity_with_cuda=True)
            ]
            runner = self.build_runner(self.epoch_based_cfg)
            runner.train()


@unittest.skipIf(
    not is_npu_available(), reason='Ascend PyTorch and npu devices not exist')
class TestNPUProfilerHook(RunnerTestCase):

    def test_init(self):

        result_path = ops.join(self.temp_dir.name, 'test/cann_profiling')

        NPUProfilerHook(result_path=result_path)

        with self.assertRaises(ValueError):
            NPUProfilerHook(begin=1, end=0, result_path=result_path)

    def test_before_run(self):
        result_path = ops.join(self.temp_dir.name, 'test/cann_profiling')
        runner = MagicMock()
        runner.max_iters = 1
        runner.logger = MMLogger.get_instance('test_npu_profiler')

        hook = NPUProfilerHook(result_path=result_path)
        hook.before_run(runner)

        with self.assertRaises(ValueError):
            hook = NPUProfilerHook(begin=0, end=10, result_path=result_path)
            hook.before_run(runner)

    def test_after_train_iter(self):
        result_path = ops.join(self.temp_dir.name, 'test/cann_profiling')
        runner = MagicMock()
        runner.max_iters = 10000
        runner.logger = MMLogger.get_instance('test_npu_profiler')

        runner.iter = 0

        hook = NPUProfilerHook(begin=0, end=10, result_path=result_path)
        hook.before_run(runner)

        hook.profiler = MagicMock()
        hook.after_train_iter(runner, 1)

    def test_with_runner(self):
        result_path = ops.join(self.temp_dir.name, 'test/cann_profiling')
        self.epoch_based_cfg['custom_hooks'] = [
            dict(
                type='NPUProfilerHook',
                begin=0,
                result_path=result_path,
                exit_after_profiling=False)
        ]
        runner = self.build_runner(self.epoch_based_cfg)
        runner.train()

        self.epoch_based_cfg['custom_hooks'] = [
            dict(
                type='NPUProfilerHook',
                result_path=result_path,
                ge_profiling_to_std_out=True,
                exit_after_profiling=False)
        ]
        runner = self.build_runner(self.epoch_based_cfg)
        runner.train()

        self.assertTrue(
            ops.exists(result_path), 'profiler result path is not generated!')

        self.assertTrue(
            os.getenv('GE_PROFILING_TO_STD_OUT', '0') == '1',
            'GE PROFILING failed to start!')
