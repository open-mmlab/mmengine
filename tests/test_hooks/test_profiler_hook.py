# Copyright (c) OpenMMLab. All rights reserved.

import unittest

import torch

import mmengine.hooks
from mmengine.testing import RunnerTestCase
from mmengine.utils import is_installed


@unittest.skipIf(
    not mmengine.hooks.profiler_hook.check_kineto(),
    reason='Due to Kineto support issues, '
    'please upgrade pytorch above 1.8.1 (windows users above 1.9.1)')
class TestProfilerHook(RunnerTestCase):

    def test_config_cpu(self):
        configs: dict = {
            'default':
            dict(type='ProfilerHook'),
            'activity':
            dict(type='ProfilerHook', activity_with_cpu=False),
            'iter':
            dict(type='ProfilerHook', by_epoch=False),
            'schedule':
            dict(
                type='ProfilerHook',
                schedule=dict(wait=1, warmup=1, active=3, repeat=1)),
            'print_table':
            dict(
                type='ProfilerHook',
                on_trace_ready=dict(
                    type='log_trace',
                    sort_by='self_cpu_time_total',
                    row_limit=10)),
            'save_json':
            dict(
                type='ProfilerHook',
                priority='NORMAL',
                json_trace_path=f'{self.temp_dir}/demo.json')
        }
        for name_config, custom_hooks in configs.items():
            self.epoch_based_cfg['custom_hooks'] = [custom_hooks]
            runner = self.build_runner(self.epoch_based_cfg)
            runner.train()
            del runner

    def test_profile_times(self):
        configs: dict = {
            'multiple_epoch':
            dict(type='ProfilerHook', by_epoch=True, profile_times=2),
            'much_iter':
            dict(type='ProfilerHook', by_epoch=False, profile_times=20000),
            'zero':
            dict(type='ProfilerHook', profile_times=0)
        }
        for name_config, custom_hooks in configs.items():
            with self.assertRaises(ValueError):
                self.epoch_based_cfg['custom_hooks'] = [custom_hooks]
                runner = self.build_runner(self.epoch_based_cfg)
                runner.train()
                del runner

    def test_on_trace_ready(self):
        configs: dict = {
            'type':
            dict(type='ProfilerHook', on_trace_ready=0),
            'type_dict':
            dict(
                type='ProfilerHook',
                priority='NORMAL',
                on_trace_ready=dict(type='unknown')),
        }
        for name_config, custom_hooks in configs.items():
            with self.assertRaises(ValueError):
                self.epoch_based_cfg['custom_hooks'] = [custom_hooks]
            runner = self.build_runner(self.epoch_based_cfg)
            runner.train()
            del runner

    @unittest.skipIf(not torch.cuda.is_available(), reason='required cuda')
    def test_cuda(self):
        self.epoch_based_cfg['custom_hooks'] = [
            dict(
                type='ProfilerHook',
                priority='NORMAL',
                activity_with_cuda=True)
        ]
        runner = self.build_runner(self.epoch_based_cfg)  # noqa
        runner.train()

    @unittest.skipIf(
        not is_installed('tensorboard'), reason='required tensorboard')
    def test_tensorboard(self):
        configs: dict = {
            'multiple_epoch':
            dict(
                type='ProfilerHook',
                on_trace_ready=dict(
                    type='tb_trace', dir_name=f'{self.temp_dir}/tb'),
                json_trace_path='will warning'),
            'much_iter':
            dict(type='ProfilerHook', on_trace_ready=dict(type='tb_trace')),
            'zero':
            dict(
                type='ProfilerHook',
                on_trace_ready=dict(type='tb_trace', dir_name='/tmp/tmp_tb'))
        }
        for name_config, custom_hooks in configs.items():
            self.epoch_based_cfg['custom_hooks'] = [custom_hooks]
            runner = self.build_runner(self.epoch_based_cfg)
            runner.train()
            del runner
