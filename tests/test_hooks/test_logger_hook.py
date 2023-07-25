# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import shutil
from unittest.mock import ANY, MagicMock, call

import torch

from mmengine.fileio import load
from mmengine.hooks import LoggerHook
from mmengine.logging import MMLogger
from mmengine.testing import RunnerTestCase
from mmengine.utils import mkdir_or_exist, scandir


class TestLoggerHook(RunnerTestCase):

    def test_init(self):
        # Test build logger hook.
        LoggerHook()
        LoggerHook(interval=100, ignore_last=False, interval_exp_name=100)

        with self.assertRaisesRegex(TypeError, 'interval must be'):
            LoggerHook(interval='100')

        with self.assertRaisesRegex(ValueError, 'interval must be'):
            LoggerHook(interval=-1)

        with self.assertRaisesRegex(TypeError, 'ignore_last must be'):
            LoggerHook(ignore_last='False')

        with self.assertRaisesRegex(TypeError, 'interval_exp_name'):
            LoggerHook(interval_exp_name='100')

        with self.assertRaisesRegex(ValueError, 'interval_exp_name'):
            LoggerHook(interval_exp_name=-1)

        with self.assertRaisesRegex(TypeError, 'out_suffix'):
            LoggerHook(out_suffix=[100])

        # out_dir should be None or string or tuple of string.
        with self.assertRaisesRegex(TypeError, 'out_dir must be'):
            LoggerHook(out_dir=1)

        with self.assertRaisesRegex(ValueError, 'file_client_args'):
            LoggerHook(file_client_args=dict(enable_mc=True))

        # test deprecated warning raised by `file_client_args`
        logger = MMLogger.get_current_instance()
        with self.assertLogs(logger, level='WARNING'):
            LoggerHook(
                out_dir=self.temp_dir.name,
                file_client_args=dict(backend='disk'))

        with self.assertRaisesRegex(
                ValueError,
                '"file_client_args" and "backend_args" cannot be '):
            LoggerHook(
                out_dir=self.temp_dir.name,
                file_client_args=dict(enable_mc=True),
                backend_args=dict(enable_mc=True))

    def test_after_train_iter(self):
        # Test LoggerHook by iter.
        # Avoid to compare `Runner.iter` (MagicMock) with other integers.
        ori_every_n_train_iters = LoggerHook.every_n_train_iters
        LoggerHook.every_n_train_iters = MagicMock(return_value=True)
        runner = MagicMock()
        runner.log_processor.get_log_after_iter = MagicMock(
            return_value=(dict(), 'log_str'))
        logger_hook = LoggerHook()
        logger_hook.after_train_iter(runner, batch_idx=5)
        # `cur_iter=10+1`, which cannot be exact division by
        # `logger_hook.interval`
        runner.log_processor.get_log_after_iter.assert_not_called()
        logger_hook.after_train_iter(runner, batch_idx=9)
        runner.log_processor.get_log_after_iter.assert_called()

        # Test LoggerHook by epoch.
        logger_hook = LoggerHook()
        runner = MagicMock()
        runner.log_processor.get_log_after_iter = MagicMock(
            return_value=(dict(), 'log_str'))
        # Only `batch_idx` will work.
        logger_hook.after_train_iter(runner, batch_idx=10)
        runner.log_processor.get_log_after_iter.assert_not_called()
        logger_hook.after_train_iter(runner, batch_idx=9)
        runner.log_processor.get_log_after_iter.assert_called()

        # Test end of the epoch.
        runner = MagicMock()
        runner.log_processor.get_log_after_iter = MagicMock(
            return_value=(dict(), 'log_str'))
        logger_hook = LoggerHook(ignore_last=False)
        runner.train_dataloader = [0] * 5
        logger_hook.after_train_iter(runner, batch_idx=4)
        runner.log_processor.get_log_after_iter.assert_called()

        # Test print exp_name
        runner = MagicMock()
        runner.log_processor.get_log_after_iter = MagicMock(
            return_value=(dict(), 'log_str'))
        runner.logger = MagicMock()
        logger_hook = LoggerHook()
        logger_hook.after_train_iter(runner, batch_idx=999)
        runner.logger.info.assert_called()

        # Test print training log when the num of
        # iterations is smaller than the default interval
        runner = MagicMock()
        runner.log_processor.get_log_after_iter = MagicMock(
            return_value=(dict(), 'log_str'))
        runner.train_dataloader = [0] * 9
        logger_hook = LoggerHook()
        logger_hook.after_train_iter(runner, batch_idx=8)
        runner.log_processor.get_log_after_iter.assert_called()
        LoggerHook.every_n_train_iters = ori_every_n_train_iters

    def test_after_val_epoch(self):
        logger_hook = LoggerHook()
        runner = MagicMock()
        # Test when `log_metric_by_epoch` is True
        runner.log_processor.get_log_after_epoch = MagicMock(
            return_value=({
                'time': 1,
                'datatime': 1,
                'acc': 0.8
            }, 'string'))
        logger_hook.after_val_epoch(runner)

        # expect visualizer log `time` and `metric` respectively
        args = {'step': ANY, 'file_path': ANY}
        calls = [
            call({
                'time': 1,
                'datatime': 1,
                'acc': 0.8
            }, **args),
        ]
        self.assertEqual(
            len(calls), len(runner.visualizer.add_scalars.mock_calls))
        runner.visualizer.add_scalars.assert_has_calls(calls)

        # Test when `log_metric_by_epoch` is False
        logger_hook = LoggerHook(log_metric_by_epoch=False)
        runner.log_processor.get_log_after_epoch = MagicMock(
            return_value=({
                'time': 5,
                'datatime': 5,
                'acc': 0.5
            }, 'string'))
        logger_hook.after_val_epoch(runner)

        # expect visualizer log `time` and `metric` jointly
        calls = [
            call({
                'time': 1,
                'datatime': 1,
                'acc': 0.8
            }, **args),
            call({
                'time': 5,
                'datatime': 5,
                'acc': 0.5
            }, **args),
        ]
        self.assertEqual(
            len(calls), len(runner.visualizer.add_scalars.mock_calls))
        runner.visualizer.add_scalars.assert_has_calls(calls)

    def test_after_test_epoch(self):
        logger_hook = LoggerHook()
        runner = MagicMock()
        runner.log_dir = self.temp_dir.name
        runner.timestamp = 'test_after_test_epoch'
        runner.log_processor.get_log_after_epoch = MagicMock(
            return_value=(
                dict(a=1, b=2, c={'list': [1, 2]}, d=torch.tensor([1, 2, 3])),
                'log_str'))
        logger_hook.before_run(runner)
        logger_hook.after_test_epoch(runner)
        runner.log_processor.get_log_after_epoch.assert_called()
        runner.logger.info.assert_called()
        osp.isfile(osp.join(runner.log_dir, 'test_after_test_epoch.json'))
        json_content = load(
            osp.join(runner.log_dir, 'test_after_test_epoch.json'))
        assert json_content == dict(a=1, b=2, c={'list': [1, 2]}, d=[1, 2, 3])

    def test_after_val_iter(self):
        logger_hook = LoggerHook()
        runner = MagicMock()
        runner.iter = 0
        runner.log_processor.get_log_after_iter = MagicMock(
            return_value=(dict(), 'log_str'))
        logger_hook.after_val_iter(runner, 1)
        runner.log_processor.get_log_after_iter.assert_not_called()
        logger_hook.after_val_iter(runner, 9)
        runner.log_processor.get_log_after_iter.assert_called()

    def test_after_test_iter(self):
        logger_hook = LoggerHook()
        runner = MagicMock()
        runner.iter = 0
        runner.log_processor.get_log_after_iter = MagicMock(
            return_value=(dict(), 'log_str'))
        logger_hook.after_test_iter(runner, 1)
        runner.log_processor.get_log_after_iter.assert_not_called()
        logger_hook.after_test_iter(runner, 9)
        runner.log_processor.get_log_after_iter.assert_called()

    def test_with_runner(self):
        # Test dumped the json exits
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.default_hooks.logger = dict(type='LoggerHook')
        cfg.train_cfg.max_epochs = 10
        runner = self.build_runner(cfg)
        runner.train()
        json_path = osp.join(runner._log_dir, 'vis_data',
                             f'{runner.timestamp}.json')
        self.assertTrue(osp.isfile(json_path))

        # Test out_dir
        out_dir = osp.join(cfg.work_dir, 'test')
        mkdir_or_exist(out_dir)
        cfg.default_hooks.logger = dict(type='LoggerHook', out_dir=out_dir)
        runner = self.build_runner(cfg)
        runner.train()
        self.assertTrue(os.listdir(out_dir))
        # clean the out_dir
        for filename in os.listdir(out_dir):
            shutil.rmtree(osp.join(out_dir, filename))

        # Test out_suffix
        cfg.default_hooks.logger = dict(
            type='LoggerHook', out_dir=out_dir, out_suffix='.log')
        runner = self.build_runner(cfg)
        runner.train()
        filenames = scandir(out_dir, recursive=True)
        self.assertTrue(
            all(filename.endswith('.log') for filename in filenames))

        # Test keep_local=False
        cfg.default_hooks.logger = dict(
            type='LoggerHook', out_dir=out_dir, keep_local=False)
        runner = self.build_runner(cfg)
        runner.train()
        filenames = scandir(runner._log_dir, recursive=True)

        for filename in filenames:
            self.assertFalse(
                filename.endswith(('.log', '.json', '.py', '.yaml')),
                f'{filename} should not be kept.')
