# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest.mock import ANY, MagicMock

import pytest
import torch

from mmengine.fileio import load
from mmengine.fileio.file_client import HardDiskBackend
from mmengine.hooks import LoggerHook


class TestLoggerHook:

    def test_init(self):
        logger_hook = LoggerHook(out_dir='tmp.txt')
        assert logger_hook.interval == 10
        assert logger_hook.ignore_last
        assert logger_hook.interval_exp_name == 1000
        assert logger_hook.out_suffix == ('.json', '.log', '.py', 'yaml')
        assert logger_hook.keep_local
        assert logger_hook.file_client_args is None
        assert isinstance(logger_hook.file_client.client, HardDiskBackend)
        # out_dir should be None or string or tuple of string.
        with pytest.raises(TypeError):
            LoggerHook(out_dir=1)

        with pytest.raises(ValueError):
            LoggerHook(file_client_args=dict(enable_mc=True))

        # test `file_client_args` and `backend_args`
        # TODO Refine this unit test
        # with pytest.warns(
        #         DeprecationWarning,
        #         match='"file_client_args" will be deprecated in future'):
        #     logger_hook = LoggerHook(
        #         out_dir='tmp.txt', file_client_args={'backend': 'disk'})

        with pytest.raises(
                ValueError,
                match='"file_client_args" and "backend_args" cannot be '
                'set at the same time'):
            logger_hook = LoggerHook(
                out_dir='tmp.txt',
                file_client_args={'backend': 'disk'},
                backend_args={'backend': 'local'})

    def test_before_run(self):
        runner = MagicMock()
        runner.iter = 10
        runner.timestamp = '20220429'
        runner._log_dir = f'work_dir/{runner.timestamp}'
        runner.work_dir = 'work_dir'
        runner.logger = MagicMock()
        logger_hook = LoggerHook(out_dir='out_dir')
        logger_hook.before_run(runner)
        assert logger_hook.out_dir == osp.join('out_dir', 'work_dir')
        assert logger_hook.json_log_path == f'{runner.timestamp}.json'

    def test_after_run(self, tmp_path):
        # Test
        timestamp = '20220429'
        out_dir = tmp_path / 'out_dir'
        out_dir.mkdir()
        work_dir = tmp_path / 'work_dir'
        work_dir.mkdir()
        log_dir = work_dir / timestamp
        log_dir.mkdir()
        log_dir_json = log_dir / 'tmp.log.json'
        runner = MagicMock()
        runner._log_dir = str(log_dir)
        runner.timestamp = timestamp
        runner.work_dir = str(work_dir)
        # Test without out_dir.
        logger_hook = LoggerHook()
        logger_hook.after_run(runner)
        # Test with out_dir and make sure json file has been moved to out_dir.
        json_f = open(log_dir_json, 'w')
        json_f.close()
        logger_hook = LoggerHook(out_dir=str(out_dir), keep_local=False)
        logger_hook.out_dir = str(out_dir)
        logger_hook.before_run(runner)
        logger_hook.after_run(runner)
        # Verify that the file has been moved to `out_dir`.
        assert not osp.exists(str(log_dir_json))
        assert osp.exists(str(out_dir / 'work_dir' / 'tmp.log.json'))

    def test_after_train_iter(self):
        # Test LoggerHook by iter.
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

    def test_after_val_epoch(self):
        logger_hook = LoggerHook()
        runner = MagicMock()
        runner.log_processor.get_log_after_epoch = MagicMock(
            return_value=(dict(), 'string'))
        logger_hook.after_val_epoch(runner)
        runner.log_processor.get_log_after_epoch.assert_called()
        runner.logger.info.assert_called()
        runner.visualizer.add_scalars.assert_called()

        # Test when `log_metric_by_epoch` is True
        runner.log_processor.get_log_after_epoch = MagicMock(
            return_value=({
                'time': 1,
                'datatime': 1,
                'acc': 0.8
            }, 'string'))
        logger_hook.after_val_epoch(runner)
        args = {'step': ANY, 'file_path': ANY}
        # expect visualizer log `time` and `metric` respectively
        runner.visualizer.add_scalars.assert_called_with({'acc': 0.8}, **args)

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
        runner.visualizer.add_scalars.assert_called_with(
            {
                'time': 5,
                'datatime': 5,
                'acc': 0.5
            }, **args)

        with pytest.raises(AssertionError):
            runner.visualizer.add_scalars.assert_any_call(
                {
                    'time': 5,
                    'datatime': 5
                }, **args)
        with pytest.raises(AssertionError):
            runner.visualizer.add_scalars.assert_any_call({'acc': 0.5}, **args)

    def test_after_test_epoch(self, tmp_path):
        logger_hook = LoggerHook()
        runner = MagicMock()
        runner.log_dir = tmp_path
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
