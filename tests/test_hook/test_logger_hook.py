# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest.mock import MagicMock

import pytest

from mmengine.fileio.file_client import HardDiskBackend
from mmengine.hooks import LoggerHook


class TestLoggerHook:

    def test_init(self):
        logger_hook = LoggerHook(out_dir='tmp.txt')
        assert logger_hook.interval == 10
        assert logger_hook.ignore_last
        assert logger_hook.interval_exp_name == 1000
        assert logger_hook.out_suffix == ('.log.json', '.log', '.py')
        assert logger_hook.keep_local
        assert logger_hook.file_client_args is None
        assert isinstance(logger_hook.file_client.client, HardDiskBackend)
        # out_dir should be None or string or tuple of string.
        with pytest.raises(TypeError):
            LoggerHook(out_dir=1)

        with pytest.raises(ValueError):
            LoggerHook(file_client_args=dict(enable_mc=True))

    def test_before_run(self):
        runner = MagicMock()
        runner.iter = 10
        runner.timestamp = 'timestamp'
        runner.work_dir = 'work_dir'
        runner.logger = MagicMock()
        logger_hook = LoggerHook(out_dir='out_dir')
        logger_hook.before_run(runner)
        assert logger_hook.out_dir == osp.join('out_dir', 'work_dir')
        assert logger_hook.json_log_path == osp.join('work_dir',
                                                     'timestamp.log.json')
        runner.writer.add_params.assert_called()

    def test_after_run(self, tmp_path):
        # Test
        out_dir = tmp_path / 'out_dir'
        out_dir.mkdir()
        work_dir = tmp_path / 'work_dir'
        work_dir.mkdir()
        work_dir_json = work_dir / 'tmp.log.json'
        runner = MagicMock()
        runner.work_dir = work_dir
        # Test without out_dir.
        logger_hook = LoggerHook()
        logger_hook.after_run(runner)
        # Test with out_dir and make sure json file has been moved to out_dir.
        json_f = open(work_dir_json, 'w')
        json_f.close()
        logger_hook = LoggerHook(out_dir=str(tmp_path), keep_local=False)
        logger_hook.out_dir = str(out_dir)
        logger_hook.after_run(runner)
        # Verify that the file has been moved to `out_dir`.
        assert not osp.exists(str(work_dir_json))
        assert osp.exists(str(out_dir / 'tmp.log.json'))

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
        runner.meta = dict(exp_name='retinanet')
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
        runner.writer.add_scalars.assert_called()

    def test_after_test_epoch(self):
        logger_hook = LoggerHook()
        runner = MagicMock()
        runner.log_processor.get_log_after_epoch = MagicMock(
            return_value=(dict(), 'log_str'))
        logger_hook.after_test_epoch(runner)
        runner.log_processor.get_log_after_epoch.assert_called()
        runner.logger.info.assert_called()

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
