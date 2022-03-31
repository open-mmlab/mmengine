# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

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
        out_dir = tmp_path / 'out_dir'
        out_dir.mkdir()
        work_dir = tmp_path / 'work_dir'
        work_dir.mkdir()
        work_dir_json = work_dir / 'tmp.log.json'
        json_f = open(work_dir_json, 'w')
        json_f.close()
        runner = MagicMock()
        runner.work_dir = work_dir

        logger_hook = LoggerHook(out_dir=str(tmp_path), keep_local=False)
        logger_hook.out_dir = str(out_dir)
        logger_hook.after_run(runner)
        # Verify that the file has been moved to `out_dir`.
        assert not osp.exists(str(work_dir_json))
        assert osp.exists(str(out_dir / 'tmp.log.json'))

    def test_after_train_iter(self):
        # Test LoggerHook by iter.
        runner = MagicMock()
        runner.log_processor.get_log = MagicMock(
            return_value=(dict(), 'log_str'))
        runner.iter = 10
        batch_idx = 5
        logger_hook = LoggerHook(by_epoch=False)
        logger_hook.after_train_iter(runner, batch_idx=batch_idx)
        # `cur_iter=10+1`, which cannot be exact division by
        # `logger_hook.interval`
        runner.log_processor.get_log.assert_not_called()
        runner.iter = 9
        logger_hook.after_train_iter(runner, batch_idx=batch_idx)
        runner.log_processor.get_log.assert_called()

        # Test LoggerHook by epoch.
        logger_hook = LoggerHook(by_epoch=True)
        runner.log_processor.get_log = MagicMock(
            return_value=(dict(), 'log_str'))
        # Only `batch_idx` will work.
        runner.iter = 9
        batch_idx = 10
        logger_hook.after_train_iter(runner, batch_idx=batch_idx)
        runner.log_processor.get_log.assert_not_called()
        batch_idx = 9
        logger_hook.after_train_iter(runner, batch_idx=batch_idx)
        runner.log_processor.get_log.assert_called()

        # Test end of the epoch.
        runner.log_processor.get_log = MagicMock(
            return_value=(dict(), 'log_str'))
        logger_hook = LoggerHook(ignore_last=False)
        runner.cur_dataloader = [0] * 5
        batch_idx = 4
        logger_hook.after_train_iter(runner, batch_idx=batch_idx)
        runner.log_processor.get_log.assert_called()

        # Test print exp_name
        runner.meta = dict(exp_name='retinanet')
        runner.logger = MagicMock()
        logger_hook = LoggerHook()
        logger_hook.after_train_iter(runner, batch_idx=batch_idx)
        runner.logger.info.assert_called()

    def test_after_val_epoch(self):
        logger_hook = LoggerHook()
        runner = MagicMock()
        runner.log_processor.get_log = MagicMock(
            return_value=(dict(), 'log_str'))
        logger_hook.after_val_epoch(runner)
        runner.log_processor.get_log.assert_called()

    def _setup_runner(self):
        runner = MagicMock()
        runner.epoch = 1
        runner.cur_dataloader = [0] * 5
        runner.iter = 10
        runner.train_loop.max_iters = 50
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for handler in logger.handlers:
            if not isinstance(handler, logging.StreamHandler):
                continue
        else:
            logger.addHandler(logging.StreamHandler(stream=sys.stdout))
        runner.logger = logger
        runner.message_hub = MagicMock()
        runner.composed_wirter = MagicMock()
        return runner
