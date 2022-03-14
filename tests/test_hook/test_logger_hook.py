# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import logging
import os.path as osp
import sys
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import torch

from mmengine.fileio.file_client import HardDiskBackend
from mmengine.hooks import LoggerHook


class TestLoggerHook:

    def test_init(self):
        logger_hook = LoggerHook(out_dir='tmp.txt')
        assert logger_hook.by_epoch
        assert logger_hook.interval == 10
        assert not logger_hook.custom_keys
        assert logger_hook.ignore_last
        assert logger_hook.time_sec_tot == 0
        assert logger_hook.interval_exp_name == 1000
        assert logger_hook.out_suffix == ('.log.json', '.log', '.py')
        assert logger_hook.keep_local
        assert logger_hook.file_client_args is None
        assert isinstance(logger_hook.file_client.client, HardDiskBackend)
        # out_dir should be None or string or tuple of string.
        with pytest.raises(TypeError):
            LoggerHook(out_dir=1)
        # time cannot be overwritten.
        with pytest.raises(AssertionError):
            LoggerHook(custom_keys=dict(time=dict(method='max')))
        LoggerHook(
            custom_keys=dict(time=[
                dict(method='max', log_name='time_max'),
                dict(method='min', log_name='time_min')
            ]))
        # Epoch window_size cannot be used when `LoggerHook.by_epoch=False`
        with pytest.raises(AssertionError):
            LoggerHook(
                by_epoch=False,
                custom_keys=dict(
                    time=dict(
                        method='max', log_name='time_max',
                        window_size='epoch')))
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
        assert logger_hook.start_iter == runner.iter
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
        runner.iter = 10
        logger_hook = LoggerHook(by_epoch=False)
        logger_hook._log_train = MagicMock()
        logger_hook.after_train_iter(runner)
        # `cur_iter=10+1`, which cannot be exact division by
        # `logger_hook.interval`
        logger_hook._log_train.assert_not_called()
        runner.iter = 9
        logger_hook.after_train_iter(runner)
        logger_hook._log_train.assert_called()

        # Test LoggerHook by epoch.
        logger_hook = LoggerHook(by_epoch=True)
        logger_hook._log_train = MagicMock()
        # Only `runner.inner_iter` will work.
        runner.iter = 9
        runner.inner_iter = 10
        logger_hook.after_train_iter(runner)
        logger_hook._log_train.assert_not_called()
        runner.inner_iter = 9
        logger_hook.after_train_iter(runner)
        logger_hook._log_train.assert_called()

        # Test end of the epoch.
        logger_hook = LoggerHook(by_epoch=True, ignore_last=False)
        logger_hook._log_train = MagicMock()
        runner.cur_dataloader = [0] * 5
        runner.inner_iter = 4
        logger_hook.after_train_iter(runner)
        logger_hook._log_train.assert_called()

        # Test print exp_name
        runner.meta = dict(exp_name='retinanet')
        logger_hook = LoggerHook()
        runner.logger = MagicMock()
        logger_hook._log_train = MagicMock()
        logger_hook.after_train_iter(runner)
        runner.logger.info.assert_called_with(
            f'Exp name: {runner.meta["exp_name"]}')

    def test_after_val_epoch(self):
        logger_hook = LoggerHook()
        runner = MagicMock()
        logger_hook._log_val = MagicMock()
        logger_hook.after_val_epoch(runner)
        logger_hook._log_val.assert_called()

    @pytest.mark.parametrize('by_epoch', [True, False])
    def test_log_train(self, by_epoch, capsys):
        runner = self._setup_runner()
        runner.meta = dict(exp_name='retinanet')
        # Prepare LoggerHook
        logger_hook = LoggerHook(by_epoch=by_epoch)
        logger_hook.writer = MagicMock()
        logger_hook.time_sec_tot = 1000
        logger_hook.start_iter = 0
        logger_hook._get_max_memory = MagicMock(return_value='100')
        logger_hook.json_log_path = 'tmp.json'

        # Prepare training information.
        train_infos = dict(
            lr=0.1, momentum=0.9, time=1.0, data_time=1.0, loss_cls=1.0)
        logger_hook._collect_info = MagicMock(return_value=train_infos)
        logger_hook._log_train(runner)
        # Verify that the correct variables have been written.
        runner.writer.add_scalars.assert_called_with(
            train_infos, step=11, file_path='tmp.json')
        # Verify that the correct context have been logged.
        out, _ = capsys.readouterr()
        time_avg = logger_hook.time_sec_tot / (
            runner.iter + 1 - logger_hook.start_iter)
        eta_second = time_avg * (runner.train_loop.max_iters - runner.iter - 1)
        eta_str = str(datetime.timedelta(seconds=int(eta_second)))
        if by_epoch:
            if torch.cuda.is_available():
                log_str = 'Epoch [2][2/5]\t' \
                          f"lr: {train_infos['lr']:.3e} " \
                          f"momentum: {train_infos['momentum']:.3e}, " \
                          f'eta: {eta_str}, ' \
                          f"time: {train_infos['time']:.3f}, " \
                          f"data_time: {train_infos['data_time']:.3f}, " \
                          f'memory: 100, ' \
                          f"loss_cls: {train_infos['loss_cls']:.4f}\n"
            else:
                log_str = 'Epoch [2][2/5]\t' \
                          f"lr: {train_infos['lr']:.3e} " \
                          f"momentum: {train_infos['momentum']:.3e}, " \
                          f'eta: {eta_str}, ' \
                          f"time: {train_infos['time']:.3f}, " \
                          f"data_time: {train_infos['data_time']:.3f}, " \
                          f"loss_cls: {train_infos['loss_cls']:.4f}\n"
            assert out == log_str
        else:
            if torch.cuda.is_available():
                log_str = 'Iter [11/50]\t' \
                          f"lr: {train_infos['lr']:.3e} " \
                          f"momentum: {train_infos['momentum']:.3e}, " \
                          f'eta: {eta_str}, ' \
                          f"time: {train_infos['time']:.3f}, " \
                          f"data_time: {train_infos['data_time']:.3f}, " \
                          f'memory: 100, ' \
                          f"loss_cls: {train_infos['loss_cls']:.4f}\n"
            else:
                log_str = 'Iter [11/50]\t' \
                          f"lr: {train_infos['lr']:.3e} " \
                          f"momentum: {train_infos['momentum']:.3e}, " \
                          f'eta: {eta_str}, ' \
                          f"time: {train_infos['time']:.3f}, " \
                          f"data_time: {train_infos['data_time']:.3f}, " \
                          f"loss_cls: {train_infos['loss_cls']:.4f}\n"
            assert out == log_str

    @pytest.mark.parametrize('by_epoch', [True, False])
    def test_log_val(self, by_epoch, capsys):
        runner = self._setup_runner()
        # Prepare LoggerHook.
        logger_hook = LoggerHook(by_epoch=by_epoch)
        logger_hook.json_log_path = 'tmp.json'
        metric = dict(accuracy=0.9, data_time=1.0)
        logger_hook._collect_info = MagicMock(return_value=metric)
        logger_hook._log_val(runner)
        # Verify that the correct context have been logged.
        out, _ = capsys.readouterr()
        runner.writer.add_scalars.assert_called_with(
            metric, step=11, file_path='tmp.json')
        if by_epoch:
            assert out == 'Epoch(val) [1][5]\taccuracy: 0.9000, ' \
                          'data_time: 1.0000\n'

        else:
            assert out == 'Iter(val) [5]\taccuracy: 0.9000, ' \
                          'data_time: 1.0000\n'

    def test_get_window_size(self):
        runner = self._setup_runner()
        logger_hook = LoggerHook()
        # Test get window size by name.
        assert logger_hook._get_window_size(runner, 'epoch') == 2
        assert logger_hook._get_window_size(runner, 'global') == 11
        assert logger_hook._get_window_size(runner, 10) == 10
        # Window size must equal to `logger_hook.interval`.
        with pytest.raises(AssertionError):
            logger_hook._get_window_size(runner, 20)

        with pytest.raises(ValueError):
            logger_hook._get_window_size(runner, 'unknwon')

    def test_parse_custom_keys(self):
        tag = OrderedDict()
        runner = self._setup_runner()
        log_buffers = OrderedDict(lr=MagicMock(), loss=MagicMock())
        cfg_dict = dict(
            lr=dict(method='min'),
            loss=[
                dict(method='min', window_size='global'),
                dict(method='max', log_name='loss_max')
            ])
        logger_hook = LoggerHook()
        for log_key, log_cfg in cfg_dict.items():
            logger_hook._parse_custom_keys(runner, log_key, log_cfg,
                                           log_buffers, tag)
        assert list(tag) == ['lr', 'loss', 'loss_max']
        assert log_buffers['lr'].min.assert_called
        assert log_buffers['loss'].min.assert_called
        assert log_buffers['loss'].max.assert_called
        assert log_buffers['loss'].mean.assert_called
        # `log_name` Cannot be repeated.
        with pytest.raises(KeyError):
            cfg_dict = dict(loss=[
                dict(method='min', window_size='global'),
                dict(method='max', log_name='loss_max'),
                dict(method='mean', log_name='loss_max')
            ])
            logger_hook.custom_keys = cfg_dict
            for log_key, log_cfg in cfg_dict.items():
                logger_hook._parse_custom_keys(runner, log_key, log_cfg,
                                               log_buffers, tag)
        # `log_key` cannot be overwritten multiple times.
        with pytest.raises(AssertionError):
            cfg_dict = dict(loss=[
                dict(method='min', window_size='global'),
                dict(method='max'),
            ])
            logger_hook.custom_keys = cfg_dict
            for log_key, log_cfg in cfg_dict.items():
                logger_hook._parse_custom_keys(runner, log_key, log_cfg,
                                               log_buffers, tag)

    def test_collect_info(self):
        runner = self._setup_runner()
        logger_hook = LoggerHook(
            custom_keys=dict(time=dict(method='max', log_name='time_max')))
        logger_hook._parse_custom_keys = MagicMock()
        # Collect with prefix.
        log_buffers = {
            'train/time': MagicMock(),
            'lr': MagicMock(),
            'train/loss_cls': MagicMock(),
            'val/metric': MagicMock()
        }
        runner.message_hub.log_buffers = log_buffers
        tag = logger_hook._collect_info(runner, mode='train')
        # Test parse custom_keys
        logger_hook._parse_custom_keys.assert_called()
        # Test training key in tag.
        assert list(tag.keys()) == ['time', 'loss_cls']
        # Test statistics lr with `current`, loss and time with 'mean'
        log_buffers['train/time'].mean.assert_called()
        log_buffers['train/loss_cls'].mean.assert_called()
        log_buffers['train/loss_cls'].current.assert_not_called()

        tag = logger_hook._collect_info(runner, mode='val')
        assert list(tag.keys()) == ['metric']
        log_buffers['val/metric'].current.assert_called()

    @patch('torch.cuda.max_memory_allocated', MagicMock())
    @patch('torch.cuda.reset_peak_memory_stats', MagicMock())
    def test_get_max_memory(self):
        logger_hook = LoggerHook()
        runner = MagicMock()
        runner.world_size = 1
        runner.model = torch.nn.Linear(1, 1)
        logger_hook._get_max_memory(runner)
        torch.cuda.max_memory_allocated.assert_called()
        torch.cuda.reset_peak_memory_stats.assert_called()

    def test_get_iter(self):
        runner = self._setup_runner()
        logger_hook = LoggerHook()
        # Get global iter when `inner_iter=False`
        iter = logger_hook._get_iter(runner)
        assert iter == 11
        # Get inner iter
        iter = logger_hook._get_iter(runner, inner_iter=True)
        assert iter == 2
        # Still get global iter when `logger_hook.by_epoch==False`
        logger_hook.by_epoch = False
        iter = logger_hook._get_iter(runner, inner_iter=True)
        assert iter == 11

    def test_get_epoch(self):
        runner = self._setup_runner()
        logger_hook = LoggerHook()
        epoch = logger_hook._get_epoch(runner, 'train')
        assert epoch == 2
        epoch = logger_hook._get_epoch(runner, 'val')
        assert epoch == 1
        with pytest.raises(ValueError):
            logger_hook._get_epoch(runner, 'test')

    def _setup_runner(self):
        runner = MagicMock()
        runner.epoch = 1
        runner.cur_dataloader = [0] * 5
        runner.inner_iter = 1
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
