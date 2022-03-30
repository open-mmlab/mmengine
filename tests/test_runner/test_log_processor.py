# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest.mock import MagicMock, patch
from collections import OrderedDict
import pytest

import torch

from mmengine.runner import LogProcessor
from mmengine.logging import MMLogger
from mmengine.logging import MessageHub


class TestLogProcessor:

    def test_init(self):
        log_processor = LogProcessor(window_size=10, by_epoch=True,
                                     custom_cfg=None)
        assert log_processor.by_epoch
        assert log_processor.window_size == 10
        assert log_processor.custom_cfg == OrderedDict()

    def test_check_custom_cfg(self):
        # ``by_epoch==False`` and `window_size='epoch'` in log config will
        # raise AssertionError.
        custom_cfg = [dict(data_src='loss', window_size='epoch')]
        with pytest.raises(AssertionError):
            LogProcessor(by_epoch=False, custom_cfg=custom_cfg)
        # Duplicate log_name will raise AssertionError.
        custom_cfg = [dict(data_src='loss', log_name='loss_1'),
                      dict(data_src='loss', log_name='loss_1')]
        with pytest.raises(AssertionError):
            LogProcessor(custom_cfg=custom_cfg)
        # Overwrite loss item twice will raise AssertionError.
        custom_cfg = [dict(data_src='loss'),
                      dict(data_src='loss')]
        with pytest.raises(AssertionError):
            LogProcessor(custom_cfg=custom_cfg)

        custom_cfg = [dict(data_src='loss_cls',
                           window_size=100,
                           method_name='min'),
                      dict(data_src='loss',
                           log_name='loss_min',
                           method_name='max'),
                      dict(data_src='loss',
                           log_name='loss_max',
                           method_name='max')
                      ]
        LogProcessor(custom_cfg=custom_cfg)

    def test_parse_windows_size(self):
        runner = self._setup_runner()
        log_processor = LogProcessor()
        # Test parse 'epoch' window_size.
        custom_cfg = [dict(data_src='loss_cls', window_size='epoch')]
        log_processor._parse_windows_size(custom_cfg, runner, 1)
        assert custom_cfg[0]['window_size'] == 2

        # Test parse 'global' window_size.
        custom_cfg = [dict(data_src='loss_cls', window_size='global')]
        log_processor._parse_windows_size(custom_cfg, runner, 1)
        assert custom_cfg[0]['window_size'] == 11

        # Test parse int window_size
        custom_cfg = [dict(data_src='loss_cls', window_size=100)]
        log_processor._parse_windows_size(custom_cfg, runner, 1)
        assert custom_cfg[0]['window_size'] == 100

        # Invalid type window_size will raise TypeError.
        custom_cfg = [dict(data_src='loss_cls', window_size=[])]
        with pytest.raises(TypeError):
            log_processor._parse_windows_size(custom_cfg, runner, 1)

    def test_get_log(self):
        runner = self._setup_runner()
        # Test train mode.
        log_processor = LogProcessor()
        log_processor._get_train_log_str = MagicMock()
        log_processor._get_val_log_str = MagicMock()
        log_processor.get_log(runner, 1, 'train')
        log_processor._get_train_log_str.assert_called()
        log_processor._get_val_log_str.assert_not_called()

        # Test validation or test mode.
        log_processor = LogProcessor()
        log_processor._get_train_log_str = MagicMock()
        log_processor._get_val_log_str = MagicMock()
        log_processor.get_log(runner, 1, 'val')
        log_processor._get_train_log_str.assert_not_called()
        log_processor._get_val_log_str.assert_called()

        # Test error mode.
        with pytest.raises(ValueError):
            log_processor.get_log(runner, 1, 'trainn')

    @pytest.mark.parametrize('by_epoch', [True, False])
    def test_get_log_train_str(self, by_epoch):
        runner = self._setup_runner()
        # Prepare LoggerHook
        log_processor = LogProcessor(by_epoch=by_epoch)
        log_processor._get_max_memory = MagicMock(return_value='100')
        eta = 40
        runner.message_hub.update_info('eta', eta)
        # Prepare training information.
        train_logs = dict(
            lr=0.1, time=1.0, data_time=1.0, loss_cls=1.0)
        cur_iter = 2 if by_epoch else 11
        out = log_processor._get_train_log_str(runner, train_logs, cur_iter)
        # Verify that the correct context have been logged.
        if by_epoch:
            if torch.cuda.is_available():
                log_str = 'Epoch [2][2/5]\t' \
                          f"lr: {train_logs['lr']:.3e}, " \
                          f'eta: 0:00:40, ' \
                          f"time: {train_logs['time']:.3f}, " \
                          f"data_time: {train_logs['data_time']:.3f}, " \
                          f'memory: 100, ' \
                          f"loss_cls: {train_logs['loss_cls']:.4f}"
            else:
                log_str = 'Epoch [2][2/5]\t' \
                          f"lr: {train_logs['lr']:.3e}, " \
                          f'eta: 0:00:40, ' \
                          f"time: {train_logs['time']:.3f}, " \
                          f"data_time: {train_logs['data_time']:.3f}, " \
                          f"loss_cls: {train_logs['loss_cls']:.4f}"
            assert out == log_str
        else:
            if torch.cuda.is_available():
                log_str = 'Iter [11/50]\t' \
                          f"lr: {train_logs['lr']:.3e}, " \
                          f'eta: 0:00:40, ' \
                          f"time: {train_logs['time']:.3f}, " \
                          f"data_time: {train_logs['data_time']:.3f}, " \
                          f'memory: 100, ' \
                          f"loss_cls: {train_logs['loss_cls']:.4f}"
            else:
                log_str = 'Iter [11/50]\t' \
                          f"lr: {train_logs['lr']:.3e}, " \
                          f'eta: 0:00:40, ' \
                          f"time: {train_logs['time']:.3f}, " \
                          f"data_time: {train_logs['data_time']:.3f}, " \
                          f"loss_cls: {train_logs['loss_cls']:.4f}"
            assert out == log_str

    @pytest.mark.parametrize('by_epoch', [True, False])
    def test_log_val(self, by_epoch):
        runner = self._setup_runner()
        # Prepare LoggerHook
        log_processor = LogProcessor(by_epoch=by_epoch)
        # Prepare validation information.
        val_logs = dict(accuracy=0.9, data_time=1.0)
        cur_iter = 2
        out = log_processor._get_val_log_str(runner, val_logs, cur_iter)
        if by_epoch:
            assert out == 'Epoch(val) [1][5]\taccuracy: 0.9000, ' \
                          'data_time: 1.0000'

        else:
            assert out == 'Iter(val) [2][5]\taccuracy: 0.9000, ' \
                          'data_time: 1.0000'

    def test_collect_scalars(self):
        runner = self._setup_runner()
        custom_cfg = [
            dict(data_src='time', method_name='mean', window_size=100),
            dict(data_src='time', method_name='max', log_name='time_max')]
        logger_hook = LogProcessor(custom_cfg=custom_cfg)
        # Collect with prefix.
        log_scalars = {
            'train/time': MagicMock(),
            'lr': MagicMock(),
            'train/loss_cls': MagicMock(),
            'val/metric': MagicMock()
        }
        runner.message_hub._log_scalars = log_scalars
        tag = logger_hook._collect_scalars(copy.deepcopy(custom_cfg),
                                           runner, mode='train')
        # Test training key in tag.
        assert list(tag.keys()) == ['time', 'loss_cls', 'time_max']
        # Test statistics lr with `current`, loss and time with 'mean'
        log_scalars['train/time'].statistics.assert_called_with(
            method_name='max')
        log_scalars['train/loss_cls'].mean.assert_called()

        tag = logger_hook._collect_scalars(copy.deepcopy(custom_cfg),
                                           runner, mode='val')
        assert list(tag.keys()) == ['metric']
        log_scalars['val/metric'].current.assert_called()

    @patch('torch.cuda.max_memory_allocated', MagicMock())
    @patch('torch.cuda.reset_peak_memory_stats', MagicMock())
    def test_get_max_memory(self):
        logger_hook = LogProcessor()
        runner = MagicMock()
        runner.world_size = 1
        runner.model = torch.nn.Linear(1, 1)
        logger_hook._get_max_memory(runner)
        torch.cuda.max_memory_allocated.assert_called()
        torch.cuda.reset_peak_memory_stats.assert_called()

    def test_get_iter(self):
        runner = self._setup_runner()
        log_processor = LogProcessor()
        # Get global iter when `inner_iter=False`
        iter = log_processor._get_iter(runner)
        assert iter == 11
        # Get inner iter
        iter = log_processor._get_iter(runner, 1)
        assert iter == 2
        # Still get global iter when `logger_hook.by_epoch==False`
        log_processor.by_epoch = False
        iter = log_processor._get_iter(runner, 1)
        assert iter == 11

    def test_get_epoch(self):
        runner = self._setup_runner()
        log_processor = LogProcessor()
        epoch = log_processor._get_epoch(runner, 'train')
        assert epoch == 2
        epoch = log_processor._get_epoch(runner, 'val')
        assert epoch == 1
        with pytest.raises(ValueError):
            log_processor._get_epoch(runner, 'test')

    def _setup_runner(self):
        runner = MagicMock()
        runner.epoch = 1
        runner.cur_dataloader = [0] * 5
        runner.iter = 10
        runner.train_loop.max_iters = 50
        logger = MMLogger.get_instance('log_processor_test')
        runner.logger = logger
        message_hub = MessageHub.get_instance('log_processor_test')
        for i in range(10):
            message_hub.update_scalar('train/loss', 10-i)
        for i in range(10):
            message_hub.update_scalar('val/acc', i*0.1)
        runner.message_hub = message_hub
        return runner
