# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from parameterized import parameterized

from mmengine.device import is_cuda_available, is_musa_available
from mmengine.logging import HistoryBuffer, MessageHub, MMLogger
from mmengine.runner import LogProcessor
from mmengine.testing import RunnerTestCase


class TestLogProcessor(RunnerTestCase):

    def test_init(self):
        log_processor = LogProcessor(
            window_size=10, by_epoch=True, custom_cfg=None)
        assert log_processor.by_epoch
        assert log_processor.window_size == 10
        assert log_processor.custom_cfg == []

    def test_check_custom_cfg(self):
        # ``by_epoch==False`` and `window_size='epoch'` in log config will
        # raise AssertionError.
        custom_cfg = [dict(data_src='loss', window_size='epoch')]
        with pytest.raises(AssertionError):
            LogProcessor(by_epoch=False, custom_cfg=custom_cfg)
        # Duplicate log_name will raise AssertionError.
        custom_cfg = [
            dict(data_src='loss', log_name='loss_1'),
            dict(data_src='loss', log_name='loss_1')
        ]
        with pytest.raises(AssertionError):
            LogProcessor(custom_cfg=custom_cfg)
        # Overwrite loss item twice will raise AssertionError.
        custom_cfg = [dict(data_src='loss'), dict(data_src='loss')]
        with pytest.raises(AssertionError):
            LogProcessor(custom_cfg=custom_cfg)

        custom_cfg = [
            dict(data_src='loss_cls', window_size=100, method_name='min'),
            dict(data_src='loss', log_name='loss_min', method_name='max'),
            dict(data_src='loss', log_name='loss_max', method_name='max')
        ]
        LogProcessor(custom_cfg=custom_cfg)

    def test_parse_windows_size(self):
        log_processor = LogProcessor()
        # Test parse 'epoch' window_size.
        custom_cfg = [dict(data_src='loss_cls', window_size='epoch')]
        custom_cfg = log_processor._parse_windows_size(self.runner, 1,
                                                       custom_cfg)
        assert custom_cfg[0]['window_size'] == 2

        # Test parse 'global' window_size.
        custom_cfg = [dict(data_src='loss_cls', window_size='global')]
        custom_cfg = log_processor._parse_windows_size(self.runner, 1,
                                                       custom_cfg)
        assert custom_cfg[0]['window_size'] == 11

        # Test parse int window_size
        custom_cfg = [dict(data_src='loss_cls', window_size=100)]
        custom_cfg = log_processor._parse_windows_size(self.runner, 1,
                                                       custom_cfg)
        assert custom_cfg[0]['window_size'] == 100

        # Invalid type window_size will raise TypeError.
        custom_cfg = [dict(data_src='loss_cls', window_size=[])]
        with pytest.raises(TypeError):
            log_processor._parse_windows_size(self.runner, 1, custom_cfg)

    # yapf: disable
    @parameterized.expand(
        ([True, 'train', True], [True, 'train', False], [False, 'train', True],
         [False, 'train', False], [True, 'val', True], [True, 'val', False],
         [False, 'val', True], [False, 'val', False], [True, 'test', True],
         [True, 'test', False], [False, 'test', True], [False, 'test', False]))
    # yapf: enable
    def test_get_log_after_iter(self, by_epoch, mode, log_with_hierarchy):
        # Prepare LoggerHook
        log_processor = LogProcessor(
            by_epoch=by_epoch, log_with_hierarchy=log_with_hierarchy)
        log_processor._get_max_memory = MagicMock(return_value='100')
        eta = 40
        self.runner.message_hub.update_info('eta', eta)
        # Prepare training information.
        if mode == 'train':
            train_logs = dict(lr=0.1, time=1.0, data_time=1.0, loss_cls=1.0)
        else:
            train_logs = dict(time=1.0, data_time=1.0, loss_cls=1.0)
        log_processor._collect_scalars = \
            lambda *args, **kwargs: copy.deepcopy(train_logs)
        tag, out = log_processor.get_log_after_iter(self.runner, 1, mode)

        # Verify that the correct context have been logged.
        cur_loop = log_processor._get_cur_loop(self.runner, mode)
        if by_epoch:
            if mode in ['train', 'val']:
                cur_epoch = log_processor._get_epoch(self.runner, mode)
                log_str = (f'Epoch({mode})  [{cur_epoch}][ 2/'
                           f'{len(cur_loop.dataloader)}]  ')
            else:
                log_str = (f'Epoch({mode}) [2/{len(cur_loop.dataloader)}]  ')

            if mode == 'train':
                log_str += f"lr: {train_logs['lr']:.4e}  "
            else:
                log_str += '  '

            log_str += (f'eta: 0:00:40  '
                        f"time: {train_logs['time']:.4f}  "
                        f"data_time: {train_logs['data_time']:.4f}  ")

            if is_cuda_available() or is_musa_available():
                log_str += 'memory: 100  '
            if mode == 'train':
                log_str += f"loss_cls: {train_logs['loss_cls']:.4f}"
            assert out == log_str

            if mode in ['train', 'val']:
                assert 'epoch' in tag
        else:
            if mode == 'train':
                max_iters = self.runner.max_iters
                log_str = f'Iter({mode}) [11/{max_iters}]  '
            elif mode == 'val':
                max_iters = len(cur_loop.dataloader)
                log_str = f'Iter({mode}) [ 2/{max_iters}]  '
            else:
                max_iters = len(cur_loop.dataloader)
                log_str = f'Iter({mode}) [2/{max_iters}]  '

            if mode == 'train':
                log_str += f"lr: {train_logs['lr']:.4e}  "
            else:
                log_str += '  '

            log_str += (f'eta: 0:00:40  '
                        f"time: {train_logs['time']:.4f}  "
                        f"data_time: {train_logs['data_time']:.4f}  ")

            if is_cuda_available() or is_musa_available():
                log_str += 'memory: 100  '

            if mode == 'train':
                log_str += f"loss_cls: {train_logs['loss_cls']:.4f}"
            assert out == log_str

        # tag always has "iter" key
        assert 'iter' in tag

    @parameterized.expand(
        ([True, 'val', True], [True, 'val', False], [False, 'val', True],
         [False, 'val', False], [True, 'test', True], [False, 'test', False]))
    def test_log_val(self, by_epoch, mode, log_with_hierarchy):
        # Prepare LoggerHook
        log_processor = LogProcessor(
            by_epoch=by_epoch, log_with_hierarchy=log_with_hierarchy)
        # Prepare validation information.
        scalar_logs = dict(accuracy=0.9, data_time=1.0)
        non_scalar_logs = dict(
            recall={
                'cat': 1,
                'dog': 0
            }, cm=torch.tensor([1, 2, 3]))
        log_processor._collect_scalars = MagicMock(return_value=scalar_logs)
        log_processor._collect_non_scalars = MagicMock(
            return_value=non_scalar_logs)
        _, out = log_processor.get_log_after_epoch(self.runner, 2, mode)
        expect_metric_str = ("accuracy: 0.9000  recall: {'cat': 1, 'dog': 0}  "
                             'cm: \ntensor([1, 2, 3])\n  data_time: 1.0000')
        if by_epoch:
            if mode == 'test':
                assert out == 'Epoch(test) [5/5]    ' + expect_metric_str
            else:
                assert out == 'Epoch(val) [1][10/10]    ' + expect_metric_str
        else:
            if mode == 'test':
                assert out == 'Iter(test) [5/5]    ' + expect_metric_str
            else:
                assert out == 'Iter(val) [10/10]    ' + expect_metric_str

    def test_collect_scalars(self):
        history_count = np.ones(100)
        time_scalars = np.random.randn(100)
        loss_cls_scalars = np.random.randn(100)
        lr_scalars = np.random.randn(100)
        metric_scalars = np.random.randn(100)

        history_time_buffer = HistoryBuffer(time_scalars, history_count)
        histroy_loss_cls = HistoryBuffer(loss_cls_scalars, history_count)
        history_lr_buffer = HistoryBuffer(lr_scalars, history_count)
        history_metric_buffer = HistoryBuffer(metric_scalars, history_count)

        custom_cfg = [
            dict(data_src='time', method_name='max', log_name='time_max')
        ]
        log_processor = LogProcessor(custom_cfg=custom_cfg)
        # Collect with prefix.
        log_scalars = {
            'train/time': history_time_buffer,
            'lr': history_lr_buffer,
            'train/loss_cls': histroy_loss_cls,
            'val/metric': history_metric_buffer
        }
        self.runner.message_hub._log_scalars = log_scalars
        tag = log_processor._collect_scalars(
            copy.deepcopy(custom_cfg), self.runner, mode='train')
        # Training key in tag.
        assert list(tag.keys()) == ['time', 'loss_cls', 'time_max']
        # Test statistics lr with `current`, loss and time with 'mean'
        assert tag['time'] == time_scalars[-10:].mean()
        assert tag['time_max'] == time_scalars.max()
        assert tag['loss_cls'] == loss_cls_scalars[-10:].mean()

        # Validation key in tag
        tag = log_processor._collect_scalars(
            copy.deepcopy(custom_cfg), self.runner, mode='val')
        assert list(tag.keys()) == ['metric']
        assert tag['metric'] == metric_scalars[-1]

        # reserve_prefix=True
        tag = log_processor._collect_scalars(
            copy.deepcopy(custom_cfg),
            self.runner,
            mode='train',
            reserve_prefix=True)
        assert list(
            tag.keys()) == ['train/time', 'train/loss_cls', 'train/time_max']
        # Test statistics lr with `current`, loss and time with 'mean'
        assert tag['train/time'] == time_scalars[-10:].mean()
        assert tag['train/time_max'] == time_scalars.max()
        assert tag['train/loss_cls'] == loss_cls_scalars[-10:].mean()

    def test_collect_non_scalars(self):
        metric1 = np.random.rand(10)
        metric2 = torch.tensor(10)

        log_processor = LogProcessor()
        # Collect with prefix.
        log_infos = {'test/metric1': metric1, 'test/metric2': metric2}
        self.runner.message_hub._runtime_info = log_infos
        tag = log_processor._collect_non_scalars(self.runner, mode='test')
        # Test training key in tag.
        assert list(tag.keys()) == ['metric1', 'metric2']
        # Test statistics lr with `current`, loss and time with 'mean'
        assert tag['metric1'] is metric1
        assert tag['metric2'] is metric2

    # TODO:haowen.han@mtheads.com MUSA does not support it yet!
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
        log_processor = LogProcessor()
        # Get batch_idx
        iter = log_processor._get_iter(self.runner, 1)
        assert iter == 2
        # Still get global iter when `logger_hook.by_epoch==False`
        log_processor.by_epoch = False
        iter = log_processor._get_iter(self.runner, 1)
        assert iter == 11

    def test_get_epoch(self):
        log_processor = LogProcessor()
        epoch = log_processor._get_epoch(self.runner, 'train')
        assert epoch == 2
        epoch = log_processor._get_epoch(self.runner, 'val')
        assert epoch == 1
        with pytest.raises(ValueError):
            log_processor._get_epoch(self.runner, 'test')

    def test_get_cur_loop(self):
        log_processor = LogProcessor()
        loop = log_processor._get_cur_loop(self.runner, 'train')
        assert len(loop.dataloader) == 20
        loop = log_processor._get_cur_loop(self.runner, 'val')
        assert len(loop.dataloader) == 10
        loop = log_processor._get_cur_loop(self.runner, 'test')
        assert len(loop.dataloader) == 5

    def setUp(self):
        runner = MagicMock()
        runner.epoch = 1
        runner.max_epochs = 10
        runner.iter = 10
        runner.max_iters = 50
        runner.train_dataloader = [0] * 20
        runner.val_dataloader = [0] * 10
        runner.test_dataloader = [0] * 5
        runner.train_loop.dataloader = [0] * 20
        runner.val_loop.dataloader = [0] * 10
        runner.test_loop.dataloader = [0] * 5
        logger = MMLogger.get_instance('log_processor_test')
        runner.logger = logger
        message_hub = MessageHub.get_instance('log_processor_test')
        for i in range(10):
            message_hub.update_scalar('train/loss', 10 - i)
        for i in range(10):
            message_hub.update_scalar('val/acc', i * 0.1)
        runner.message_hub = message_hub
        self.runner = runner
        super().setUp()

    def test_with_runner(self):
        cfg = self.epoch_based_cfg.copy()
        cfg.log_processor = dict(
            custom_cfg=[
                dict(
                    data_src='time',
                    window_size='epoch',
                    log_name='iter_time',
                    method_name='mean')
            ],
            log_with_hierarchy=True)
        runner = self.build_runner(cfg)
        runner.train()
        runner.val()
        runner.test()

        cfg = self.iter_based_cfg.copy()
        cfg.log_processor = dict(
            by_epoch=False,
            custom_cfg=[
                dict(
                    data_src='time',
                    window_size=100,
                    log_name='iter_time',
                    method_name='mean')
            ],
            log_with_hierarchy=True)
        runner = self.build_runner(cfg)
        runner.train()
        runner.val()
        runner.test()
