from mmengine.hooks import LoggerHook
import pytest
from mmengine.fileio.file_client import HardDiskBackend
from unittest.mock import MagicMock, patch

from collections import OrderedDict


class TestLoggerHook:
    def test_init(self):
        logger_hook = LoggerHook(out_dir='tmp.txt')
        assert logger_hook.by_epoch
        assert logger_hook.interval == 10
        assert logger_hook.custom_keys is None
        assert logger_hook.composed_writers is None
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
        LoggerHook(custom_keys=dict(time=dict(method='max',
                                              log_name='time_max')))
        with pytest.raises(AssertionError):
            LoggerHook(by_epoch=False,
                       custom_keys=dict(
                           time=dict(method='max',
                                     log_name='time_max',
                                     window_size='epoch')))

    def test_before_run(self):
        runner = MagicMock()
        logger_hook = LoggerHook()
        logger_hook.logger = MagicMock()
        logger_hook.composed_writers = MagicMock()
        logger_hook.before_run(runner)

    def test_after_run(self, tmp_path):
        json_path = tmp_path / 'tmp.json'
        json_file = open(json_path, 'w')
        json_file.close()

        runner = MagicMock()
        runner.work_dir = tmp_path
        logger_hook = LoggerHook(out_dir=str(tmp_path / 'out_path'),
                                 keep_local=False)
        logger_hook.logger = MagicMock()
        logger_hook.composed_writers = MagicMock()
        logger_hook.after_run(runner)

    @pytest.mark.parametrize('by_epoch', [True, False], )
    def test_after_train_iter(self, by_epoch):
        runner = MagicMock()
        runner.iter = 10
        runner.inner_iter = 10
        logger_hook = LoggerHook(ignore_last=False)
        logger_hook.log_train = MagicMock()
        logger_hook.after_train_iter(runner)
        logger_hook.end_of_epoch = MagicMock(return_value=True)
        logger_hook.after_train_iter(runner)

    def test_after_val_epoch(self):
        runner = MagicMock()
        logger_hook = LoggerHook()
        logger_hook._collect_info(runner, mode='train')

    @pytest.mark.parametrize('by_epoch', [True, False])
    def test_log_train(self, by_epoch):
        runner = MagicMock()
        runner.meta = dict(exp_name='retinanet')

        logger_hook = LoggerHook(by_epoch=by_epoch)
        logger_hook.logger = MagicMock()
        logger_hook.composed_writers = MagicMock()
        logger_hook.time_sec_tot = 1000
        logger_hook.start_iter = 0
        logger_hook._collect_info = MagicMock(return_value=
                                              dict(lr=1, time=1, data_time=1))
        logger_hook._get_max_memory = MagicMock(return_value='100')

        logger_hook.log_train(runner)

    @pytest.mark.parametrize('by_epoch', [True, False])
    def test_log_val(self, by_epoch,):
        runner = MagicMock()

        logger_hook = LoggerHook(by_epoch=by_epoch)
        logger_hook.logger = MagicMock()
        logger_hook.composed_writers = MagicMock()
        logger_hook._collect_info = MagicMock(return_value=
                                              dict(lr=1, time=1, data_time=1))
        logger_hook.log_val(runner)

    @pytest.mark.parametrize('window_size', ['epoch', 'global',
                                             'current', 10, 20])
    def test_get_window_size(self, window_size):
        runner = MagicMock()
        runner.inner_iter = 1
        runner.iter = 10
        logger_hook = LoggerHook()
        # Test get window size by name.
        if window_size == 'epoch':
            assert logger_hook._get_window_size(runner, window_size) == 2
        if window_size == 'global':
            assert logger_hook._get_window_size(runner, window_size) == 11
        if window_size == 10:
            assert logger_hook._get_window_size(runner, window_size) == 10
        # Window size must equal to `logger_hook.interval`.
        if window_size == 20:
            with pytest.raises(AssertionError):
                logger_hook._get_window_size(runner, window_size)

    def test_parse_custom_keys(self):
        tag = OrderedDict()
        runner = MagicMock()
        log_buffers = OrderedDict(lr=MagicMock(),
                                  loss=MagicMock())
        cfg_dict = dict(lr=dict(method='min'),
                        loss=[dict(method='min'),
                              dict(method='max', log_name='loss_max')])
        logger_hook = LoggerHook()
        logger_hook.custom_keys = cfg_dict
        logger_hook._statistics_single_key = MagicMock()
        for log_key, log_cfg in cfg_dict.items():
            logger_hook._parse_custom_keys(runner, log_key, log_cfg,
                                           log_buffers, tag)
        assert list(tag) == ['lr', 'loss', 'loss_max']
        assert log_buffers['lr'].min.assert_called
        assert log_buffers['loss'].min.assert_called
        assert log_buffers['loss'].max.assert_called
        assert log_buffers['loss'].mean.assert_called

    def test_collect_info(self):
        runner = MagicMock()
        runner.message_hub = MagicMock()
        logger_hook = LoggerHook()
        # Collect with prefix.
        log_buffers = {'train/time': MagicMock(),
                       'lr': MagicMock(),
                       'train/loss_cls': MagicMock(),
                       'val/metric': MagicMock()}
        runner.message_hub.log_buffers = log_buffers
        tag = logger_hook._collect_info(runner, mode='train')
        # Test training key in tag.
        assert list(tag.keys()) == ['time', 'loss_cls']
        # Test statistics lr with `current`, loss and time with 'mean'
        log_buffers['train/time'].mean.assert_called()
        log_buffers['train/loss_cls'].mean.assert_called()
        log_buffers['train/loss_cls'].current.assert_not_called()

        tag = logger_hook._collect_info(runner, mode='val')
        assert list(tag.keys()) == ['metric']
        log_buffers['val/metric'].current.assert_called()











