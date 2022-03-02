from mmengine.hooks import LoggerHook
import pytest
from mmengine.fileio.file_client import HardDiskBackend
from unittest.mock import MagicMock, patch


class TestLoggerHook:
    def test_init(self):
        logger_hook = LoggerHook()
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

    def test_after_train_iter(self):
        runner = MagicMock()
        logger_hook = LoggerHook()
        logger_hook._collect_info(runner, mode='train')

    def test_after_val_epoch(self):
        runner = MagicMock()
        logger_hook = LoggerHook()
        logger_hook._collect_info(runner, mode='train')

    def test_collect_info(self):
        runner = MagicMock()
        runner.message_hub = MagicMock()
        logger_hook = LoggerHook()
        # Collect with prefix.
        runner.message_hub.log_buffers = {'train/time': MagicMock(),
                                          'lr': MagicMock(),
                                          'train/loss_cls': MagicMock()}
        logger_hook._collect_info(runner, mode='train')
        runner.message_hub.log_buffers['train/time'].mean.assert_called()
        runner.message_hub.log_buffers['lr'].current.assert_not_called()
        runner.message_hub.log_buffers['train/loss_cls'].mean.assert_called()







