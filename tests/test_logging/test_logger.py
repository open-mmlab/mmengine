# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import re
import sys
from collections import OrderedDict
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from mmengine.logging import MMLogger, print_log
from mmengine.logging.logger import _get_device_id
from mmengine.utils import is_installed


class TestLogger:
    stream_handler_regex_time = r'\d{2}/\d{2} \d{2}:\d{2}:\d{2}'
    file_handler_regex_time = r'\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}'

    @patch('mmengine.logging.logger._get_rank', lambda: 0)
    def test_init_rank0(self, tmp_path):
        logger = MMLogger.get_instance('rank0.pkg1', log_level='INFO')
        assert logger.name == 'mmengine'
        assert logger.instance_name == 'rank0.pkg1'
        assert logger.instance_name == 'rank0.pkg1'
        # Logger get from `MMLogger.get_instance` does not inherit from
        # `logging.root`
        assert logger.parent is None
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert logger.level == logging.NOTSET
        assert logger.handlers[0].level == logging.INFO
        # If `rank=0`, the `log_level` of stream_handler and file_handler
        # depends on the given arguments.
        tmp_file = tmp_path / 'tmp_file.log'
        logger = MMLogger.get_instance(
            'rank0.pkg2', log_level='INFO', log_file=str(tmp_file))
        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) == 2
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert isinstance(logger.handlers[1], logging.FileHandler)
        logger_pkg3 = MMLogger.get_instance('rank0.pkg2')
        assert id(logger_pkg3) == id(logger)
        logger = MMLogger.get_instance(
            'rank0.pkg3', logger_name='logger_test', log_level='INFO')
        assert logger.name == 'logger_test'
        assert logger.instance_name == 'rank0.pkg3'
        # `FileHandler` should be closed in Windows, otherwise we cannot
        # delete the temporary directory
        logging.shutdown()
        MMLogger._instance_dict.clear()

    @patch('mmengine.logging.logger._get_rank', lambda: 1)
    @patch('mmengine.logging.logger._get_device_id', lambda: 1)
    @patch('mmengine.logging.logger._get_world_size', lambda: 2)
    @patch('mmengine.logging.logger._get_host_info', lambda: 'test')
    def test_init_rank1(self, tmp_path):
        # If `rank!=1`, the `loglevel` of file_handler is `logging.ERROR`.
        tmp_file = tmp_path / 'tmp_file.log'
        log_path = tmp_path / 'tmp_file_test_device1_rank1.log'
        logger = MMLogger.get_instance(
            'rank1.pkg2', log_level='INFO', log_file=str(tmp_file))
        assert len(logger.handlers) == 1
        logger = MMLogger.get_instance(
            'rank1.pkg3',
            log_level='INFO',
            log_file=str(tmp_file),
            distributed=True)
        assert logger.handlers[0].level == logging.ERROR
        assert logger.handlers[1].level == logging.INFO
        assert len(logger.handlers) == 2
        assert os.path.exists(str(log_path))
        # `FileHandler` should be closed in Windows, otherwise we cannot
        # delete the temporary directory
        logging.shutdown()
        MMLogger._instance_dict.clear()

    @pytest.mark.parametrize('log_level',
                             [logging.WARNING, logging.INFO, logging.DEBUG])
    def test_handler(self, capsys, tmp_path, log_level):
        # test stream handler can output correct format logs
        instance_name = f'test_stream_{str(log_level)}'
        logger = MMLogger.get_instance(instance_name, log_level=log_level)
        logger.log(level=log_level, msg='welcome')
        out, _ = capsys.readouterr()
        # Skip match colored INFO
        loglevl_name = logging._levelToName[log_level]
        match = re.fullmatch(
            self.stream_handler_regex_time + f' - mmengine - '
            f'(.*){loglevl_name}(.*) - welcome\n', out)
        assert match is not None

        # test file_handler output plain text without color.
        tmp_file = tmp_path / 'tmp_file.log'
        instance_name = f'test_file_{log_level}'
        logger = MMLogger.get_instance(
            instance_name, log_level=log_level, log_file=tmp_file)
        logger.log(level=log_level, msg='welcome')

        with open(tmp_file) as f:
            log_text = f.read()
            match = re.fullmatch(
                self.file_handler_regex_time +
                f' - mmengine - {loglevl_name} - '
                f'welcome\n', log_text)
            assert match is not None
        # `FileHandler` should be closed in Windows, otherwise we cannot
        # delete the temporary directory
        logging.shutdown()
        MMLogger._instance_dict.clear()

    def test_error_format(self, capsys):
        # test error level log can output file path, function name and
        # line number
        logger = MMLogger.get_instance('test_error', log_level='INFO')
        logger.error('welcome')
        lineno = sys._getframe().f_lineno - 1
        # replace \ for windows:
        # origin: c:\\a\\b\\c.py
        # replaced: c:\\\\a\\\\b\\\\c.py for re.match.
        file_path = __file__.replace('\\', '\\\\')
        function_name = sys._getframe().f_code.co_name
        pattern = self.stream_handler_regex_time + \
            r' - mmengine - (.*)ERROR(.*) - ' \
            f'{file_path} - {function_name} - ' \
            f'{lineno} - welcome\n'
        out, _ = capsys.readouterr()
        match = re.fullmatch(pattern, out)
        assert match is not None

    def test_print_log(self, capsys, tmp_path):
        # caplog cannot record MMLogger's logs.
        # Test simple print.
        print_log('welcome', logger=None)
        out, _ = capsys.readouterr()
        assert out == 'welcome\n'
        # Test silent logger and skip print.
        print_log('welcome', logger='silent')
        out, _ = capsys.readouterr()
        assert out == ''
        logger = MMLogger.get_instance('test_print_log')
        # Test using specified logger
        print_log('welcome', logger=logger)
        out, _ = capsys.readouterr()
        match = re.fullmatch(
            self.stream_handler_regex_time + ' - mmengine - (.*)INFO(.*) - '
            'welcome\n', out)
        assert match is not None
        # Test access logger by name.
        print_log('welcome', logger='test_print_log')
        out, _ = capsys.readouterr()
        match = re.fullmatch(
            self.stream_handler_regex_time + ' - mmengine - (.*)INFO(.*) - '
            'welcome\n', out)
        assert match is not None
        # Test access the latest created logger.
        print_log('welcome', logger='current')
        out, _ = capsys.readouterr()
        match = re.fullmatch(
            self.stream_handler_regex_time + ' - mmengine - (.*)INFO(.*) - '
            'welcome\n', out)
        assert match is not None
        # Test invalid logger type.
        with pytest.raises(TypeError):
            print_log('welcome', logger=dict)
        with pytest.raises(ValueError):
            print_log('welcome', logger='unknown')

    def test_get_instance(self):
        # Test get root mmengine logger.
        MMLogger._instance_dict = OrderedDict()
        root_logger = MMLogger.get_current_instance()
        mmdet_logger = MMLogger.get_instance('mmdet')
        assert root_logger.name == mmdet_logger.name
        assert id(root_logger) != id(mmdet_logger)
        assert id(MMLogger.get_instance('mmengine')) == id(root_logger)
        # Test original `get_current_instance` function.
        MMLogger.get_instance('mmdet')
        assert MMLogger.get_current_instance().instance_name == 'mmdet'

    def test_set_level(self, capsys):
        logger = MMLogger.get_instance('test_set_level')
        logger.info('hello')
        out, _ = capsys.readouterr()
        assert 'INFO' in out
        logger.setLevel('WARNING')
        logger.info('hello')
        out, _ = capsys.readouterr()
        assert not out
        logger.warning('hello')
        out, _ = capsys.readouterr()
        assert 'WARNING' in out

    def test_filter(self, capsys):
        logger = MMLogger.get_instance('test_filter')
        logger.warning('hello')
        out, _ = capsys.readouterr()
        assert 'WARNING' in out
        # Filter repeated warning.
        logger.warning('hello')
        out, _ = capsys.readouterr()
        assert not out
        # Pass new warning
        logger.warning('hello1')
        out, _ = capsys.readouterr()
        assert 'WARNING' in out

    def test_file_handlers(self, tmp_path):
        tmp_file = tmp_path / 'tmp_file.log'
        fh = None
        logger = MMLogger(
            name='test_file_handlers', log_file=tmp_file, file_handler_cfg=fh)
        assert isinstance(logger.handlers[-1], logging.FileHandler)
        fh = dict(type='BaseRotatingHandler', mode='a')
        logger = MMLogger(
            name='test_file_handlers', log_file=tmp_file, file_handler_cfg=fh)
        assert isinstance(logger.handlers[-1],
                          logging.handlers.BaseRotatingHandler)
        fh = dict(type='RotatingFileHandler', maxBytes=1024)
        logger = MMLogger(
            name='test_file_handlers', log_file=tmp_file, file_handler_cfg=fh)
        assert isinstance(logger.handlers[-1],
                          logging.handlers.RotatingFileHandler)
        fh = dict(type='TimedRotatingFileHandler', when='MIDNIGHT')
        logger = MMLogger(
            name='test_file_handlers', log_file=tmp_file, file_handler_cfg=fh)
        assert isinstance(logger.handlers[-1],
                          logging.handlers.TimedRotatingFileHandler)
        fh = dict(type='WatchedFileHandler')
        logger = MMLogger(
            name='test_file_handlers', log_file=tmp_file, file_handler_cfg=fh)
        assert isinstance(logger.handlers[-1],
                          logging.handlers.WatchedFileHandler)
        # `FileHandler` should be closed in Windows, otherwise we cannot
        # delete the temporary directory
        logging.shutdown()
        MMLogger._instance_dict.clear()


@pytest.mark.skipif(not is_installed('torch'), reason='tests requires torch')
@patch('torch.cuda.device_count', lambda: 4)
def test_get_device_id():

    @contextmanager
    def patch_env(local_rank, cuda_visible_devices):
        ori_local_rank = os.getenv('LOCAL_RANK', None)
        ori_cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', None)

        if local_rank is not None:
            os.environ['LOCAL_RANK'] = local_rank
        if cuda_visible_devices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
        yield
        if ori_local_rank is not None:
            os.environ['LOCAL_RANK'] = ori_local_rank
        elif 'LOCAL_RANK' in os.environ:
            os.environ.pop('LOCAL_RANK')
        if ori_cuda_visible_devices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = ori_cuda_visible_devices
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            os.environ.pop('CUDA_VISIBLE_DEVICES')

    # cuda is not available and local_rank is not set
    with patch('torch.cuda.is_available', lambda: False), \
         patch_env(None, '0,1,2,3'):
        assert _get_device_id() == 0

    # cuda is not available and local_rank is set
    with patch('torch.cuda.is_available', lambda: False), \
         patch_env('1', '0,1,2,3'):
        assert _get_device_id() == 1

    # CUDA_VISIBLE_DEVICES will not influence non-cuda device
    with patch('torch.cuda.is_available', lambda: False), \
         patch_env('1', '0,100,2,3'):
        assert _get_device_id() == 1

    # cuda is available and local_rank is not set
    with patch('torch.cuda.is_available', lambda: True), \
         patch_env(None, '0,1,2,3'):
        assert _get_device_id() == 0

    # cuda is available and local_rank is set
    with patch('torch.cuda.is_available', lambda: True), \
         patch_env('2', '0,1,2,3'):
        assert _get_device_id() == 2

    # CUDA_VISIBLE_DEVICES worked
    with patch('torch.cuda.is_available', lambda: True), \
         patch_env('2', '0,1,3,5'):
        assert _get_device_id() == 3
