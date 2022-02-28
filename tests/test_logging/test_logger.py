import pytest

from mmengine import MMLogger, print_log
import logging
from unittest.mock import patch
import os
import re
import sys


class TestLogger:
    regex_time = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}'

    @patch('torch.distributed.get_rank', lambda: 0)
    @patch('torch.distributed.is_initialized', lambda: True)
    @patch('torch.distributed.is_available', lambda: True)
    def test_init_rank0(self, tmp_path):
        logger = MMLogger.create_instance('rank0.pkg1', log_level='INFO')
        assert logger.name == 'rank0.pkg1'
        assert logger.instance_name == 'rank0.pkg1'
        # Logger get from `MMLogger.get_instance` does not inherit from
        # `logging.root`
        assert logger.parent is None
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert logger.level == logging.NOTSET
        assert logger.handlers[0].level == logging.INFO

        tmp_file = tmp_path / 'tmp_file.log'
        logger = MMLogger.create_instance('rank0.pkg2',
                                          log_level='INFO',
                                          log_file=str(tmp_file))
        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) == 2
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert isinstance(logger.handlers[1], logging.FileHandler)
        logger_pkg3 = MMLogger.get_instance('rank0.pkg2')
        assert id(logger_pkg3) == id(logger)
        logging.shutdown()

    @patch('torch.distributed.get_rank', lambda: 1)
    @patch('torch.distributed.is_initialized', lambda: True)
    @patch('torch.distributed.is_available', lambda: True)
    def test_init_rank1(self, tmp_path):
        tmp_file = tmp_path / 'tmp_file.log'
        log_path = tmp_path / 'rank1_tmp_file.log'
        logger = MMLogger.create_instance('rank1.pkg2',
                                          log_level='INFO',
                                          log_file=str(tmp_file))
        assert len(logger.handlers) == 2
        assert logger.handlers[0].level == logging.ERROR
        assert logger.handlers[1].level == logging.INFO
        assert os.path.exists(log_path)
        logging.shutdown()

    @pytest.mark.parametrize('log_level', [logging.WARNING,
                                           logging.INFO,
                                           logging.DEBUG])
    def test_handler(self, capsys, tmp_path, log_level):
        # test stream handler can output correct format logs
        logger_name = f'test_stream_{str(log_level)}'
        logger = MMLogger.create_instance(logger_name,
                                          log_level=log_level)
        logger.log(level=log_level, msg='welcome')
        out, _ = capsys.readouterr()
        # Skip match colored INFO
        loglevl_name = logging._levelToName[log_level]
        match = re.fullmatch(self.regex_time + f' - {logger_name} - '
                             f'(.*){loglevl_name}(.*) - welcome\n', out)
        assert match is not None

        # test file_handler output plain text without color.
        tmp_file = tmp_path / 'tmp_file.log'
        logger_name = f'test_file_{log_level}'
        logger = MMLogger.create_instance(logger_name, log_level=log_level,
                                          log_file=tmp_file)
        logger.log(level=log_level, msg='welcome')
        with open(tmp_file, 'r') as f:
            log_text = f.read()
            match = re.fullmatch(self.regex_time +
                                 f' - {logger_name} - {loglevl_name} - '
                                 f'welcome\n',
                                 log_text)
            assert match is not None
        logging.shutdown()

    def test_erro_format(self, capsys):
        # test error level log can output file path, function name and
        # line number
        logger = MMLogger.create_instance('test_error', log_level='INFO')
        logger.error('welcome')
        lineno = sys._getframe().f_lineno - 1
        file_path = __file__
        function_name = sys._getframe().f_code.co_name
        pattern = self.regex_time + \
                  r' - test_error - (.*)ERROR(.*) - ' + \
                  f'{file_path} - {function_name} - {lineno} - welcome\n'
        out, _ = capsys.readouterr()
        match = re.fullmatch(pattern, out)
        assert match is not None

    def test_print_log(self, capsys, tmp_path):
        # caplog cannot record MMLogger's logs.
        # test simple print.
        print_log('welcome', logger=None)
        out, _ = capsys.readouterr()
        assert out == 'welcome\n'
        # test silent logger and skip print.
        print_log('welcome', logger='silent')
        out, _ = capsys.readouterr()
        assert out == ''



