from mmengine import MMLogger, print_log
import logging
from unittest.mock import patch
import os
import re


class TestLogger:
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
        # the name can not be used to open the file a second time in windows,
        # so `delete` should be set as `False` and we need to manually remove
        # it more details can be found at https://github.com/open-mmlab/mmcv/pull/1077 # noqa: E501
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
        os.remove(tmp_file)

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
        os.remove(log_path)


def test_print_log_print(capsys):
    print_log('welcome', logger=None)
    out, _ = capsys.readouterr()
    assert out == 'welcome\n'


def test_print_log_silent(capsys, caplog):
    print_log('welcome', logger='silent')
    out, _ = capsys.readouterr()
    assert out == ''
    assert len(caplog.records) == 0


def test_log_print(capsys, caplog, tmp_path):
    # MMLogger.create_instance('rank0.pkg3', log_level='ERROR')
    # print_log('welcome', logger='rank0.pkg3')
    logger = logging.getLogger('rank0.pkg3')
    logger.info('aaa')
    print(f'============================={caplog.handler}================')
    assert caplog.record_tuples[-1] == ('rank0.pkg3', logging.INFO, 'welcome')

    print_log('welcome', logger='mmcv', level=logging.ERROR)
    assert caplog.record_tuples[-1] == ('rank0.pkg3', logging.ERROR, 'welcome')

    # the name can not be used to open the file a second time in windows,
    # so `delete` should be set as `False` and we need to manually remove it
    # more details can be found at https://github.com/open-mmlab/mmcv/pull/1077
    tmp_file = tmp_path / 'tmp_file.log'
    logger = MMLogger.create_instance('abc', log_file=str(tmp_file))
    print_log('welcome', logger=logger)
    assert caplog.record_tuples[-1] == ('abc', logging.INFO, 'welcome')
    with open(tmp_file, 'r') as fin:
        log_text = fin.read()
        regex_time = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}'
        match = re.fullmatch(regex_time + r' - abc - INFO - welcome\n',
                             log_text)
        assert match is not None
    # flushing and closing all handlers in order to remove `f.name`
    logging.shutdown()

    os.remove(tmp_file)