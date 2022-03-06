# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import sys
from logging import Logger, LogRecord
from typing import Optional, Union

import torch.distributed as dist
from termcolor import colored

from .base_global_accsessible import BaseGlobalAccessible


class MMFormatter(logging.Formatter):
    """Colorful format for MMLogger. If the log level is error, the logger will
    additionally output the location of the code.

    Args:
        color (bool): Whether to use colorful format. filehandler is not
            allowed to use color format, otherwise it will be garbled.
    """
    _color_mapping: dict = dict(
        ERROR='red', WARNING='yellow', INFO='white', DEBUG='green')

    def __init__(self, color: bool = True, **kwargs):

        super().__init__(**kwargs)
        # Get prefix format according to color.
        error_prefix = self._get_prefix('ERROR', color)
        warn_prefix = self._get_prefix('WARNING', color)
        info_prefix = self._get_prefix('INFO', color)
        debug_prefix = self._get_prefix('DEBUG', color)
        # Config output format.
        self.err_format = f'%(asctime)s - %(name)s - {error_prefix} - ' \
                          f'%(pathname)s - %(funcName)s - %(lineno)d - ' \
                          '%(message)s'
        self.warn_format = f'%(asctime)s - %(name)s - {warn_prefix} - %(' \
                           'message)s'
        self.info_format = f'%(asctime)s - %(name)s - {info_prefix} - %(' \
                           'message)s'
        self.debug_format = f'%(asctime)s - %(name)s - {debug_prefix} - %(' \
                            'message)s'

    def _get_prefix(self, level: str, color: bool) -> str:
        """Get the prefix of the target log level.

        Args:
            level (str): log level.
            color (bool): Whether to get colorful prefix.

        Returns:
            str: The plain or colorful prefix.
        """
        if color:
            prefix = colored(
                level,
                self._color_mapping[level],
                attrs=['blink', 'underline'])
        else:
            prefix = level
        return prefix

    def format(self, record: LogRecord) -> str:
        """Override the `logging.Formatter.format`` method `. Output the
        message according to the specified log level.

        Args:
            record (LogRecord): A LogRecord instance represents an event being
                logged.

        Returns:
            str: Formatted result.
        """
        if record.levelno == logging.ERROR:
            self._style._fmt = self.err_format
        elif record.levelno == logging.WARNING:
            self._style._fmt = self.warn_format
        elif record.levelno == logging.INFO:
            self._style._fmt = self.info_format
        elif record.levelno == logging.DEBUG:
            self._style._fmt = self.debug_format

        result = logging.Formatter.format(self, record)
        return result


class MMLogger(Logger, BaseGlobalAccessible):
    """The Logger manager which can create formatted logger and get specified
    logger globally. MMLogger is created and accessed in the same way as
    BaseGlobalAccessible.

    Args:
        name (str): Logger name. Defaults to ''.
        log_file (str, optional): The log filename. If specified, a
            ``FileHandler`` will be added to the logger. Defaults to None.
        log_level: The log level of the handler. Defaults to 'NOTSET'.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.
    """

    def __init__(self,
                 name: str = '',
                 log_file: Optional[str] = None,
                 log_level: str = 'NOTSET',
                 file_mode: str = 'w'):
        Logger.__init__(self, name)
        BaseGlobalAccessible.__init__(self, name)
        # Get rank in DDP mode.
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        # Config stream_handler. If `rank != 0`. stream_handler can only
        # export ERROR logs.
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(MMFormatter(color=True))
        stream_handler.setLevel(log_level) if rank == 0 else \
            stream_handler.setLevel(logging.ERROR)
        self.handlers.append(stream_handler)

        if log_file is not None:
            if rank != 0:
                # rename `log_file` with rank prefix.
                path_split = log_file.split(os.sep)
                path_split[-1] = f'rank{rank}_{path_split[-1]}'
                log_file = os.sep.join(path_split)
            # Here, the default behaviour of the official logger is 'a'. Thus,
            # we provide an interface to change the file mode to the default
            # behaviour. `FileHandler` is not supported to have colors,
            # otherwise it will appear garbled.
            file_handler = logging.FileHandler(log_file, file_mode)
            file_handler.setFormatter(MMFormatter(color=False))
            file_handler.setLevel(log_level)
            self.handlers.append(file_handler)


def print_log(msg,
              logger: Optional[Union[Logger, str]] = None,
              level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (Logger or str, optional): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - "current": Log message via the latest created logger.
            - other str: the logger obtained with `MMLogger.get_instance`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif logger == 'current':
        logger_instance = MMLogger.get_instance(current=True)
        logger_instance.log(level, msg)
    elif isinstance(logger, str):
        try:
            _logger = MMLogger.get_instance(logger)
            _logger.log(level, msg)
        except AssertionError:
            raise ValueError(f'MMLogger: {logger} has not been created!')
    else:
        raise TypeError(
            '`logger` should be either a logging.Logger object, str, '
            f'"silent", "current" or None, but got {type(logger)}')
