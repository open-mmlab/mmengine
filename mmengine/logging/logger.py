import sys
import platform
from typing import Optional

import logging
from logging import Logger
from termcolor import colored
import torch.distributed as dist

from .base_global_accsessible import BaseGlobalAccessible


class MMFormatter(logging.Formatter):
    _color_mapping = dict(ERROR='red', WARNING='yellow', INFO='white',
                          DEBUG='green')

    def __init__(self, color=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def _get_prefix(self, level, color):
        if color:
            prefix = colored(level, self._color_mapping[level],
                             attrs=["blink", "underline"])
        else:
            prefix = level
        return prefix

    def format(self, record):
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
    def __init__(self,
                 name: Optional[str] = None,
                 log_file: Optional[str] = None,
                 log_level: str = 'NOTSET',
                 file_mode='w'):
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
                if platform.system() == 'Windows':
                    separator = '\\'
                else:
                    separator = '/'
                path_split = log_file.split(separator)
                path_split[-1] = f'rank{rank}_{path_split[-1]}'
                log_file = separator.join(path_split)
            # Here, the default behaviour of the official logger is 'a'. Thus,
            # we provide an interface to change the file mode to the default
            # behaviour. `FileHandler` is not supported to have colors,
            # otherwise it will appear garbled.
            file_handler = logging.FileHandler(log_file, file_mode)
            file_handler.setFormatter(MMFormatter(color=False))
            file_handler.setLevel(log_level)
            self.handlers.append(file_handler)


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
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
    elif isinstance(logger, str):
        _logger = MMLogger.get_instance(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')
