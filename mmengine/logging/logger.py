import sys
import platform
from typing import Optional

import logging
from logging import Logger
from termcolor import colored

from .base_global_accsessible import BaseGlobalAccessible


class MMFormatter(logging.Formatter):
    _color_mapping = dict(ERROR='red', WARN='yellow', INFO='white',
                          debug='green')

    def __init__(self, color=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get prefix format according to color.
        error_prefix = self._get_prefix('ERROR', color)
        warn_prefix = self._get_prefix('WARN', color)
        info_prefix = self._get_prefix('INFO', color)
        debug_prefix = self._get_prefix('debug', color)
        # Config output format.
        self.err_format = f'%(asctime)s - %(name)s - {error_prefix} ' \
                          f'Filepath: %(pathname)s Function: %(funcName)s ' \
                          f'Line: %(lineno)d'
        self.warn_format = f'%(asctime)s - %(name)s - {warn_prefix} %(' \
                           f'message)s'
        self.info_format = f'%(asctime)s - %(name)s - {info_prefix} %(' \
                           f'message)s'
        self.debug_format = f'%(asctime)s - %(name)s - {debug_prefix} %(' \
                          f'message)s'

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
        Logger.__init__(self, name, log_level)
        BaseGlobalAccessible.__init__(self, name)
        # Get rank in DDP mode.
        # if dist.is_available() and dist.is_initialized():
        #     rank = dist.get_rank()
        # else:
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
                path_split[-1] = f'rank_{rank}_{path_split[-1]}'
                log_file = separator.join(path_split)
            # Here, the default behaviour of the official logger is 'a'. Thus,
            # we provide an interface to change the file mode to the default
            # behaviour. `FileHandler` is not supported to have colors,
            # otherwise it will appear garbled.
            file_handler = logging.FileHandler(log_file, file_mode)
            file_handler.setFormatter(MMFormatter(color=False))
            file_handler.setLevel(log_level)
            self.handlers.append(file_handler)