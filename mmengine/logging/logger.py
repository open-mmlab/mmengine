# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import sys
from logging import Logger, LogRecord
from typing import Optional, Union

from termcolor import colored

from mmengine import dist
from mmengine.utils import ManagerMixin


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
        self.err_format = (f'%(asctime)s - %(name)s - {error_prefix} - '
                           '%(pathname)s - %(funcName)s - %(lineno)d - '
                           '%(message)s')
        self.warn_format = (f'%(asctime)s - %(name)s - {warn_prefix} - %('
                            'message)s')
        self.info_format = (f'%(asctime)s - %(name)s - {info_prefix} - %('
                            'message)s')
        self.debug_format = (f'%(asctime)s - %(name)s - {debug_prefix} - %('
                             'message)s')

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


class MMLogger(Logger, ManagerMixin):
    """Formatted logger used to record messages.

    ``MMLogger`` can create formatted logger to log message with different
    log levels and get instance in the same way as ``ManagerMixin``.
    ``MMLogger`` has the following features:

    - Distributed log storage, ``MMLogger`` can choose whether to save log of
      different ranks according to `log_file`.
    - Message with different log levels will have different colors and format
      when displayed on terminal.

    Note:
        - The `name` of logger and the ``instance_name`` of ``MMLogger`` could
          be different. We can only get ``MMLogger`` instance by
          ``MMLogger.get_instance`` but not ``logging.getLogger``. This feature
          ensures ``MMLogger`` will not be incluenced by third-party logging
          config.
        - Different from ``logging.Logger``, ``MMLogger`` will not log warrning
          or error message without ``Handler``.

    Examples:
        >>> logger = MMLogger.get_instance(name='MMLogger',
        >>>                                logger_name='Logger')
        >>> # Although logger has name attribute just like `logging.Logger`
        >>> # We cannot get logger instance by `logging.getLogger`.
        >>> assert logger.name == 'Logger'
        >>> assert logger.instance_name = 'MMLogger'
        >>> assert id(logger) != id(logging.getLogger('Logger'))
        >>> # Get logger that do not store logs.
        >>> logger1 = MMLogger.get_instance('logger1')
        >>> # Get logger only save rank0 logs.
        >>> logger2 = MMLogger.get_instance('logger2', log_file='out.log')
        >>> # Get logger only save multiple ranks logs.
        >>> logger3 = MMLogger.get_instance('logger3', log_file='out.log',
        >>>                                 distributed=True)

    Args:
        name (str): Global instance name.
        logger_name (str): ``name`` attribute of ``Logging.Logger`` instance.
            If `logger_name` is not defined, defaults to 'mmengine'.
        log_file (str, optional): The log filename. If specified, a
            ``FileHandler`` will be added to the logger. Defaults to None.
        log_level (str): The log level of the handler and logger. Defaults to
            "NOTSET".
        file_mode (str): The file mode used to open log file. Defaults to 'w'.
        distributed (bool): Whether to save distributed logs, Defaults to
            false.
    """

    def __init__(self,
                 name: str,
                 logger_name='mmengine',
                 log_file: Optional[str] = None,
                 log_level: str = 'INFO',
                 file_mode: str = 'w',
                 distributed=False):
        Logger.__init__(self, logger_name)
        ManagerMixin.__init__(self, name)
        # Get rank in DDP mode.
        rank = dist.get_rank()

        # Config stream_handler. If `rank != 0`. stream_handler can only
        # export ERROR logs.
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        # `StreamHandler` record month, day, hour, minute, and second
        # timestamp.
        stream_handler.setFormatter(
            MMFormatter(color=True, datefmt='%m/%d %H:%M:%S'))
        # Only rank0 `StreamHandler` will log messages below error level.
        stream_handler.setLevel(log_level) if rank == 0 else \
            stream_handler.setLevel(logging.ERROR)
        self.handlers.append(stream_handler)

        if log_file is not None:
            if rank != 0:
                # rename `log_file` with rank suffix.
                path_split = log_file.split(os.sep)
                if '.' in path_split[-1]:
                    filename_list = path_split[-1].split('.')
                    filename_list[-2] = f'{filename_list[-2]}_rank{rank}'
                    path_split[-1] = '.'.join(filename_list)
                else:
                    path_split[-1] = f'{path_split[-1]}_rank{rank}'
                log_file = os.sep.join(path_split)
            # Save multi-ranks logs if distributed is True. The logs of rank0
            # will always be saved.
            if rank == 0 or distributed:
                # Here, the default behaviour of the official logger is 'a'.
                # Thus, we provide an interface to change the file mode to
                # the default behaviour. `FileHandler` is not supported to
                # have colors, otherwise it will appear garbled.
                file_handler = logging.FileHandler(log_file, file_mode)
                # `StreamHandler` record year, month, day hour, minute,
                # and second timestamp. file_handler will only record logs
                # without color to avoid garbled code saved in files.
                file_handler.setFormatter(
                    MMFormatter(color=False, datefmt='%Y/%m/%d %H:%M:%S'))
                file_handler.setLevel(log_level)
                self.handlers.append(file_handler)

    @classmethod
    def get_current_instance(cls) -> 'MMLogger':
        """Get latest created ``MMLogger`` instance.

        :obj:`MMLogger` can call :meth:`get_current_instance` before any
        instance has been created, and return a logger with the instance name
        "mmengine".

        Returns:
            MMLogger: Configured logger instance.
        """
        if not cls._instance_dict:
            cls.get_instance('mmengine')
        return super(MMLogger, cls).get_current_instance()

    def callHandlers(self, record: LogRecord) -> None:
        """Pass a record to all relevant handlers.

        Override ``callHandlers`` method in ``logging.Logger`` to avoid
        multiple warning messages in DDP mode. Loop through all handlers of
        the logger instance and its parents in the logger hierarchy. If no
        handler was found, the record will not be output.

        Args:
            record (LogRecord): A ``LogRecord`` instance contains logged
                message.
        """
        for handler in self.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)


def print_log(msg,
              logger: Optional[Union[Logger, str]] = None,
              level=logging.INFO) -> None:
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (Logger or str, optional): If the type of logger is
        ``logging.Logger``, we directly use logger to log messages.
            Some special loggers are:
            - "silent": No message will be printed.
            - "current": Use latest created logger to log message.
            - other str: Instance name of logger. The corresponding logger
              will log message if it has been created, otherwise ``print_log``
              will raise a `ValueError`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object, "current", or a created logger instance name.
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif logger == 'current':
        logger_instance = MMLogger.get_current_instance()
        logger_instance.log(level, msg)
    elif isinstance(logger, str):
        # If the type of `logger` is `str`, but not with value of `current` or
        # `silent`, we assume it indicates the name of the logger. If the
        # corresponding logger has not been created, `print_log` will raise
        # a `ValueError`.
        if MMLogger.check_instance_created(logger):
            logger_instance = MMLogger.get_instance(logger)
            logger_instance.log(level, msg)
        else:
            raise ValueError(f'MMLogger: {logger} has not been created!')
    else:
        raise TypeError(
            '`logger` should be either a logging.Logger object, str, '
            f'"silent", "current" or None, but got {type(logger)}')
