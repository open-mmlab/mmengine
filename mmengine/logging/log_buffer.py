# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np


class BaseLogBuffer:
    """Unified storage format for different log types.

    Record the history of log for further statistics. The subclass inherited
    from ``BaseLogBuffer`` will implement the specific statistical methods.

    Args:
        log_history (Sequence): The history logs. Defaults to [].
        count_history (Sequence): The counts of the history logs. Defaults to
            [].
        max_length (int): The max length of history logs. Defaults to 1000000.
    """
    _statistics_dict: dict = dict()

    def __init__(self,
                 log_history: Sequence = [],
                 count_history: Sequence = [],
                 max_length: int = 1000000):

        self.max_length = max_length
        assert len(log_history) == len(count_history), \
            'The lengths of log_history and count_histroy should be equal'
        if len(log_history) > max_length:
            warnings.warn(f'The length of history buffer({len(log_history)}) '
                          f'exceeds the max_length({max_length}), the first '
                          'few elements will be ignored.')
            self._log_history = np.array(log_history[-max_length:])
            self._count_history = np.array(count_history[-max_length:])
        else:
            self._log_history = np.array(log_history)
            self._count_history = np.array(count_history)

    def update(self, log_val: Union[int, float], count: int = 1) -> None:
        """update the log history. If the length of the buffer exceeds
        ``self._max_length``, the oldest element will be removed from the
        buffer.

        Args:
            log_val (int, float): The value of log.
            count (int): The accumulation times of log, defaults to 1. count
                will be used in smooth statistics.
        """
        if not isinstance(log_val, (int, float)) or \
                not isinstance(count, (int, float)):
            raise TypeError(f'log_val must be int or float but got '
                            f'{type(log_val)}, count must be int but got '
                            f'{type(count)}')
        self._log_history = np.append(self._log_history, log_val)
        self._count_history = np.append(self._count_history, count)
        if len(self._log_history) > self.max_length:
            self._log_history = self._log_history[-self.max_length:]
            self._count_history = self._count_history[-self.max_length:]

    @property
    def data(self) -> Tuple[Sequence, Sequence]:
        """Get the ``_log_history`` and ``_count_history``.

        Returns:
            Tuple[Sequence, Sequence]: The history logs and the counts of the
            history logs.
        """
        return self._log_history, self._count_history

    @classmethod
    def register_statistics(cls, method: Callable) -> Callable:
        """Register custom statistics method to ``_statistics_dict``.

        Args:
            method (Callable): custom statistics method.

        Returns:
            Callable: Original custom statistics method.
        """
        method_name = method.__name__
        assert method_name not in cls._statistics_dict, \
            'method_name cannot be registered twice!'
        cls._statistics_dict[method_name] = method
        return method

    def statistics(self, method_name: str, *arg, **kwargs) -> Any:
        """Access statistics method by name.

        Args:
            method_name (str): The name of method.

        Returns:
            Any: Depends on corresponding method.
        """
        if method_name not in self._statistics_dict:
            raise KeyError(f'{method_name} has not been registered in '
                           f'BaseLogBuffer._statistics_dict')
        method = self._statistics_dict[method_name]
        # Provide self arguments for registered functions.
        method = partial(method, self)
        return method(*arg, **kwargs)


class LogBuffer(BaseLogBuffer):
    """The subclass of ``BaseLogBuffer`` and provide basic statistics method,
    such as ``min``, ``max``, ``current`` and ``mean``."""

    @BaseLogBuffer.register_statistics
    def mean(self, window_size: Optional[int] = None) -> np.ndarray:
        """Return the mean of the latest ``window_size`` values in log
        histories. If ``window_size is None``, return the global mean of
        history logs.

        Args:
            window_size (int, optional): The size of statistics window.

        Returns:
            np.ndarray: The mean value within the window.
        """
        if window_size is not None:
            assert isinstance(window_size, int), \
                'The type of window size should be int, but got ' \
                f'{type(window_size)}'
        else:
            window_size = len(self._log_history)
        logs_sum = self._log_history[-window_size:].sum()
        counts_sum = self._count_history[-window_size:].sum()
        return logs_sum / counts_sum

    @BaseLogBuffer.register_statistics
    def max(self, window_size: Optional[int] = None) -> np.ndarray:
        """Return the maximum value of the latest ``window_size`` values in log
        histories. If ``window_size is None``, return the global maximum value
        of history logs.

        Args:
            window_size (int, optional): The size of statistics window.

        Returns:
            np.ndarray: The maximum value within the window.
        """
        if window_size is not None:
            assert isinstance(window_size, int), \
                'The type of window size should be int, but got ' \
                f'{type(window_size)}'
        else:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].max()

    @BaseLogBuffer.register_statistics
    def min(self, window_size: Optional[int] = None) -> np.ndarray:
        """Return the minimum value of the latest ``window_size`` values in log
        histories. If ``window_size is None``, return the global minimum value
        of history logs.

        Args:
            window_size (int, optional): The size of statistics window.

        Returns:
            np.ndarray: The minimum value within the window.
        """
        if window_size is not None:
            assert isinstance(window_size, int), \
                'The type of window size should be int, but got ' \
                f'{type(window_size)}'
        else:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].min()

    @BaseLogBuffer.register_statistics
    def current(self) -> np.ndarray:
        """Return the recently updated values in log histories.

        Returns:
            np.ndarray: The recently updated values in log histories.
        """
        if len(self._log_history) == 0:
            raise ValueError('LogBuffer._log_history is an empty array! '
                             'please call update first')
        return self._log_history[-1]
