import numpy as np
from typing import Optional
from functools import partial
import copy


class BaseLogBuffer:
    _statistics_dict: dict = dict()

    def __init__(self, log_history=[], count_histroy=[], max_length=1000000):
        self.max_length = max_length
        assert len(log_history) == len(count_histroy), \
            'The lengths of log_history and count_histroy should be equal'
        if len(log_history) > max_length:
            self._log_history = np.array(log_history[-max_length:])
            self._count_history = np.array(count_histroy[-max_length:])
        else:
            self._log_history = np.array(log_history)
            self._count_history = np.array(count_histroy)

    def update(self, log_val, count=1):
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
    def data(self):
        return self._log_history, self._count_history

    @classmethod
    def register_statistics(cls, method=None):
        method_name = method.__name__
        assert method_name not in cls._statistics_dict, \
            f'method_name cannot be registered twice!'
        cls._statistics_dict[method_name] = method
        return method

    def statistics(self, name: str, *arg, **kwargs):
        if name not in self._statistics_dict:
            raise KeyError(f'{name} has not been registered in '
                           f'BaseLogBuffer._statistics_dict')
        method = self._statistics_dict[name]
        # Provide self arguments for registered functions.
        method = partial(method, self)
        return method(*arg, **kwargs)


class LogBuffer(BaseLogBuffer):
    @BaseLogBuffer.register_statistics
    def mean(self, window_size: Optional[int] = None):
        if not window_size:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].sum() / \
               self._count_history[-window_size:].sum()

    @BaseLogBuffer.register_statistics
    def max(self, window_size: Optional[int] = None):
        if not window_size:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].max()

    @BaseLogBuffer.register_statistics
    def min(self, window_size: Optional[int] = None):
        if not window_size:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].min()

    @BaseLogBuffer.register_statistics
    def current(self):
        if len(self._log_history) == 0:
            raise ValueError('LogBuffer._log_history is an empty array! '
                             'please call update first')
        return self._log_history[-1]




