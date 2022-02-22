import warnings

import numpy as np
from abc import ABCMeta, abstractmethod
from mmengine import LOG_BUFFER
from typing import Sequence, Iterable


class BaseLogBuffer(metaclass=ABCMeta):
    def __init__(self,
                 log_history: Sequence = [], count_histroy: Sequence = [],
                 max_length=1000000):
        self.max_length = max_length
        assert len(log_history) == len(count_histroy), \
            'The lengths of log_history and count_histroy should be equal'
        if len(log_history) > max_length:
            self._log_history = np.array(log_history[-max_length:])
            self._count_history = np.array(count_histroy[-max_length:])
        else:
            self._log_history = np.array(log_history)
            self._count_history = np.array(count_histroy)

    @abstractmethod
    def statistics(self, window_size=None):
        pass

    def update(self, log_val, count=1):
        if isinstance(log_val, Iterable) or isinstance(count, Iterable):
            raise TypeError('log_val and count should be single value but '
                            'got Iterable.')
        self._log_history = np.append(self._log_history, log_val)
        self._count_history = np.append(self._count_history, count)
        if len(self._log_history) > self.max_length:
            self._log_history = self._log_history[-self.max_length:]
            self._count_history = self._count_history[-self.max_length:]

    @property
    def data(self):
        return self._log_history, self._count_history


@LOG_BUFFER.register_module('mean')
class MeanLogBuffer(BaseLogBuffer):
    def statistics(self, window_size=None):
        if not window_size:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].sum() / \
               self._count_history[-window_size:].sum()


@LOG_BUFFER.register_module('max')
class MaxLogBuffer(BaseLogBuffer):
    def statistics(self, window_size=None):
        if not window_size:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].max()


@LOG_BUFFER.register_module('min')
class MinLogBuffer(BaseLogBuffer):
    def statistics(self, window_size=None):
        if not window_size:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].min()


@LOG_BUFFER.register_module('current')
class CurrentLogBuffer(BaseLogBuffer):
    def statistics(self, window_size=None):
        if len(self._log_history) > 0:
            return self._log_history[-1]
        else:
            warnings.warn('CurrentLogBuffer has not been update, you will'
                          'get an empty array')
        return self._log_history
