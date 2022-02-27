import numpy as np
from abc import ABC


class BaseLogBuffer(ABC):
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

    def update(self, log_val, count):
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
        cls._statistics_dict[method.__name__] = method
        return cls


class LogBuffer(BaseLogBuffer):
    @BaseLogBuffer.register_statistics
    def mean(self, window_size=None):
        if not window_size:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].sum() / \
               self._count_history[-window_size:].sum()

    @BaseLogBuffer.register_statistics
    def max(self, window_size=None):
        if not window_size:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].max()

    @BaseLogBuffer.register_statistics
    def min(self, window_size=None):
        if not window_size:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].min()

    @BaseLogBuffer.register_statistics
    def current(self):
        return self._log_history[-1]