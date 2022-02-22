import numpy as np


class MethodRegister:
    def __init__(self, fn):
        self.fn = fn

    def __set_name__(self, owner, name):
        # do something with owner, i.e.
        if not hasattr(owner, 'registered_method'):
            owner.registered_method = dict()
        else:
            assert isinstance(owner.registered_method, dict)
        owner.registered_method[name] = self.fn


class LogBuffer:
    registered_method: dict = dict()

    def __init__(self,
                 log_val: list = [], count: list = [],
                 max_length=1000000):
        self.max_length = max_length
        self._log_history = np.array(log_val)
        self._count_history = np.array(count)

    @MethodRegister
    def update(self, log_val, count):
        self._log_history = np.append(self._log_history, log_val)
        self._count_history = np.append(self._count_history, count)
        if len(self._log_history) > self.max_length:
            self._log_history = self._log_history[-self.max_length:]
            self._count_history = self._count_history[-self.max_length:]

    @MethodRegister
    def moving_mean(self, window_size=None):
        if not window_size:
            window_size = len(self._log_history)
        return np.sum(self._log_history[-window_size:] / self._count_history[
                                                         -window_size:])

    @MethodRegister
    def max(self, window_size=None):
        if not window_size:
            window_size = len(self._log_history)
        idx = self._log_history[-window_size:].argmax()
        return self._log_history[idx], self._count_history[idx]

    @MethodRegister
    def min(self, window_size=None):
        if not window_size:
            window_size = len(self._log_history)
        idx = self._log_history[-window_size:].argmin()
        return self._log_history[idx], self._count_history[idx]

    @MethodRegister
    def latest(self, window_size=None):
        if len(self._log_history) > 0:
            return self._log_history[-1]
        else:
            return self._log_history

    def data(self):
        return self._log_history, self._count_history

    def statistics(self, name, *args, **kwargs):
        return self.registered_method[name](self, *args, **kwargs)

