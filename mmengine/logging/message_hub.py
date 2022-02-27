from collections import OrderedDict
from .base_global_accsessible import BaseGlobalAccessible
from .log_buffer import LogBuffer


class MessageHub(BaseGlobalAccessible):
    def __init__(self, name=''):
        self._log_buffers = OrderedDict()
        self._runtime = OrderedDict()
        super().__init__(name)

    def update_log(self, key, value, count=1):
        if key in self._log_buffers:
            self._log_buffers[key].update(value, count)
        else:
            self._log_buffers[key] = LogBuffer([value], [count])

    def update_runtime(self, key, value):
        self._runtime[key] = value

    @property
    def log_buffers(self):
        return self._log_buffers

    @property
    def runtime(self):
        return self._runtime

    def get_log(self, key):
        return self._log_buffers[key]

    def get_info(self, key):
        return self._runtime[key]



