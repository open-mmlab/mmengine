from collections import OrderedDict
from .base_global_accsessible import BaseGlobalAccessible
from .log_buffer import LogBuffer

from typing import Union


class MessageHub(BaseGlobalAccessible):
    def __init__(self, name=''):
        self._log_buffers = OrderedDict()
        self._runtime = OrderedDict()
        super().__init__(name)

    def update_log(self, key: str, value: Union[int, float], count=1):
        if key in self._log_buffers:
            self._log_buffers[key].update(value, count)
        else:
            self._log_buffers[key] = LogBuffer([value], [count])

    def update_info(self, key, value):
        self._runtime[key] = value

    @property
    def log_buffers(self):
        return self._log_buffers

    @property
    def runtime_info(self):
        return self._runtime

    def get_log(self, key):
        if key not in self.log_buffers:
            raise KeyError(f'{key} is not fount in Messagehub.log_buffers: '
                           f'instance name is: {MessageHub.instance_name}')
        return self._log_buffers[key]

    def get_info(self, key):
        if key not in self.runtime_info:
            raise KeyError(f'{key} is not fount in Messagehub.log_buffers: '
                           f'instance name is: {MessageHub.instance_name}')
        return self._runtime[key]



