from collections import OrderedDict
from typing import Callable
from .log_buffer import LogBuffer


class TaskSingleton:
    def __init__(self, method: Callable):
        self.method = method
        self.instances = dict()
        self.owner = None

    def get_instance(self,
                     name: str = '',
                     latest: bool = False,
                     *args, **
                     kwargs):
        if not self.instances.get('root', False):
            root = self.owner.__new__(self.owner, 'root', *args, **kwargs)
            self.instances['root'] = root
            self.latest = root

        if name:
            if name in self.instances:
                return self.instances[name]
            else:
                instance = self.method.__get__(self.name)(name, *args, **kwargs)
                self.instances[name] = instance
                self.latest = instance
                return instance
        else:
            if latest:
                return self.latest
            else:
                return self.instances['root']

    def __set_name__(self, owner, name):
        self.owner = owner
        self.name = name

    def __get__(self, instance, owner):
        return self.get_instance


class MessageHub(object):
    def __new__(cls, name: str):
        instance = super().__new__(cls)
        instance._log_buffers = OrderedDict()
        instance._caches = OrderedDict()
        instance.name = name
        return instance

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @TaskSingleton
    @staticmethod
    def get_message_hub(name='', latest=False):
        return MessageHub.__new__(MessageHub, name)

    def update_log(self, key, value, count):
        if not isinstance(value, (int, float)):
            raise TypeError('value must be ')
        if key in self._log_buffers:
            self._log_buffers[key].update(value, count)
        else:
            self._log_buffers[key] = LogBuffer([value], [count])

    def update_cache(self, key, value):
        self._caches[key] = value

    @property
    def log_buffers(self):
        return self._log_buffers

    @property
    def caches(self):
        return self._log_buffers

    def get_log_buffer(self, key):
        return self._log_buffers[key]

    def get_caches(self, key):
        return self.caches[key]



