from collections import OrderedDict
from mmengine import LOG_BUFFER


class TaskSingleton:
    def __init__(self, method: classmethod):
        self.method = method
        self.instances = dict()
        self.owner = None

    def get_instance(self,
                     name: str = '',
                     current: bool = False,
                     *args,
                     **kwargs):
        if not self.instances.get('root', False):
            root = self.owner('root', *args, **kwargs)
            self.instances['root'] = root
            self.current = root

        if name:
            if name in self.instances:
                return self.instances[name]
            else:
                instance = self.method.__get__(None, self.owner)(name, *args, **kwargs)
                self.instances[name] = instance
                self.current = instance
                return instance
        else:
            if current:
                return self.current
            else:
                return self.instances['root']

    def __set_name__(self, owner, name):
        self.owner = owner
        self.name = name

    def __get__(self, instance, owner):
        return self.get_instance


class MessageHub(object):
    def __init__(self, name: str):
        self._log_buffers = OrderedDict()
        self._runtime = OrderedDict()
        self.name = name

    @TaskSingleton
    @classmethod
    def get_message_hub(cls, name='', current=False):
        return cls(name)

    def update_log(self, key, value, count=1, log_type='current'):
        if log_type not in LOG_BUFFER.module_dict:
            raise KeyError(f'{log_type} is not registered in LOG_BUFFER!'
                           'please register it and import your custom module')
        if key in self._log_buffers:
            self._log_buffers[key].update(value, count)
        else:
            self._log_buffers[key] = LOG_BUFFER.get(log_type)([value], [count])

    def update_runtime(self, key, value):
        self._runtime[key] = value

    @property
    def log_buffers(self):
        return self._log_buffers

    @property
    def runtime(self):
        return self._runtime

    def get_log_buffer(self, key):
        return self._log_buffers[key]

    def get_runtime(self, key):
        return self._runtime[key]



