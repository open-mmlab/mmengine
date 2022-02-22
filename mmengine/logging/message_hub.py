from collections import OrderedDict


class TaskSingleton:
    def __init__(self, method):
        self.method = method
        self.instances = dict()
        self.owner = None

    def get_instance(self, name='', latest=False, *args, **kwargs):
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
    def __new__(cls, name):
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

if __name__ == '__main__':
    x = MessageHub.get_message_hub('task1')
    print(x.name)