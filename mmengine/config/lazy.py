import copy
import importlib
from typing import Any


class Placeholder:
    ...


class LazyModule:

    def __init__(self, module, imported=None):
        self._module = module
        self._imported = imported

    def build(self):
        if not isinstance(self._module, list):
            # For import xxx.xxx as xxx or from xxx.xxx import xxx
            module = importlib.import_module(self._module)
            if self._imported:
                module = getattr(module, self._imported)
            return module
        else:
            # import xxx.xxx
            # import xxx.yyy
            # import xxx.zzz
            # return imported xxx
            for module in self._module:
                module = importlib.import_module(module)
            module_name = self._module[0].split('.')[0]
            return importlib.import_module(module_name)

    def __call__(self, *args, **kwargs):
        if self._imported:
            type = self._imported
        else:
            assert len(self._module) == 0
            type = self._module[0]
        return LazyCall(type, self, None, *args, **kwargs)

    def __deepcopy__(self, memo):
        return LazyModule(self._module, self._imported)

    def __getattr__(self, name):
        return LazyAttr(name, self)


class LazyAttr:

    def __init__(self, name, source) -> None:
        self.name = name
        self.source = source

    def __call__(self, *args, **kwargs: Any) -> Any:
        return LazyCall(self.name, self, None, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return LazyAttr(name, self)

    def __deepcopy__(self, memo):
        return LazyAttr(self.name, self.source)

    def build(self):
        return getattr(self.source.build(), self.name)


class LazyCall:

    def __init__(self,
                 type,
                 source,
                 instance_id=None,
                 *args,
                 **kwargs) -> None:
        super().__setattr__('type', type)
        super().__setattr__('kwargs', kwargs)
        super().__setattr__('source', source)
        super().__setattr__('args', args)
        instance_id = id(self) if instance_id is None else instance_id
        super().__setattr__('instance_id', instance_id)
        # self.args = args

    def build(self, memo=None):
        if memo is None:
            memo = dict()
        # built is used for duplicated built.
        def _build_lazy_call(kwargs, global_built):
            if isinstance(kwargs, dict):
                return type(kwargs)({
                    key: _build_lazy_call(value, global_built)
                    for key, value in kwargs.items()
                })
                # for key, value in kwargs.items():
                #     kwargs[key] = _build_lazy_call(value, global_built)
                # return kwargs
            elif isinstance(kwargs, (list, tuple)):
                return type(kwargs)([
                    _build_lazy_call(value, global_built) for value in kwargs
                ])
                # for i, value in enumerate(kwargs):
                #     kwargs[i] = _build_lazy_call(kwargs[i], global_built)
                # # return kwargs
            elif isinstance(kwargs, LazyCall):
                if kwargs.instance_id not in global_built:
                    ret = kwargs.build(memo=global_built)
                    global_built[kwargs.instance_id] = ret
                    return ret
                else:
                    return global_built[kwargs.instance_id]
            elif isinstance(kwargs, (LazyAttr, LazyModule)):
                return kwargs.build()
            else:
                return kwargs

        kwargs = _build_lazy_call(copy.deepcopy(self.kwargs), memo)
        args = _build_lazy_call(copy.deepcopy(self.args), memo)

        return self._build(*args, **kwargs)

    def _build(self, *args, **kwargs):
        for value in (self.args, self.kwargs.values):
            if value == '???' or isinstance(value, Placeholder):
                assert not self.args
                return self
        return self.source.build()(*args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'kwargs' or name == 'type' or name == 'instance_id' or name == 'args':
            super().__setattr__(name, value)
            return
        assert not self.args, (
            f'If you want to set attribute, please build {self.type} with '
            'keyword args, but not positional args.')
        self.kwargs[name] = value
        super().__setattr__(name, value)

    def setdefault(self, name, value):
        if name not in self.kwargs:
            self.kwargs[name] = value

    def __getattr__(self, name: str) -> Any:
        if name in self.kwargs:
            return self.kwargs[name]
        return LazyAttr(name, self)
        # try:
        #     return super().__getattr__(name)
        # except:
        #     builder = partial(LazyCall, name)
        #     return builder

    def __contains__(self, key):
        return key in self.kwargs

    def __deepcopy__(self, memo):
        # try:
        cls = self.__class__
        ret = cls(
            copy.deepcopy(self.type), self.source, self.instance_id,
            *copy.deepcopy(self.args), **copy.deepcopy(self.kwargs))

        return ret

    def set_build_variable(self, global_dict, module_dict):
        super().__setattr__('global_dict', global_dict)
        super().__setattr__('module_dict', module_dict)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return LazyCall(self.type, self, None, *args, **kwds)
