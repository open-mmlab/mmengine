# Copyright (c) OpenMMLab. All rights reserved.
import importlib
from typing import Any, Union


class LazyModule:

    def __init__(self, module, imported=None):
        self._module = module
        if isinstance(module, (list, tuple)):
            _module = set(module)
            assert len(module) == len(_module), 'Duplicate module name'
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
        raise RuntimeError()

    def __deepcopy__(self, memo):
        return LazyModule(self._module, self._imported)

    def __getattr__(self, name):
        return LazyAttr(name, self)

    def __str__(self) -> str:
        if isinstance(self._imported, str):
            return str(self._imported)
        else:
            return self._module[0].split('.')[0]

    __repr__ = __str__


class LazyAttr:

    def __init__(self, name: str, source: Union['LazyModule',
                                                'LazyAttr']) -> None:
        self.name = name
        self.source = source
        # import a.b as b
        # b.c -> _module of c is `a.b`
        # b.c.d -> _module of d = `a.b.c`
        if isinstance(self.source, LazyModule):
            self._module = self.source._module
        elif isinstance(self.source, LazyAttr):
            self._module = self.source._module + self.source.name

    def __call__(self, *args, **kwargs: Any) -> Any:
        raise RuntimeError()

    def __getattr__(self, name: str) -> 'LazyAttr':
        return LazyAttr(name, self)

    def __deepcopy__(self, memo):
        return LazyAttr(self.name, self.source)

    def build(self):
        return getattr(self.source.build(), self.name)

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__
