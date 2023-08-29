# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import re
import sys
from importlib.util import find_spec, spec_from_loader
from typing import Any, Optional


class LazyObject:
    """LazyObject is used to lazily initialize the imported module during
    parsing the configuration file.

    During parsing process, the syntax like:

    Examples:
        >>> import torch.nn as nn
        >>> from mmdet.models import RetinaNet
        >>> import mmcls.models

    Will be parsed as:

    Examples:
        >>> # import torch.nn as nn
        >>> nn = LazyObject('torch.nn')
        >>> # from mmdet.models import RetinaNet
        >>> RetinaNet = LazyObject('RetinaNet', LazyObject('mmdet.models'))
        >>> # import mmcls.models
        >>> mmcls = LazyObject('mmcls.models')

    ``LazyObject`` records all module information and will be further
    referenced by the configuration file.

    Args:
        name (str): The name of a module or attribution.
        source (LazyObject, optional): The source of the lazy object.
            Defaults to None.
    """

    def __init__(self, name: str, source: Optional['LazyObject'] = None):
        self.name = name
        self.source = source

    def build(self) -> Any:
        """Return imported object.

        Returns:
            Any: Imported object
        """
        if self.source is not None:
            module = self.source.build()
            try:
                return getattr(module, self.name)
            except AttributeError:
                raise ImportError(
                    f'Failed to import {self.name} from {self.source}')
        else:
            try:
                for idx in range(self.name.count('.') + 1):
                    module, *attrs = self.name.rsplit('.', idx)
                    try:
                        spec = find_spec(module)
                    except ImportError:
                        spec = None
                    if spec is not None:
                        res = importlib.import_module(module)
                        for attr in attrs:
                            res = getattr(res, attr)
                        return res
                raise ImportError(f'No module named `{module}`.')
            except (ImportError, AttributeError) as e:
                raise ImportError(f'Failed to import {self.name} for {e}')

    def __deepcopy__(self, memo):
        return LazyObject(self.name, self.source)

    def __getattr__(self, name):
        return LazyObject(name, self)

    def __str__(self) -> str:
        if self.source is not None:
            return str(self.source) + '.' + self.name
        return self.name

    def __repr__(self) -> str:
        arg = f'name={repr(self.name)}'
        if self.source is not None:
            arg += f', source={repr(self.source)}'
        return f'LazyObject({arg})'

    @property
    def dump_str(self) -> str:
        return f'<{str(self)}>'

    @classmethod
    def from_str(cls, string):
        match_ = re.match(r'<([\w\.]+)>', string)
        if match_ and '.' in match_.group(1):
            source, _, name = match_.group(1).rpartition('.')
            return cls(name, cls(source))
        elif match_:
            return cls(match_.group(1))
        return None

    # `pickle.dump` will try to get the `__getstate__` and `__setstate__`
    # methods of the dumped object. If these two methods are not defined,
    # LazyObject will return a `__getstate__` LazyObject` or `__setstate__`
    # LazyObject.
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


class LazyImportContext:

    def __init__(self, enable=True):
        self.enable = enable

    def find_spec(self, fullname, path=None, target=None):
        if not self.enable or 'mmengine.config' in fullname:
            # avoid lazy import mmengine functions
            return None
        spec = spec_from_loader(fullname, self)
        return spec

    def create_module(self, spec):
        self.lazy_modules.append(spec.name)
        return LazyObject(spec.name)

    @classmethod
    def exec_module(self, module):
        pass

    def __enter__(self):
        # insert after FrozenImporter
        index = sys.meta_path.index(importlib.machinery.FrozenImporter)
        sys.meta_path.insert(index + 1, self)
        self.lazy_modules = []

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.meta_path.remove(self)
        for name in self.lazy_modules:
            if '.' in name:
                parent_module, _, child_name = name.rpartition('.')
                if parent_module in sys.modules:
                    delattr(sys.modules[parent_module], child_name)
            sys.modules.pop(name, None)

    def __repr__(self):
        return f'<LazyImportContext (enable={self.enable})>'


def recover_lazy_field(cfg):

    if isinstance(cfg, dict):
        for k, v in cfg.items():
            cfg[k] = recover_lazy_field(v)
        return cfg
    elif isinstance(cfg, (tuple, list)):
        container_type = type(cfg)
        cfg = list(cfg)
        for i, v in enumerate(cfg):
            cfg[i] = recover_lazy_field(v)
        return container_type(cfg)
    elif isinstance(cfg, str):
        recover = LazyObject.from_str(cfg)
        return recover if recover is not None else cfg
    return cfg
