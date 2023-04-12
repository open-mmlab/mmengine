# Copyright (c) OpenMMLab. All rights reserved.
import importlib
from typing import Any, Optional, Union

from mmengine.utils import is_seq_of


class LazyObject:
    """LazyObject is used to lazy initialize the imported module during parsing
    the configuration file.

    During parsing process, the syntax like:

    Examples:
        >>> import torch.nn as nn
        >>> from mmdet.models import RetinaNet
        >>> import mmcls.models
        >>> import mmcls.datasets
        >>> import mmcls

    Will be parsed as:

    Examples:
        >>> # import torch.nn as nn
        >>> nn = lazyObject('torch.nn')
        >>> # from mmdet.models import RetinaNet
        >>> RetinaNet = lazyObject(['mmdet.models', 'RetinaNet'])
        >>> # import mmcls.models; import mmcls.datasets; import mmcls
        >>> mmcls = lazyObject(['mmcls', 'mmcls.datasets', 'mmcls.models'])

    ``LazyObject``s record all module information and will be further
    referenced by configuration file.

    Args:
        module (str or list or tuple): The module name to be imported.
        imported (str, optional): The imported module name. Defaults to None.
    """

    def __init__(self,
                 module: Union[str, list, tuple],
                 imported: Optional[str] = None):
        if (not isinstance(module, str) and not is_seq_of(module, str)):
            raise TypeError('module should be `str`, `list`, `tuple` or '
                            f'None, but got {type(imported)}, this might be '
                            'a bug of MMEngine, please report it to '
                            'https://github.com/open-mmlab/mmengine/issues')
        self._module: Union[str, list, tuple] = module

        if not isinstance(imported, str) and imported is not None:
            raise TypeError('imported should be `str` or None, but got '
                            f'{type(imported)}, this might be '
                            'a bug of MMEngine, please report it to '
                            'https://github.com/open-mmlab/mmengine/issues')
        self._imported = imported

    def build(self) -> Any:
        """Return imported object.

        Returns:
            Any: Imported object
        """
        if isinstance(self._module, str):
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
                importlib.import_module(module)  # type: ignore
            module_name = self._module[0].split('.')[0]
            return importlib.import_module(module_name)

    def __call__(self, *args, **kwargs):
        raise RuntimeError()

    def __deepcopy__(self, memo):
        return LazyObject(self._module, self._imported)

    def __getattr__(self, name):
        return LazyAttr(name, self)

    def __str__(self) -> str:
        if isinstance(self._imported, str):
            return str(self._imported)
        else:
            return self._module[0].split('.')[0]

    __repr__ = __str__


class LazyAttr:
    """The attribute of the LazyObject.

    When parsing the configuration file, the imported syntax will be
    parsed as the assignment ``LazyObject``. During the subsequent parsing
    process, users may reference the attributes of the LazyObject.
    To ensure that these attributes also contain information needed to
    reconstruct the attribute itself, LazyAttr was introduced.

    Examples:
        >>> models = LazyObject(['mmdet.models'])
        >>> model = dict(type=models.RetinaNet)
        >>> print(type(model['type']))  # <class 'mmengine.config.lazy.LazyAttr'>
        >>> print(model['type'].build())  # <class 'mmdet.models.detectors.retinanet.RetinaNet'>
    """  # noqa: E501

    def __init__(self, name: str, source: Union['LazyObject', 'LazyAttr']):
        self.name = name
        self.source = source
        # What is self._module meaning:
        # import a.b as b
        # b.c -> self._module of c is `a.b`
        # b.c.d -> self._module of d = `a.b.c`
        if isinstance(self.source, LazyObject):
            if isinstance(self.source._module, str):
                # In this case, the source code of LazyObject could be one of
                # the following:
                # 1. import xxx.yyy as zzz
                # 2. from xxx.yyy import zzz

                # The equivalent code of LazyObject is:
                # zzz = LazyObject('')

                # The source code of LazyAttr will be:
                # eee = zzz.eee
                # Then, eee._module = xxx.yyy
                self._module = self.source._module
            else:
                # The source code of LazyObject should be
                # 1. import xxx.yyy
                # 2. import xxx.zzz
                # Equivalent to
                # xxx = LazyObject(['xxx.yyy', 'xxx.zzz'])

                # The source code of LazyAttr should be
                # eee = xxx.eee
                # Then, eee._module = xxx
                self._module = str(self.source)
        elif isinstance(self.source, LazyAttr):
            # 1. import xxx
            # 2. zzz = xxx.yyy.zzz

            # Equivalent to:
            # xxx = LazyObject('xxx')
            # zzz = xxx.yyy.zzz
            # zzz._module = xxx.yyy._module + zzz.name
            self._module = self.source._module + self.source.name

    def __call__(self, *args, **kwargs: Any) -> Any:
        raise RuntimeError()

    def __getattr__(self, name: str) -> 'LazyAttr':
        return LazyAttr(name, self)

    def __deepcopy__(self, memo):
        return LazyAttr(self.name, self.source)

    def build(self) -> Any:
        """Return the attribute of the imported object.

        Returns:
            Any: attribute of the imported object.
        """
        return getattr(self.source.build(), self.name)

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__
