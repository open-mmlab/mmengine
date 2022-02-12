# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import sys
from collections.abc import Callable
from typing import Dict, List, Optional, Tuple, Type, Union

from mmengine.utils import is_seq_of


def build_from_cfg(cfg: dict,
                   registry: 'Registry',
                   default_args: Optional[dict] = None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')
    if not isinstance(registry, Registry):
        raise TypeError('registry must be a mmengine.Registry object, '
                        f'but got {type(registry)}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}')

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')
    try:
        return obj_cls(**args)  # type: ignore
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{obj_cls.__name__}: {e}')  # type: ignore


class Registry:
    """A registry to map strings to classes.

    Registered object could be built from registry.

    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODELS.build(dict(type='ResNet'))

    Please refer to
    https://mmengine.readthedocs.io/en/latest/tutorials/registry.html for
    advanced usage.

    Args:
        name (str): Registry name.
        build_func(callable, optional): Build function to construct instance
            from Registry, func:`build_from_cfg` is used if neither ``parent``
            or ``build_func`` is specified. If ``parent`` is specified and
            ``build_func`` is not given,  ``build_func`` will be inherited
            from ``parent``. Defaults to None.
        parent (Registry, optional): Parent registry. The class registered in
            children registry could be built from parent. Default: None.
        scope (str, optional): The scope of registry. It is the key to search
            for children registry. If not specified, scope will be the name of
            the package where class is defined, e.g. mmdet, mmcls, mmseg.
            Defaults to None.
    """

    def __init__(self,
                 name: str,
                 build_func: Optional[Callable] = None,
                 parent: Optional['Registry'] = None,
                 scope: Optional[str] = None):
        self._name = name
        self._module_dict: Dict[str, Type] = dict()
        self._children: Dict[str, 'Registry'] = dict()

        if scope:
            assert isinstance(scope, str)
            self._scope = scope
        else:
            self._scope = self.infer_scope() if scope is None else scope

        # self.build_func will be set with the following priority:
        # 1. build_func
        # 2. parent.build_func
        # 3. build_from_cfg
        if build_func is None:
            if parent is not None:
                assert isinstance(parent, Registry)
                self.build_func = parent.build_func  # type: ignore
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func

        # See https://mypy.readthedocs.io/en/stable/common_issues.html#
        # variables-vs-type-aliases for the use
        self.parent: Optional['Registry']
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_children(self)
            self.parent = parent
        else:
            self.parent = None

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(name={self._name}, ' \
                     f'items={self._module_dict})'
        return format_str

    @staticmethod
    def infer_scope() -> str:
        """Infer the scope of registry.

        The name of the package where registry is defined will be returned.

        Example:
            >>> # in mmdet/models/backbone/resnet.py
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            The scope of ``ResNet`` will be ``mmdet``.

        Returns:
            str: The inferred scope name.
        """
        # `sys._getframe` returns the frame object that many calls below the
        # top of the stack. The call stack for `infer_scope` can be listed as
        # follow:
        # frame-0: `infer_scope` itself
        # frame-1: `__init__` of `Registry` which calls the `infer_scope`
        # frame-2: Where the `Registry(...)` is called
        filename = inspect.getmodule(sys._getframe(2)).__name__  # type: ignore
        split_filename = filename.split('.')
        return split_filename[0]

    @staticmethod
    def split_scope_key(key: str) -> Tuple[Optional[str], str]:
        """Split scope and key.

        The first scope will be split from key.

        Examples:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'

        Return:
            tuple[str | None, str]: The former element is the first scope of
            the key, which can be ``None``. The latter is the remaining key.
        """
        split_index = key.find('.')
        if split_index != -1:
            return key[:split_index], key[split_index + 1:]
        else:
            return None, key

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def children(self):
        return self._children

    def _goto_root(self) -> 'Registry':
        """Return the root registry."""
        parent = self
        while parent.parent is not None:
            parent = parent.parent
        return parent

    def get(self, key: str) -> Optional[Type]:
        """Get the registry record.

        If the ``key`` contains a scope, it first get the key in the current
        registry. If failed, it will try to find the key in the whole registry
        tree.
        If the ``key`` does not contain a scope, it will find the key from
        the current registry to its parent or ancestors until finding the key.

        Args:
            key (str): The class name in string format.

        Returns:
            Type or None: Return the corresponding class if key exists else
            return None.
        """
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope:
            # get from self
            if real_key in self._module_dict:
                return self._module_dict[real_key]

            if scope is None:
                # New Feature: unit tests
                # try to get from its parent
                parent = self.parent
                while parent is not None:
                    if real_key in parent._module_dict:
                        return parent._module_dict[real_key]
                    parent = parent.parent
        else:
            # get from self._children
            if scope in self._children:
                return self._children[scope].get(real_key)
            else:
                # jump to the root registry
                root = self._goto_root()
                return root.get(key)

        return None

    def _find_registry(self, scope: str) -> Optional['Registry']:
        """Depth-first find for the corresponding registry.

        Note that the method only find the corresponding registry from the
        current registry. Therefore, if we want to find from the root registry,
        we need to call :meth:`_goto_root` first.

        Args:
            scope (str): The ``scope`` to find the corresponding registry.

        Returns:
            Registry or None: Return the corresponding registry if the
            ``scope`` exists else None.
        """
        if self._scope == scope:
            return self

        for child in self._children.values():
            registry = child._find_registry(scope)
            if registry is not None:
                return registry

        return None

    def build(self,
              *args,
              default_scope: Optional[str] = None,
              **kwargs) -> None:
        """Build an instance by calling the :attr:`build_func`.

        Args:
            default_scope (str, optional): The ``default_scope`` is used to
                reset the current registry. Defaults to None.
        """
        if default_scope:
            root = self._goto_root()
            registry = root._find_registry(default_scope)
            if registry is None:
                raise KeyError(
                    f'{default_scope} does not exist in the registry tree.')
        else:
            registry = self

        return registry.build_func(*args, **kwargs, registry=registry)

    def _add_children(self, registry: 'Registry') -> None:
        """Add children for a registry.

        The ``registry`` will be added as children based on its scope.
        The parent registry could build objects from children registry.

        Example:
            >>> models = Registry('models')
            >>> mmdet_models = Registry('models', parent=models)
            >>> @mmdet_models.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet = models.build(dict(type='mmdet.ResNet'))
        """

        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert registry.scope not in self.children, \
            f'scope {registry.scope} exists in {self.name} registry'
        self.children[registry.scope] = registry

    def _register_module(self,
                         module_class: Type,
                         module_name: Optional[Union[str, List[str]]] = None,
                         force: bool = False) -> None:
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, '
                            f'but got {type(module_class)}')

        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f'{name} is already registered '
                               f'in {self.name}')
            self._module_dict[name] = module_class

    def register_module(
            self,
            name: Optional[Union[str, List[str]]] = None,
            force: bool = False,
            module: Optional[Type] = None) -> Union[type, Callable]:
        """Register a module.

        A record will be added to ``self._module_dict``, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(module=ResNet)

        Args:
            name (str or list of str, optional): The module name to be
                registered. If not specified, the class name will be used.
            force (bool): Whether to override an existing class with the same
                name. Default to False.
            module (type, optional): Module class to be registered. Defaults to
                None.
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                'name must be either of None, an instance of str or a sequence'
                f' of str, but got {type(name)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(
                module_class=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(
                module_class=cls, module_name=name, force=force)
            return cls

        return _register
