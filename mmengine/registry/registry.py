# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import sys
import warnings
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from ..config import Config, ConfigDict
from ..utils import ManagerMixin, is_seq_of
from .default_scope import DefaultScope


def build_runner_from_cfg(cfg: Union[dict, ConfigDict, Config],
                          registry: 'Registry') -> Any:
    """Build a Runner object.
    Examples:
        >>> from mmengine import Registry, build_runner_from_cfg
        >>> RUNNERS = Registry('runners', build_func=build_runner_from_cfg)
        >>> @RUNNERS.register_module()
        >>> class CustomRunner(Runner):
        >>>     def setup_env(env_cfg):
        >>>         pass
        >>> cfg = dict(runner_type='CustomRunner', ...)
        >>> custom_runner = RUNNERS.build(cfg)

    Args:
        cfg (dict or ConfigDict or Config): Config dict. If "runner_type" key
            exists, it will be used to build a custom runner. Otherwise, it
            will be used to build a default runner.
        registry (:obj:`Registry`): The registry to search the type from.

    Returns:
        object: The constructed runner object.
    """
    from ..logging.logger import MMLogger

    assert isinstance(
        cfg,
        (dict, ConfigDict, Config
         )), f'cfg should be a dict, ConfigDict or Config, but got {type(cfg)}'
    assert isinstance(
        registry, Registry), ('registry should be a mmengine.Registry object',
                              f'but got {type(registry)}')

    args = cfg.copy()
    obj_type = args.pop('runner_type', 'mmengine.Runner')
    if isinstance(obj_type, str):
        runner_cls = registry.get(obj_type)
        if runner_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry. '
                f'Please check whether the value of `{obj_type}` is correct or'
                ' it was registered as expected. More details can be found at'
                ' https://mmengine.readthedocs.io/en/latest/tutorials/config.html#import-custom-python-modules'  # noqa: E501
            )
    elif inspect.isclass(obj_type):
        runner_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')

    try:
        runner = runner_cls.from_cfg(args)  # type: ignore
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info(
            f'An `{runner_cls.__name__}` instance is built '  # type: ignore
            f'from registry, its implementation can be found in'
            f'{runner_cls.__module__}')  # type: ignore
        return runner

    except Exception as e:
        # Normal TypeError does not print class name.
        cls_location = '/'.join(
            runner_cls.__module__.split('.'))  # type: ignore
        raise type(e)(
            f'class `{runner_cls.__name__}` in '  # type: ignore
            f'{cls_location}.py: {e}')


def build_from_cfg(
        cfg: Union[dict, ConfigDict, Config],
        registry: 'Registry',
        default_args: Optional[Union[dict, ConfigDict, Config]] = None) -> Any:
    """Build a module from config dict.

    At least one of the ``cfg`` and ``default_args`` contains the key "type"
    which type should be either str or class. If they all contain it, the key
    in ``cfg`` will be used because ``cfg`` has a high priority than
    ``default_args`` that means if a key exist in both of them, the value of
    the key will be ``cfg[key]``. They will be merged first and the key "type"
    will be popped up and the remaining keys will be used as initialization
    arguments.

    Examples:
        >>> from mmengine import Registry, build_from_cfg
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     def __init__(self, depth, stages=4):
        >>>         self.depth = depth
        >>>         self.stages = stages
        >>> cfg = dict(type='ResNet', depth=50)
        >>> model = build_from_cfg(cfg, MODELS)

    Args:
        cfg (dict or ConfigDict or Config): Config dict. It should at least
            contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict or ConfigDict or Config, optional): Default
            initialization arguments. Defaults to None.

    Returns:
        object: The constructed object.
    """
    # Avoid circular import
    from ..logging.logger import MMLogger

    if not isinstance(cfg, (dict, ConfigDict, Config)):
        raise TypeError(
            f'cfg should be a dict, ConfigDict or Config, but got {type(cfg)}')

    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')

    if not isinstance(registry, Registry):
        raise TypeError('registry must be a mmengine.Registry object, '
                        f'but got {type(registry)}')

    if not (isinstance(default_args,
                       (dict, ConfigDict, Config)) or default_args is None):
        raise TypeError(
            'default_args should be a dict, ConfigDict, Config or None, '
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
                f'{obj_type} is not in the {registry.name} registry. '
                f'Please check whether the value of `{obj_type}` is correct or'
                ' it was registered as expected. More details can be found at'
                ' https://mmengine.readthedocs.io/en/latest/tutorials/config.html#import-custom-python-modules'  # noqa: E501
            )
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')

    try:
        # If `obj_cls` inherits from `ManagerMixin`, it should be instantiated
        # by `ManagerMixin.get_instance` to ensure that it can be accessed
        # globally.
        if issubclass(obj_cls, ManagerMixin):  # type: ignore
            obj = obj_cls.get_instance(**args)  # type: ignore
        else:
            obj = obj_cls(**args)  # type: ignore

        logger: MMLogger = MMLogger.get_current_instance()
        logger.info(
            f'An `{obj_cls.__name__}` instance is built from '  # type: ignore
            f'registry, its implementation can be found in '
            f'{obj_cls.__module__}')  # type: ignore
        return obj

    except Exception as e:
        # Normal TypeError does not print class name.
        cls_location = '/'.join(obj_cls.__module__.split('.'))  # type: ignore
        raise type(e)(
            f'class `{obj_cls.__name__}` in '  # type: ignore
            f'{cls_location}.py: {e}')


class Registry:
    """A registry to map strings to classes.

    Registered objects could be built from registry.

    Args:
        name (str): Registry name.
        build_func (callable, optional): A function to construct instance
            from Registry. :func:`build_from_cfg` is used if neither ``parent``
            or ``build_func`` is specified. If ``parent`` is specified and
            ``build_func`` is not given,  ``build_func`` will be inherited
            from ``parent``. Defaults to None.
        parent (:obj:`Registry`, optional): Parent registry. The class
            registered in children registry could be built from parent.
            Defaults to None.
        scope (str, optional): The scope of registry. It is the key to search
            for children registry. If not specified, scope will be the name of
            the package where class is defined, e.g. mmdet, mmcls, mmseg.
            Defaults to None.

    Examples:
        >>> # define a registry
        >>> MODELS = Registry('models')
        >>> # registry the `ResNet` to `MODELS`
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> # build model from `MODELS`
        >>> resnet = MODELS.build(dict(type='ResNet'))

        >>> # hierarchical registry
        >>> DETECTORS = Registry('detectors', parent=MODELS, scope='det')
        >>> @DETECTORS.register_module()
        >>> class FasterRCNN:
        >>>     pass
        >>> fasterrcnn = DETECTORS.build(dict(type='FasterRCNN'))

    More advanced usages can be found at
    https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
    """

    def __init__(self,
                 name: str,
                 build_func: Optional[Callable] = None,
                 parent: Optional['Registry'] = None,
                 scope: Optional[str] = None):
        self._name = name
        self._module_dict: Dict[str, Type] = dict()
        self._children: Dict[str, 'Registry'] = dict()

        if scope is not None:
            assert isinstance(scope, str)
            self._scope = scope
        else:
            self._scope = self.infer_scope() if scope is None else scope

        # See https://mypy.readthedocs.io/en/stable/common_issues.html#
        # variables-vs-type-aliases for the use
        self.parent: Optional['Registry']
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_child(self)
            self.parent = parent
        else:
            self.parent = None

        # self.build_func will be set with the following priority:
        # 1. build_func
        # 2. parent.build_func
        # 3. build_from_cfg
        self.build_func: Callable
        if build_func is None:
            if self.parent is not None:
                self.build_func = self.parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func

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

        Returns:
            str: The inferred scope name.

        Examples:
            >>> # in mmdet/models/backbone/resnet.py
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            >>> # The scope of ``ResNet`` will be ``mmdet``.
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

        Return:
            tuple[str | None, str]: The former element is the first scope of
            the key, which can be ``None``. The latter is the remaining key.

        Examples:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'
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

    @property
    def root(self):
        return self._get_root_registry()

    def _get_root_registry(self) -> 'Registry':
        """Return the root registry."""
        root = self
        while root.parent is not None:
            root = root.parent
        return root

    def get(self, key: str) -> Optional[Type]:
        """Get the registry record.

        The method will first parse :attr:`key` and check whether it contains
        a scope name. The logic to search for :attr:`key`:

        - ``key`` does not contain a scope name, i.e., it is purely a module
          name like "ResNet": :meth:`get` will search for ``ResNet`` from the
          current registry to its parent or ancestors until finding it.

        - ``key`` contains a scope name and it is equal to the scope of the
          current registry (e.g., "mmcls"), e.g., "mmcls.ResNet": :meth:`get`
          will only search for ``ResNet`` in the current registry.

        - ``key`` contains a scope name and it is not equal to the scope of
          the current registry (e.g., "mmdet"), e.g., "mmcls.FCNet": If the
          scope exists in its children, :meth:`get`will get "FCNet" from
          them. If not, :meth:`get` will first get the root registry and root
          registry call its own :meth:`get` method.

        Args:
            key (str): Name of the registered item, e.g., the class name in
                string format.

        Returns:
            Type or None: Return the corresponding class if ``key`` exists,
            otherwise return None.

        Examples:
            >>> # define a registry
            >>> MODELS = Registry('models')
            >>> # register `ResNet` to `MODELS`
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet_cls = MODELS.get('ResNet')

            >>> # hierarchical registry
            >>> DETECTORS = Registry('detector', parent=MODELS, scope='det')
            >>> # `ResNet` does not exist in `DETECTORS` but `get` method
            >>> # will try to search from its parenet or ancestors
            >>> resnet_cls = DETECTORS.get('ResNet')
            >>> CLASSIFIER = Registry('classifier', parent=MODELS, scope='cls')
            >>> @CLASSIFIER.register_module()
            >>> class MobileNet:
            >>>     pass
            >>> # `get` from its sibling registries
            >>> mobilenet_cls = DETECTORS.get('cls.MobileNet')
        """
        # Avoid circular import
        from ..logging.logger import MMLogger

        scope, real_key = self.split_scope_key(key)
        obj_cls = None
        registry_name = self.name
        scope_name = self.scope
        if scope is None or scope == self._scope:
            # get from self
            if real_key in self._module_dict:
                obj_cls = self._module_dict[real_key]

            elif scope is None:
                # try to get the target from its parent or ancestors
                parent = self.parent
                while parent is not None:
                    if real_key in parent._module_dict:
                        obj_cls = parent._module_dict[real_key]
                        registry_name = parent.name
                        scope_name = parent.scope
                        break
                    parent = parent.parent
        else:
            # get from self._children
            if scope in self._children:
                obj_cls = self._children[scope].get(real_key)
                registry_name = self._children[scope].name
                scope_name = scope
            else:
                root = self._get_root_registry()

                if scope != root._scope and scope not in root._children:
                    # If not skip directly, `root.get(key)` will recursively
                    # call itself until RecursionError is thrown.
                    pass
                else:
                    obj_cls = root.get(key)

        if obj_cls is not None:
            logger: MMLogger = MMLogger.get_current_instance()
            logger.info(
                f'Get class `{obj_cls.__name__}` from "{registry_name}"'
                f' registry in "{scope_name}"')
        return obj_cls

    def _search_child(self, scope: str) -> Optional['Registry']:
        """Depth-first search for the corresponding registry in its children.

        Note that the method only search for the corresponding registry from
        the current registry. Therefore, if we want to search from the root
        registry, :meth:`_get_root_registry` should be called to get the
        root registry first.

        Args:
            scope (str): The scope name used for searching for its
                corresponding registry.

        Returns:
            Registry or None: Return the corresponding registry if ``scope``
            exists, otherwise return None.
        """
        if self._scope == scope:
            return self

        for child in self._children.values():
            registry = child._search_child(scope)
            if registry is not None:
                return registry

        return None

    def build(self, cfg, *args, **kwargs) -> Any:
        """Build an instance.

        Build an instance by calling :attr:`build_func`. If the global
        variable default scope (:obj:`DefaultScope`) exists ,
        :meth:`build` will firstly get the responding registry and then call
        its own :meth:`build`.

        Examples:
            >>> from mmengine import Registry
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     def __init__(self, depth, stages=4):
            >>>         self.depth = depth
            >>>         self.stages = stages
            >>> cfg = dict(type='ResNet', depth=50)
            >>> model = MODELS.build(cfg)
        """
        with DefaultScope.overwrite_default_scope(cfg.pop('_scope_', None)):
            # get the global default scope
            default_scope = DefaultScope.get_current_instance()
            if default_scope is not None:
                scope_name = default_scope.scope_name
                root = self._get_root_registry()
                registry = root._search_child(scope_name)
                if registry is None:
                    # if `default_scope` can not be found, fallback to use self
                    warnings.warn(
                        f'Failed to search registry with scope "{scope_name}" '
                        f'in the "{root.name}" registry tree. '
                        f'As a workaround, the current "{self.name}" registry '
                        f'in "{self.scope}" is used to build instance. This '
                        f'may cause unexpected failure when running the built '
                        f'modules. Please check whether "{scope_name}" is a '
                        f'correct scope, or whether the registry is '
                        f'initialized.')
                    registry = self
            else:
                registry = self
            return registry.build_func(cfg, *args, **kwargs, registry=registry)

    def _add_child(self, registry: 'Registry') -> None:
        """Add a child for a registry.

        Args:
            registry (:obj:`Registry`): The ``registry`` will be added as a
                child of the ``self``.
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
        """Register a module.

        Args:
            module_class (type): Module class to be registered.
            module_name (str or list of str, optional): The module name to be
                registered. If not specified, the class name will be used.
                Defaults to None.
            force (bool): Whether to override an existing class with the same
                name. Defaults to False.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, '
                            f'but got {type(module_class)}')

        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                existed_module = self.module_dict[name]
                raise KeyError(f'{name} is already registered in {self.name} '
                               f'at {existed_module.__module__}')
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

        Args:
            name (str or list of str, optional): The module name to be
                registered. If not specified, the class name will be used.
            force (bool): Whether to override an existing class with the same
                name. Default to False.
            module (type, optional): Module class to be registered. Defaults to
                None.

        Examples:
            >>> backbones = Registry('backbone')
            >>> # as a decorator
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> # as a normal function
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(module=ResNet)
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                'name must be None, an instance of str, or a sequence of str, '
                f'but got {type(name)}')

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
