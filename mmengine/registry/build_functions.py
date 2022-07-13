# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from typing import Any, Optional, Union

from ..config import Config, ConfigDict
from ..utils import ManagerMixin
from .registry import Registry


def build_from_cfg(
        cfg: Union[dict, ConfigDict, Config],
        registry: Registry,
        default_args: Optional[Union[dict, ConfigDict, Config]] = None) -> Any:
    """Build a module from config dict when it is a class configuration, or
    call a function from config dict when it is a function configuration.

    If the global variable default scope (:obj:`DefaultScope`) exists,
    :meth:`build` will firstly get the responding registry and then call
    its own :meth:`build`.

    At least one of the ``cfg`` and ``default_args`` contains the key "type",
    which should be either str or class. If they all contain it, the key
    in ``cfg`` will be used because ``cfg`` has a high priority than
    ``default_args`` that means if a key exists in both of them, the value of
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
        >>> # Returns an instantiated object
        >>> @MODELS.register_module()
        >>> def resnet50():
        >>>     pass
        >>> resnet = build_from_cfg(dict(type='resnet50'), MODELS)
        >>> # Return a result of the calling function

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
    from ..logging import print_log

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

    # Instance should be built under target scope, if `_scope_` is defined
    # in cfg, current default scope should switch to specified scope
    # temporarily.
    scope = args.pop('_scope_', None)
    with registry.switch_scope_and_registry(scope) as registry:
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = registry.get(obj_type)
            if obj_cls is None:
                raise KeyError(
                    f'{obj_type} is not in the {registry.name} registry. '
                    f'Please check whether the value of `{obj_type}` is '
                    'correct or it was registered as expected. More details '
                    'can be found at '
                    'https://mmengine.readthedocs.io/en/latest/tutorials/config.html#import-custom-python-modules'  # noqa: E501
                )
        elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        try:
            # If `obj_cls` inherits from `ManagerMixin`, it should be
            # instantiated by `ManagerMixin.get_instance` to ensure that it
            # can be accessed globally.
            if issubclass(obj_cls, ManagerMixin):  # type: ignore
                obj = obj_cls.get_instance(**args)  # type: ignore
            else:
                obj = obj_cls(**args)  # type: ignore

            print_log(
                f'An `{obj_cls.__name__}` instance is built from '  # type: ignore # noqa: E501
                'registry, its implementation can be found in '
                f'{obj_cls.__module__}',  # type: ignore
                logger='current')
            return obj

        except Exception as e:
            # Normal TypeError does not print class name.
            cls_location = '/'.join(
                obj_cls.__module__.split('.'))  # type: ignore
            raise type(e)(
                f'class `{obj_cls.__name__}` in '  # type: ignore
                f'{cls_location}.py: {e}')


def build_runner_from_cfg(cfg: Union[dict, ConfigDict, Config],
                          registry: Registry) -> Any:
    """Build a Runner object.
    Examples:
        >>> from mmengine.registry import Registry, build_runner_from_cfg
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
    from ..logging import print_log

    assert isinstance(
        cfg,
        (dict, ConfigDict, Config
         )), f'cfg should be a dict, ConfigDict or Config, but got {type(cfg)}'
    assert isinstance(
        registry, Registry), ('registry should be a mmengine.Registry object',
                              f'but got {type(registry)}')

    args = cfg.copy()
    # Runner should be built under target scope, if `_scope_` is defined
    # in cfg, current default scope should switch to specified scope
    # temporarily.
    scope = args.pop('_scope_', None)
    with registry.switch_scope_and_registry(scope) as registry:
        obj_type = args.pop('runner_type', 'mmengine.Runner')
        if isinstance(obj_type, str):
            runner_cls = registry.get(obj_type)
            if runner_cls is None:
                raise KeyError(
                    f'{obj_type} is not in the {registry.name} registry. '
                    f'Please check whether the value of `{obj_type}` is '
                    'correct or it was registered as expected. More details '
                    'can be found at https://mmengine.readthedocs.io/en/latest/tutorials/config.html#import-custom-python-modules'  # noqa: E501
                )
        elif inspect.isclass(obj_type):
            runner_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        try:
            runner = runner_cls.from_cfg(args)  # type: ignore
            print_log(
                f'An `{runner_cls.__name__}` instance is built from '  # type: ignore # noqa: E501
                'registry, its implementation can be found in'
                f'{runner_cls.__module__}',  # type: ignore
                logger='current')
            return runner

        except Exception as e:
            # Normal TypeError does not print class name.
            cls_location = '/'.join(
                runner_cls.__module__.split('.'))  # type: ignore
            raise type(e)(
                f'class `{runner_cls.__name__}` in '  # type: ignore
                f'{cls_location}.py: {e}')


def build_model_from_cfg(cfg, registry, default_args=None):
    """Build a PyTorch model from config dict(s). Different from
    ``build_from_cfg``, if cfg is a list, a ``nn.Sequential`` will be built.

    Args:
        cfg (dict, list[dict]): The config of modules, which is either a config
            dict or a list of config dicts. If cfg is a list, the built
            modules will be wrapped with ``nn.Sequential``.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn.Module.
    """
    from ..model import Sequential
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(_cfg, registry, default_args) for _cfg in cfg
        ]
        return Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)
