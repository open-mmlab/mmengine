# Copyright (c) OpenMMLab. All rights reserved.
import ast
import copy
import os
import tempfile
import warnings
from abc import ABCMeta, abstractmethod
from argparse import Action, ArgumentParser, Namespace
from collections import OrderedDict, abc
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

from addict import Dict
from yapf.yapflib.yapf_api import FormatCode

from mmengine.fileio import dump
from mmengine.logging import print_log
from .lazy import LazyObject
from .utils import ConfigParsingError, _is_builtin_module

BASE_KEY = '_base_'
DELETE_KEY = '_delete_'


def _lazy2string(cfg_dict, dict_type=None):
    if isinstance(cfg_dict, dict):
        dict_type = dict_type or type(cfg_dict)
        return dict_type({k: _lazy2string(v) for k, v in dict.items(cfg_dict)})
    elif isinstance(cfg_dict, (tuple, list)):
        return type(cfg_dict)(_lazy2string(v) for v in cfg_dict)
    elif isinstance(cfg_dict, LazyObject):
        return cfg_dict.dump_str
    else:
        return cfg_dict


class ConfigDict(Dict):
    """A dictionary for config which has the same interface as python's built-
    in dictionary and can be used as a normal dictionary.

    The Config class would transform the nested fields (dictionary-like fields)
    in config file into ``ConfigDict``.

    If the class attribute ``lazy``  is ``False``, users will get the
    object built by ``LazyObject`` or ``LazyAttr``, otherwise users will get
    the ``LazyObject`` or ``LazyAttr`` itself.

    The ``lazy`` should be set to ``True`` to avoid building the imported
    object during configuration parsing, and it should be set to False outside
    the Config to ensure that users do not experience the ``LazyObject``.
    """
    lazy = False

    def __init__(__self, *args, **kwargs):
        object.__setattr__(__self, '__parent', kwargs.pop('__parent', None))
        object.__setattr__(__self, '__key', kwargs.pop('__key', None))
        object.__setattr__(__self, '__frozen', False)
        for arg in args:
            if not arg:
                continue
            # Since ConfigDict.items will convert LazyObject to real object
            # automatically, we need to call super().items() to make sure
            # the LazyObject will not be converted.
            if isinstance(arg, ConfigDict):
                for key, val in dict.items(arg):
                    __self[key] = __self._hook(val)
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    __self[key] = __self._hook(val)
            elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
                __self[arg[0]] = __self._hook(arg[1])
            else:
                for key, val in iter(arg):
                    __self[key] = __self._hook(val)

        for key, val in dict.items(kwargs):
            __self[key] = __self._hook(val)

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
            if isinstance(value, LazyObject) and not self.lazy:
                value = value.build()
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no "
                                 f"attribute '{name}'")
        except Exception as e:
            raise e
        else:
            return value

    @classmethod
    def _hook(cls, item):
        # avoid to convert user defined dict to ConfigDict.
        if type(item) in (dict, OrderedDict):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)
        return item

    def __setattr__(self, name, value):
        value = self._hook(value)
        return super().__setattr__(name, value)

    def __setitem__(self, name, value):
        value = self._hook(value)
        return super().__setitem__(name, value)

    def __getitem__(self, key):
        return self.build_lazy(super().__getitem__(key))

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for key, value in super().items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return other

    def __copy__(self):
        other = self.__class__()
        for key, value in super().items():
            other[key] = value
        return other

    copy = __copy__

    def __iter__(self):
        # Implement `__iter__` to overwrite the unpacking operator `**cfg_dict`
        # to get the built lazy object
        return iter(self.keys())

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get the value of the key. If class attribute ``lazy`` is True, the
        LazyObject will be built and returned.

        Args:
            key (str): The key.
            default (any, optional): The default value. Defaults to None.

        Returns:
            Any: The value of the key.
        """
        return self.build_lazy(super().get(key, default))

    def pop(self, key, default=None):
        """Pop the value of the key. If class attribute ``lazy`` is True, the
        LazyObject will be built and returned.

        Args:
            key (str): The key.
            default (any, optional): The default value. Defaults to None.

        Returns:
            Any: The value of the key.
        """
        return self.build_lazy(super().pop(key, default))

    def update(self, *args, **kwargs) -> None:
        """Override this method to make sure the LazyObject will not be built
        during updating."""
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError('update only accept one positional argument')
            # Avoid to used self.items to build LazyObject
            for key, value in dict.items(args[0]):
                other[key] = value

        for key, value in dict(kwargs).items():
            other[key] = value
        for k, v in other.items():
            if ((k not in self) or (not isinstance(self[k], dict))
                    or (not isinstance(v, dict))):
                self[k] = self._hook(v)
            else:
                self[k].update(v)

    def build_lazy(self, value: Any) -> Any:
        """If class attribute ``lazy`` is False, the LazyObject will be built
        and returned.

        Args:
            value (Any): The value to be built.

        Returns:
            Any: The built value.
        """
        if isinstance(value, LazyObject) and not self.lazy:
            value = value.build()
        return value

    def values(self):
        """Yield the values of the dictionary.

        If class attribute ``lazy`` is False, the value of ``LazyObject`` or
        ``LazyAttr`` will be built and returned.
        """
        values = []
        for value in super().values():
            values.append(self.build_lazy(value))
        return values

    def items(self):
        """Yield the keys and values of the dictionary.

        If class attribute ``lazy`` is False, the value of ``LazyObject`` or
        ``LazyAttr`` will be built and returned.
        """
        items = []
        for key, value in super().items():
            items.append((key, self.build_lazy(value)))
        return items

    def merge(self, other: dict):
        """Merge another dictionary into current dictionary.

        Args:
            other (dict): Another dictionary.
        """
        default = object()

        def _merge_a_into_b(a, b):
            if isinstance(a, dict):
                if not isinstance(b, dict):
                    a.pop(DELETE_KEY, None)
                    return a
                if a.pop(DELETE_KEY, False):
                    b.clear()
                all_keys = list(b.keys()) + list(a.keys())
                return {
                    key:
                    _merge_a_into_b(a.get(key, default), b.get(key, default))
                    for key in all_keys if key != DELETE_KEY
                }
            else:
                return a if a is not default else b

        merged = _merge_a_into_b(copy.deepcopy(other), copy.deepcopy(self))
        self.clear()
        for key, value in merged.items():
            self[key] = value

    def __getstate__(self):
        state = {}
        for key, value in super().items():
            state[key] = value
        return state

    def __setstate__(self, state):
        for key, value in state.items():
            self[key] = value

    def __eq__(self, other):
        if isinstance(other, ConfigDict):
            return other.to_dict() == self.to_dict()
        elif isinstance(other, dict):
            return {k: v for k, v in self.items()} == other
        else:
            return False

    def _to_lazy_dict(self):
        """Convert the ConfigDict to a normal dictionary recursively, and keep
        the ``LazyObject`` or ``LazyAttr`` object not built."""

        def _to_dict(data):
            if isinstance(data, ConfigDict):
                return {
                    key: _to_dict(value)
                    for key, value in Dict.items(data)
                }
            elif isinstance(data, dict):
                return {key: _to_dict(value) for key, value in data.items()}
            elif isinstance(data, (list, tuple)):
                return type(data)(_to_dict(item) for item in data)
            else:
                return data

        return _to_dict(self)

    def to_dict(self):
        """Convert the ConfigDict to a normal dictionary recursively, and
        convert the ``LazyObject`` or ``LazyAttr`` to string."""
        return _lazy2string(self, dict_type=dict)


def add_args(parser: ArgumentParser,
             cfg: dict,
             prefix: str = '') -> ArgumentParser:
    """Add config fields into argument parser.

    Args:
        parser (ArgumentParser): Argument parser.
        cfg (dict): Config dictionary.
        prefix (str, optional): Prefix of parser argument.
            Defaults to ''.

    Returns:
        ArgumentParser: Argument parser containing config fields.
    """
    for k, v in cfg.items():
        if isinstance(v, str):
            parser.add_argument('--' + prefix + k)
        elif isinstance(v, bool):
            parser.add_argument('--' + prefix + k, action='store_true')
        elif isinstance(v, int):
            parser.add_argument('--' + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument('--' + prefix + k, type=float)
        elif isinstance(v, dict):
            add_args(parser, v, prefix + k + '.')
        elif isinstance(v, abc.Iterable):
            parser.add_argument(
                '--' + prefix + k, type=type(next(iter(v))), nargs='+')
        else:
            print_log(
                f'cannot parse key {prefix + k} of type {type(v)}',
                logger='current')
    return parser


class Config(metaclass=ABCMeta):
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml.
    ``Config.fromfile`` can parse a dictionary from a config file, then
    build a ``Config`` instance with the dictionary.
    The interface is the same as a dict object and also allows access config
    values as attributes.

    Args:
        cfg_dict (dict, optional): A config dictionary. Defaults to None.
        cfg_text (str, optional): Text of config. Defaults to None.
        filename (str or Path, optional): Name of config file.
            Defaults to None.
        format_python_code (bool): Whether to format Python code by yapf.
            Defaults to True.

    Here is a simple example:

    Examples:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/username/projects/mmengine/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/username/projects/mmengine/tests/data/config/a.py]
        :"
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"

    You can find more advance usage in the `config tutorial`_.

    .. _config tutorial: https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html
    """  # noqa: E501

    @classmethod
    def __new__(cls, *args, **kwargs):
        if cls is Config:
            from .new_config import ConfigV2
            cls = ConfigV2
        return super().__new__(cls)

    @staticmethod
    def fromfile(filename: Union[str, Path],
                 lazy_import: Optional[bool] = None,
                 format_python_code: bool = True,
                 **kwargs) -> 'Config':
        """Build a Config instance from config file.

        Args:
            filename (str or Path): Name of config file.
            use_predefined_variables (bool, optional): Whether to use
                predefined variables. Defaults to True.
            import_custom_modules (bool, optional): Whether to support
                importing custom modules in config. Defaults to None.
            lazy_import (bool): Whether to load config in `lazy_import` mode.
                If it is `None`, it will be deduced by the content of the
                config file. Defaults to None.
            format_python_code (bool): Whether to format Python code by yapf.
                Defaults to True.

        Returns:
            Config: Config instance built from config file.
        """
        filename = str(filename) if isinstance(filename, Path) else filename
        if lazy_import is None:
            lazy_import = Config._is_lazy_import(filename)

        if not lazy_import:
            from .old_config import ConfigV1
            return ConfigV1.fromfile(
                filename=filename,
                format_python_code=format_python_code,
                **kwargs)
        else:
            from .new_config import ConfigV2
            return ConfigV2.fromfile(
                filename=filename,
                format_python_code=format_python_code,
                **kwargs)

    @staticmethod
    def fromstring(cfg_str: str, file_format: str) -> 'Config':
        """Build a Config instance from config text.

        Args:
            cfg_str (str): Config text.
            file_format (str): Config file format corresponding to the
               config str. Only py/yml/yaml/json type are supported now!

        Returns:
            Config: Config object generated from ``cfg_str``.
        """
        if file_format not in ['.py', '.json', '.yaml', '.yml']:
            raise OSError('Only py/yml/yaml/json type are supported now!')
        if file_format != '.py' and 'dict(' in cfg_str:
            # check if users specify a wrong suffix for python
            warnings.warn(
                'Please check "file_format", the file format may be .py')

        # A temporary file can not be opened a second time on Windows.
        # See https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile for more details. # noqa
        # `temp_file` is opened first in `tempfile.NamedTemporaryFile` and
        #  second in `Config.from_file`.
        # In addition, a named temporary file will be removed after closed.
        # As a workaround we set `delete=False` and close the temporary file
        # before opening again.

        with tempfile.NamedTemporaryFile(
                'w', encoding='utf-8', suffix=file_format,
                delete=False) as temp_file:
            temp_file.write(cfg_str)

        cfg = Config.fromfile(temp_file.name)
        os.remove(temp_file.name)  # manually delete the temporary file
        return cfg

    @staticmethod
    def _validate_py_syntax(filename: str):
        """Validate syntax of python config.

        Args:
            filename (str): Filename of python config file.
        """
        with open(filename, encoding='utf-8') as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError('There are syntax errors in config '
                              f'file {filename}: {e}')

    @staticmethod
    def _merge_a_into_b(a: dict,
                        b: dict,
                        allow_list_keys: bool = False) -> dict:
        """merge dict ``a`` into dict ``b`` (non-inplace).

        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.

        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in source ``a`` and will replace the element of the
              corresponding index in b if b is a list. Defaults to False.

        Returns:
            dict: The modified dict of ``b`` using ``a``.

        Examples:
            # Normally merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # Delete b first and merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # b is a list
            >>> Config._merge_a_into_b(
            ...     {'0': dict(a=2)}, [dict(a=1), dict(b=2)], True)
            [{'a': 2}, {'b': 2}]
        """
        b = b.copy()
        for k, v in a.items():
            if allow_list_keys and k.isdigit() and isinstance(b, list):
                k = int(k)
                if len(b) <= k:
                    raise KeyError(f'Index {k} exceeds the length of list {b}')
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
            elif isinstance(v, dict):
                if k in b and not v.pop(DELETE_KEY, False):
                    allowed_types: Union[Tuple, type] = (
                        dict, list) if allow_list_keys else dict
                    if not isinstance(b[k], allowed_types):
                        raise TypeError(
                            f'{k}={v} in child config cannot inherit from '
                            f'base because {k} is a dict in the child config '
                            f'but is of type {type(b[k])} in base config. '
                            f'You may set `{DELETE_KEY}=True` to ignore the '
                            f'base config.')
                    b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
                else:
                    b[k] = ConfigDict(v)
            else:
                b[k] = v
        return b

    @staticmethod
    def auto_argparser(description=None):
        """Generate argparser from config file automatically (experimental)"""
        partial_parser = ArgumentParser(description=description)
        partial_parser.add_argument('config', help='config file path')
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.fromfile(cfg_file)
        parser = ArgumentParser(description=description)
        parser.add_argument('config', help='config file path')
        add_args(parser, cfg)
        return parser, cfg

    @property
    @abstractmethod
    def filename(self) -> str:
        """get file name of config."""

    @property
    @abstractmethod
    def text(self) -> str:
        """get config text."""

    @property
    def pretty_text(self) -> str:
        """get formatted python config text."""

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = repr(v)
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f'{k_str}: {v_str}'
            else:
                attr_str = f'{str(k)}={v_str}'
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list_tuple(k, v, use_mapping=False):
            if isinstance(v, list):
                left = '['
                right = ']'
            else:
                left = '('
                right = ')'

            v_str = f'{left}\n'
            # check if all items in the list are dict
            for item in v:
                if isinstance(item, dict):
                    v_str += f'dict({_indent(_format_dict(item), indent)}),\n'
                elif isinstance(item, tuple):
                    v_str += f'{_indent(_format_list_tuple(None, item), indent)},\n'  # noqa: 501
                elif isinstance(item, list):
                    v_str += f'{_indent(_format_list_tuple(None, item), indent)},\n'  # noqa: 501
                elif isinstance(item, str):
                    v_str += f'{_indent(repr(item), indent)},\n'
                else:
                    v_str += str(item) + ',\n'
            if k is None:
                return _indent(v_str, indent) + right
            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f'{k_str}: {v_str}'
            else:
                attr_str = f'{str(k)}={v_str}'
            attr_str = _indent(attr_str, indent) + right
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= \
                    (not str(key_name).isidentifier())
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ''
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += '{'
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = '' if outest_level or is_last else ','
                if isinstance(v, dict):
                    v_str = '\n' + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f'{k_str}: dict({v_str}'
                    else:
                        attr_str = f'{str(k)}=dict({v_str}'
                    attr_str = _indent(attr_str, indent) + ')' + end
                elif isinstance(v, (list, tuple)):
                    attr_str = _format_list_tuple(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += '\n'.join(s)
            if use_mapping:
                r += '}'
            return r

        cfg_dict = self.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        if self._format_python_code:
            # copied from setup.cfg
            yapf_style = dict(
                based_on_style='pep8',
                blank_line_before_nested_class_or_def=True,
                split_before_expression_after_opening_paren=True)
            try:
                text, _ = FormatCode(
                    text, style_config=yapf_style, verify=True)
            except:  # noqa: E722
                raise SyntaxError('Failed to format the config file, please '
                                  f'check the syntax of: \n{text}')
        return text

    def __repr__(self):
        return f'Config (path: {self.filename}): {self._cfg_dict.__repr__()}'

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    @abstractmethod
    def __getstate__(self) -> Tuple:
        pass

    def __deepcopy__(self, memo):
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            super(Config, other).__setattr__(key, copy.deepcopy(value, memo))

        return other

    def __copy__(self):
        cls = self.__class__
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)
        super(Config, other).__setattr__('_cfg_dict', self._cfg_dict.copy())

        return other

    copy = __copy__

    @abstractmethod
    def __setstate__(self, state: Tuple):
        pass

    def dump(self, file: Optional[Union[str, Path]] = None):
        """Dump config to file or return config text.

        Args:
            file (str or Path, optional): If not specified, then the object
            is dumped to a str, otherwise to a file specified by the filename.
            Defaults to None.

        Returns:
            str or None: Config text.
        """
        file = str(file) if isinstance(file, Path) else file
        cfg_dict = self.to_dict()
        if file is None:
            if self.filename is None or self.filename.endswith('.py'):
                return self.pretty_text
            else:
                file_format = self.filename.split('.')[-1]
                return dump(cfg_dict, file_format=file_format)
        elif file.endswith('.py'):
            with open(file, 'w', encoding='utf-8') as f:
                f.write(self.pretty_text)
        else:
            file_format = file.split('.')[-1]
            return dump(cfg_dict, file=file, file_format=file_format)

    def merge_from_dict(self,
                        options: dict,
                        allow_list_keys: bool = True) -> None:
        """Merge list into cfg_dict.

        Merge the dict parsed by MultipleKVAction into this cfg.

        Args:
            options (dict): dict of configs to merge from.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
                are allowed in ``options`` and will replace the element of the
                corresponding index in the config if the config is a list.
                Defaults to True.

        Examples:
            >>> from mmengine import Config
            >>> #  Merge dictionary element
            >>> options = {'model.backbone.depth': 50, 'model.backbone.with_cp': True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
            >>> cfg._cfg_dict
            {'model': {'backbone': {'type': 'ResNet', 'depth': 50, 'with_cp': True}}}
            >>> # Merge list element
            >>> cfg = Config(
            >>>     dict(pipeline=[dict(type='LoadImage'),
            >>>                    dict(type='LoadAnnotations')]))
            >>> options = dict(pipeline={'0': dict(type='SelfLoadImage')})
            >>> cfg.merge_from_dict(options, allow_list_keys=True)
            >>> cfg._cfg_dict
            {'pipeline': [{'type': 'SelfLoadImage'}, {'type': 'LoadAnnotations'}]}
        """  # noqa: E501
        option_cfg_dict: dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split('.')
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super().__getattribute__('_cfg_dict')
        super().__setattr__(
            '_cfg_dict',
            Config._merge_a_into_b(
                option_cfg_dict, cfg_dict, allow_list_keys=allow_list_keys))

    @staticmethod
    def _is_lazy_import(filename: str) -> bool:
        if not filename.endswith('.py'):
            return False
        with open(filename, encoding='utf-8') as f:
            codes_str = f.read()
            parsed_codes = ast.parse(codes_str)
        for node in ast.walk(parsed_codes):
            if (isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == BASE_KEY):
                return False

            if isinstance(node, ast.With):
                expr = node.items[0].context_expr
                if (not isinstance(expr, ast.Call)
                        or not expr.func.id == 'read_base'):  # type: ignore
                    raise ConfigParsingError(
                        'Only `read_base` context manager can be used in the '
                        'config')
                return True
            if isinstance(node, ast.ImportFrom):
                # relative import -> lazy_import
                if node.level != 0:
                    return True
                # Skip checking when using `mmengine.config` in cfg file
                if (node.module == 'mmengine' and len(node.names) == 1
                        and node.names[0].name == 'Config'):
                    continue
                if not isinstance(node.module, str):
                    continue
                # non-builtin module -> lazy_import
                if not _is_builtin_module(node.module):
                    return True
            if isinstance(node, ast.Import):
                for alias_node in node.names:
                    if not _is_builtin_module(alias_node.name):
                        return True
        return False

    def to_dict(self):
        """Convert all data in the config to a builtin ``dict``."""
        return self._cfg_dict.to_dict()


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val: str) -> Union[int, float, bool, Any]:
        """parse int/float/bool value in the string."""
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        if val == 'None':
            return None
        return val

    @staticmethod
    def _parse_iterable(val: str) -> Union[list, tuple, Any]:
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple | Any: The expanded list or tuple from the string,
            or single value if no iterable values are found.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count('(') == string.count(')')) and (
                    string.count('[') == string.count(']')), \
                f'Imbalanced brackets exist in {string}'
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if ((char == ',') and (pre.count('(') == pre.count(')'))
                        and (pre.count('[') == pre.count(']'))):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip('\'\"').replace(' ', '')
        is_tuple = False
        if val.startswith('(') and val.endswith(')'):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith('[') and val.endswith(']'):
            val = val[1:-1]
        elif ',' not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1:]

        if is_tuple:
            return tuple(values)

        return values

    def __call__(self,
                 parser: ArgumentParser,
                 namespace: Namespace,
                 values: Union[str, Sequence[Any], None],
                 option_string: str = None):
        """Parse Variables in string and add them into argparser.

        Args:
            parser (ArgumentParser): Argument parser.
            namespace (Namespace): Argument namespace.
            values (Union[str, Sequence[Any], None]): Argument string.
            option_string (list[str], optional): Option string.
                Defaults to None.
        """
        # Copied behavior from `argparse._ExtendAction`.
        options = copy.copy(getattr(namespace, self.dest, None) or {})
        if values is not None:
            for kv in values:
                key, val = kv.split('=', maxsplit=1)
                options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)
