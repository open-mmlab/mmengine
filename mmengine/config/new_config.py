# Copyright (c) OpenMMLab. All rights reserved.
import copy
import importlib
import inspect
import platform
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import PathFinder
from importlib.util import spec_from_loader
from pathlib import Path
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Any, Optional, Tuple, Union

from yapf.yapflib.yapf_api import FormatCode

from mmengine.fileio import dump
from .config import ConfigDict
from .lazy import LazyImportContext, LazyObject

DELETE_KEY = '_delete_'
RESERVED_KEYS = ['filename', 'text', 'pretty_text']

if platform.system() == 'Windows':
    import regex as re
else:
    import re  # type: ignore


def format_inpsect(obj):
    file = inspect.getsourcefile(obj)
    lines, lineno = inspect.getsourcelines(obj)
    msg = f'File "{file}", line {lineno}\n--> {lines[0]}'
    return msg


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


class Config:
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
    _max_parent_depth = 4
    _parent_pkg = '_cfg_parent'

    def __init__(self,
                 cfg_dict: dict = None,
                 cfg_text: Optional[str] = None,
                 filename: Optional[Union[str, Path]] = None,
                 format_python_code: bool = True):
        filename = str(filename) if isinstance(filename, Path) else filename
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')

        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f'{key} is reserved for config file')
        self._sanity_check(cfg_dict)

        if not isinstance(cfg_dict, ConfigDict):
            cfg_dict = ConfigDict(cfg_dict)
        super().__setattr__('_cfg_dict', cfg_dict)
        super().__setattr__('_filename', filename)
        super().__setattr__('_format_python_code', format_python_code)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, encoding='utf-8') as f:
                text = f.read()
        else:
            text = ''
        super().__setattr__('_text', text)

    @staticmethod
    def _sanity_check(cfg):
        if isinstance(cfg, dict):
            for v in cfg.values():
                Config._sanity_check(v)
        elif isinstance(cfg, (tuple, list, set)):
            for v in cfg:
                Config._sanity_check(v)
        elif isinstance(cfg, (type, FunctionType)):
            if (Config._parent_pkg in cfg.__module__
                    or '__main__' in cfg.__module__):
                msg = ('You cannot use temporary functions '
                       'as the value of a field.\n\n')
                msg += format_inpsect(cfg)
                raise ValueError(msg)

    @staticmethod
    def fromfile(filename: Union[str, Path],
                 lazy_import: Optional[bool] = None) -> 'Config':
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

        Returns:
            Config: Config instance built from config file.
        """
        # Enable lazy import when parsing the config.
        # Using try-except to make sure ``ConfigDict.lazy`` will be reset
        # to False. See more details about lazy in the docstring of
        # ConfigDict
        ConfigDict.lazy = lazy_import
        try:
            cfg_dict = Config._parse_lazy_import(filename)
        finally:
            ConfigDict.lazy = False

        for key, value in list(cfg_dict.to_dict().items()):
            # Remove functions or modules
            if isinstance(value, (LazyObject, ModuleType, FunctionType, type)):
                cfg_dict.pop(key)

        # Recover dumped lazy object like '<torch.nn.Linear>' from string
        cfg_dict = recover_lazy_field(cfg_dict)

        cfg = Config(cfg_dict, filename=filename)
        return cfg

    @staticmethod
    def _parse_lazy_import(filename: Union[str, Path]) -> ConfigDict:
        """Transform file to variables dictionary.

        Args:
            filename (str): Name of config file.

        Returns:
            Tuple[dict, dict]: ``cfg_dict`` and ``imported_names``.

              - cfg_dict (dict): Variables dictionary of parsed config.
              - imported_names (set): Used to mark the names of
                imported object.
        """
        module = Config._get_config_module(filename)
        module_dict = {
            k: getattr(module, k)
            for k in dir(module) if not k.startswith('__')
        }

        return ConfigDict(module_dict)

    @staticmethod
    def _get_config_module(filename: Union[str, Path], level=0):
        file = Path(filename).absolute()
        module_name = re.sub(r'\W|^(?=\d)', '_', file.stem)
        parent_pkg = Config._parent_pkg + str(level)
        fullname = '.'.join([parent_pkg] * Config._max_parent_depth +
                            [module_name])

        # import config file as a module
        with LazyImportContext():
            spec = importlib.util.spec_from_file_location(fullname, file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        return module

    @staticmethod
    def _dict_to_config_dict_lazy(cfg: dict):
        """Recursively converts ``dict`` to :obj:`ConfigDict`. The only
        difference between ``_dict_to_config_dict_lazy`` and
        ``_dict_to_config_dict_lazy`` is that the former one does not consider
        the scope, and will not trigger the building of ``LazyObject``.

        Args:
            cfg (dict): Config dict.

        Returns:
            ConfigDict: Converted dict.
        """
        # Only the outer dict with key `type` should have the key `_scope_`.
        if isinstance(cfg, dict):
            cfg_dict = ConfigDict()
            for key, value in cfg.items():
                cfg_dict[key] = Config._dict_to_config_dict_lazy(value)
            return cfg_dict
        if isinstance(cfg, (tuple, list)):
            return type(cfg)(
                Config._dict_to_config_dict_lazy(_cfg) for _cfg in cfg)
        return cfg

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

    @property
    def filename(self) -> str:
        """get file name of config."""
        return self._filename

    @property
    def text(self) -> str:
        """get config text."""
        return self._text

    @property
    def pretty_text(self) -> str:
        """get formatted python config text."""

        def _format_dict(input_dict):
            use_mapping = not all(str(k).isidentifier() for k in input_dict)

            if use_mapping:
                item_tmpl = '{k}: {v}'
            else:
                item_tmpl = '{k}={v}'

            items = []
            for k, v in input_dict.items():
                v_str = _format_basic_types(v)
                k_str = _format_basic_types(k) if use_mapping else k
                items.append(item_tmpl.format(k=k_str, v=v_str))
            items = ','.join(items)

            if use_mapping:
                return '{' + items + '}'
            else:
                return f'dict({items})'

        def _format_list_tuple_set(input_container):
            items = []

            for item in input_container:
                items.append(_format_basic_types(item))

            if isinstance(input_container, tuple):
                items = items + [''] if len(items) == 1 else items
                return '(' + ','.join(items) + ')'
            elif isinstance(input_container, list):
                return '[' + ','.join(items) + ']'
            elif isinstance(input_container, set):
                return '{' + ','.join(items) + '}'

        def _format_basic_types(input_):
            if isinstance(input_, str):
                return repr(input_)
            elif isinstance(input_, dict):
                return _format_dict(input_)
            elif isinstance(input_, (list, set, tuple)):
                return _format_list_tuple_set(input_)
            elif isinstance(input_, LazyObject):
                return repr(input_.dump_str)
            elif isinstance(input_, (type, FunctionType, BuiltinFunctionType)):
                if Config._parent_pkg in input_.__module__:
                    # defined in the config file.
                    module = input_.__module__.rpartition('.')[-1]
                else:
                    module = input_.__module__
                return repr('<' + module + '.' + input_.__name__ + '>')
            elif isinstance(input_, ModuleType):
                return repr(f'<{input_.__name__}>')
            elif 'torch.dtype' in str(type(input_)):
                return repr('<' + str(input_) + '>')
            else:
                return str(input_)

        cfg_dict = self._to_lazy_dict()

        items = []
        for k, v in cfg_dict.items():
            items.append(f'{k} = {_format_basic_types(v)}')

        text = '\n'.join(items)
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
        self._sanity_check(value)
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        self._sanity_check(value)
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(self) -> Tuple[dict, Optional[str], Optional[str], dict]:
        return (self._cfg_dict, self._filename, self._text)

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

        return other

    def __setstate__(self, state: Tuple[dict, Optional[str], Optional[str],
                                        dict]):
        _cfg_dict, _filename, _text = state
        super().__setattr__('_cfg_dict', _cfg_dict)
        super().__setattr__('_filename', _filename)
        super().__setattr__('_text', _text)

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
        cfg_dict = super().__getattribute__('_cfg_dict').to_dict()
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

    def _to_lazy_dict(self, keep_imported: bool = False) -> dict:
        """Convert config object to dictionary and filter the imported
        object."""
        res = self._cfg_dict.to_dict()

        def filter_item(item):
            _, v = item
            if isinstance(v, (LazyObject, ModuleType, FunctionType, type)):
                return False
            if v is read_base:
                return False
            return True

        if keep_imported:
            return res
        else:
            return dict(filter(filter_item, res.items()))

    def to_dict(self, keep_imported: bool = False):
        """Convert all data in the config to a builtin ``dict``.

        Args:
            keep_imported (bool): Whether to keep the imported field.
                Defaults to False

        If you import third-party objects in the config file, all imported
        objects will be converted to a string like ``torch.optim.SGD``
        """
        _cfg_dict = self._to_lazy_dict(keep_imported=keep_imported)

        def lazy2string(cfg_dict):
            if isinstance(cfg_dict, dict):
                return type(cfg_dict)(
                    {k: lazy2string(v)
                     for k, v in cfg_dict.items()})
            elif isinstance(cfg_dict, (tuple, list)):
                return type(cfg_dict)(lazy2string(v) for v in cfg_dict)
            elif isinstance(cfg_dict, LazyObject):
                return str(cfg_dict)
            else:
                return cfg_dict

        return lazy2string(_cfg_dict)


class BaseConfigLoader(Loader):

    def __init__(self, filepath, level) -> None:
        self.filepath = filepath
        self.level = level

    def create_module(self, spec):
        file = self.filepath
        return Config._get_config_module(file, level=self.level)

    def exec_module(self, module):
        for k in dir(module):
            module.__dict__[k] = Config._dict_to_config_dict_lazy(
                getattr(module, k))


class ParentFolderLoader(Loader):

    @staticmethod
    def create_module(spec):
        return ModuleType(spec.name)

    @staticmethod
    def exec_module(module):
        pass


class BaseImportContext(MetaPathFinder):

    def find_spec(self, fullname, path=None, target=None):
        """Try to find a spec for 'fullname' on sys.path or 'path'.

        The search is based on sys.path_hooks and sys.path_importer_cache.
        """
        parent_pkg = Config._parent_pkg + str(self.level)
        names = fullname.split('.')

        if names[-1] == parent_pkg:
            self.base_modules.append(fullname)
            # Create parent package
            return spec_from_loader(
                fullname, loader=ParentFolderLoader, is_package=True)
        elif names[0] == parent_pkg:
            self.base_modules.append(fullname)
            # relative imported base package
            filepath = self.root_path
            for name in names:
                if name == parent_pkg:
                    # Use parent to remove `..` at the end of the root path
                    filepath = filepath.parent
                else:
                    filepath = filepath / name
            if filepath.is_dir():
                # If a dir, create a package.
                return spec_from_loader(
                    fullname, loader=ParentFolderLoader, is_package=True)

            pypath = filepath.with_suffix('.py')

            if not pypath.exists():
                raise ImportError(f'Not found base path {filepath.resolve()}')
            return importlib.util.spec_from_loader(
                fullname, BaseConfigLoader(pypath, self.level + 1))
        else:
            # Absolute import
            pkg = PathFinder.find_spec(names[0])
            if pkg and pkg.submodule_search_locations:
                self.base_modules.append(fullname)
                path = Path(pkg.submodule_search_locations[0])
                for name in names[1:]:
                    path = path / name
                if path.is_dir():
                    return spec_from_loader(
                        fullname, loader=ParentFolderLoader, is_package=True)
                pypath = path.with_suffix('.py')
                if not pypath.exists():
                    raise ImportError(f'Not found base path {path.resolve()}')
                return importlib.util.spec_from_loader(
                    fullname, BaseConfigLoader(pypath, self.level + 1))
        return None

    def __enter__(self):
        # call from which file
        stack = inspect.stack()[1]
        file = inspect.getfile(stack[0])
        folder = Path(file).parent
        self.root_path = folder.joinpath(*(['..'] * Config._max_parent_depth))

        self.base_modules = []
        self.level = len(
            [p for p in sys.meta_path if isinstance(p, BaseImportContext)])

        # Disable enabled lazy loader during parsing base
        self.lazy_importers = []
        for p in sys.meta_path:
            if isinstance(p, LazyImportContext) and p.enable:
                self.lazy_importers.append(p)
                p.enable = False

        index = sys.meta_path.index(importlib.machinery.FrozenImporter)
        sys.meta_path.insert(index + 1, self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.meta_path.remove(self)
        for name in self.base_modules:
            sys.modules.pop(name, None)
        for p in self.lazy_importers:
            p.enable = True

    def __repr__(self):
        return f'<BaseImportContext (level={self.level})>'


read_base = BaseImportContext
