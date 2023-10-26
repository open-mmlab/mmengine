# Copyright (c) OpenMMLab. All rights reserved.
import builtins
import importlib
import inspect
import os
import platform
import sys
from contextlib import contextmanager
from importlib.machinery import PathFinder
from pathlib import Path
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Optional, Tuple, Union

from yapf.yapflib.yapf_api import FormatCode

from .config import Config, ConfigDict, ConfigList, ConfigSet, ConfigTuple
from .lazy import LazyImportContext, LazyObject, recover_lazy_field

RESERVED_KEYS = ['filename', 'text', 'pretty_text']
_CFG_UID = 0

if platform.system() == 'Windows':
    import regex as re
else:
    import re  # type: ignore


def format_inpsect(obj):
    file = inspect.getsourcefile(obj)
    lines, lineno = inspect.getsourcelines(obj)
    msg = f'File "{file}", line {lineno}\n--> {lines[0]}'
    return msg


def dump_extra_type(value):
    if isinstance(value, LazyObject):
        return value.dump_str
    if isinstance(value, (type, FunctionType, BuiltinFunctionType)):
        return LazyObject(value.__name__, value.__module__).dump_str
    if isinstance(value, ModuleType):
        return LazyObject(value.__name__).dump_str

    typename = type(value).__module__ + '.' + type(value).__name__
    if typename == 'torch.dtype':
        return LazyObject(str(value)).dump_str

    return None


def filter_imports(item):
    k, v = item
    # If the name is the same as the function/type name,
    # It should come from import instead of a field
    if isinstance(v, (FunctionType, type)):
        return v.__name__ != k
    elif isinstance(v, LazyObject):
        return v.name != k
    elif isinstance(v, ModuleType):
        return False
    return True


class ConfigV2(Config):
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
    _pkg_prefix = '_mmengine_cfg'

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

        if not isinstance(cfg_dict, ConfigDict):
            cfg_dict = ConfigDict(cfg_dict)
        # Recover dumped lazy object like '<torch.nn.Linear>' from string
        cfg_dict = recover_lazy_field(cfg_dict)

        super(Config, self).__setattr__('_cfg_dict', cfg_dict)
        super(Config, self).__setattr__('_filename', filename)
        super(Config, self).__setattr__('_format_python_code',
                                        format_python_code)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, encoding='utf-8') as f:
                text = f.read()
        else:
            text = ''

        super(Config, self).__setattr__('_text', text)

        self._sanity_check(self._to_lazy_dict())

    @staticmethod
    def _sanity_check(cfg):
        if isinstance(cfg, dict):
            for v in cfg.values():
                ConfigV2._sanity_check(v)
        elif isinstance(cfg, (tuple, list, set)):
            for v in cfg:
                ConfigV2._sanity_check(v)
        elif isinstance(cfg, (type, FunctionType)):
            if (ConfigV2._pkg_prefix in cfg.__module__
                    or '__main__' in cfg.__module__):
                msg = ('You cannot use temporary functions '
                       'as the value of a field.\n\n')
                msg += format_inpsect(cfg)
                raise ValueError(msg)

    @staticmethod
    def fromfile(  # type: ignore
            filename: Union[str, Path],
            keep_imported: bool = False,
            format_python_code: bool = True) -> 'ConfigV2':
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
        with ConfigV2._lazy_context():
            module = ConfigV2._get_config_module(filename)
            module_dict = {
                k: getattr(module, k)
                for k in dir(module) if not k.startswith('__')
            }
            if not keep_imported:
                module_dict = dict(filter(filter_imports, module_dict.items()))

            cfg_dict = ConfigDict(module_dict)

            cfg = ConfigV2(
                cfg_dict,
                filename=filename,
                format_python_code=format_python_code)
            ConfigDict.lazy = False
            global _CFG_UID
            _CFG_UID = 0
            for mod in list(sys.modules):
                if mod.startswith(ConfigV2._pkg_prefix):
                    del sys.modules[mod]

        return cfg

    @staticmethod
    @contextmanager
    def _lazy_context():
        ConfigDict.lazy = True
        ConfigSet.lazy = True
        ConfigList.lazy = True
        ConfigTuple.lazy = True

        yield

        ConfigDict.lazy = False
        ConfigSet.lazy = False
        ConfigList.lazy = False
        ConfigTuple.lazy = False

    @staticmethod
    def _get_config_module(filename: Union[str, Path]):
        file = Path(filename).absolute()
        module_name = re.sub(r'\W|^(?=\d)', '_', file.stem)
        global _CFG_UID
        # Build a unique module name to avoid conflict.
        fullname = f'{ConfigV2._pkg_prefix}{_CFG_UID}_{module_name}'
        _CFG_UID += 1

        # import config file as a module
        with LazyImportContext():
            spec = importlib.util.spec_from_file_location(fullname, file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
            sys.modules[fullname] = module

        return module

    @staticmethod
    def _to_lazy_container(cfg: dict):
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
                cfg_dict[key] = ConfigV2._to_lazy_container(value)
            return cfg_dict
        if isinstance(cfg, list):
            return ConfigList(
                ConfigV2._to_lazy_container(_cfg) for _cfg in cfg)
        if isinstance(cfg, tuple):
            return ConfigTuple(
                ConfigV2._to_lazy_container(_cfg) for _cfg in cfg)
        if isinstance(cfg, set):
            return ConfigSet(ConfigV2._to_lazy_container(_cfg) for _cfg in cfg)

        return cfg

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
            else:
                dump_str = dump_extra_type(input_)
                if dump_str is not None:
                    return repr(dump_str)
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

    def __getstate__(
        self
    ) -> Tuple[dict, Optional[str], Optional[str], bool]:  # type: ignore
        return (self._cfg_dict, self._filename, self._text,
                self._format_python_code)

    def __setstate__(  # type: ignore
        self,
        state: Tuple[dict, Optional[str], Optional[str], bool],
    ):
        super(Config, self).__setattr__('_cfg_dict', state[0])
        super(Config, self).__setattr__('_filename', state[1])
        super(Config, self).__setattr__('_text', state[2])
        super(Config, self).__setattr__('_format_python_code', state[3])

    def _to_lazy_dict(self, keep_imported: bool = False) -> dict:
        """Convert config object to dictionary and filter the imported
        object."""
        res = self._cfg_dict._to_lazy_dict()

        if keep_imported:
            return res
        else:
            return dict(filter(filter_imports, res.items()))

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
            else:
                dump_str = dump_extra_type(cfg_dict)
                return dump_str if dump_str is not None else cfg_dict

        return lazy2string(_cfg_dict)


class BaseImportContext():

    def __enter__(self):
        # Disable enabled lazy loader during parsing base
        self.lazy_importers = []
        for p in sys.meta_path:
            if isinstance(p, LazyImportContext) and p.enable:
                self.lazy_importers.append(p)
                p.enable = False

        old_import = builtins.__import__

        def new_import(name, globals=None, locals=None, fromlist=(), level=0):
            # For relative import, the new import allows import from files
            # which are not in a package.
            # For absolute import, the new import will try to find the python
            # file according to the module name literally, it's used to handle
            # importing from installed packages, like
            # `mmpretrain.configs.resnet.resnet18_8xb32_in1k`.

            cur_file = None

            # Try to import the base config source file
            if level != 0 and globals is not None:
                # For relative import path
                if '__file__' in globals:
                    loc = Path(globals['__file__']).parent
                else:
                    loc = Path(os.getcwd())
                cur_file = self.find_relative_file(loc, name, level - 1)
                if not cur_file.exists():
                    raise ImportError(
                        f'Cannot find the base config "{name}" from '
                        f'{loc}: {cur_file} does not exist.')
            elif level == 0:
                # For absolute import path
                pkg, _, mod = name.partition('.')
                pkg = PathFinder.find_spec(pkg)
                if mod and pkg.submodule_search_locations:
                    loc = Path(pkg.submodule_search_locations[0])
                    cur_file = self.find_relative_file(loc, mod)
                    if not cur_file.exists():
                        raise ImportError(
                            f'Cannot find the base config "{name}": '
                            f'{cur_file} does not exist.')

            # Recover the original import during handle the base config file.
            builtins.__import__ = old_import

            if cur_file is not None:
                mod = ConfigV2._get_config_module(cur_file)

                for k in dir(mod):
                    mod.__dict__[k] = ConfigV2._to_lazy_container(
                        getattr(mod, k))
            else:
                mod = old_import(
                    name, globals, locals, fromlist=fromlist, level=level)

            builtins.__import__ = new_import

            return mod

        self.old_import = old_import
        builtins.__import__ = new_import

    def __exit__(self, exc_type, exc_val, exc_tb):
        builtins.__import__ = self.old_import
        for p in self.lazy_importers:
            p.enable = True

    @staticmethod
    def find_relative_file(loc: Path, relative_import_path, level=0):
        if level > 0:
            loc = loc.parents[level - 1]
        names = relative_import_path.lstrip('.').split('.')

        for name in names:
            loc = loc / name

        return loc.with_suffix('.py')


read_base = BaseImportContext
