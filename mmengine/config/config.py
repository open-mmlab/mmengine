# Copyright (c) OpenMMLab. All rights reserved.
import ast
import copy
import os
import os.path as osp
import platform
import shutil
import tempfile
import types
import uuid
import warnings
from argparse import Action, ArgumentParser, Namespace
from collections import abc
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

from addict import Dict
from yapf.yapflib.yapf_api import FormatCode

from mmengine.fileio import dump, load
from mmengine.utils import (check_file_exist, get_installed_path,
                            import_modules_from_strings, is_installed)
from .utils import (RemoveAssignFromAST, _get_external_cfg_base_path,
                    _get_external_cfg_path, _get_package_and_cfg_path)

BASE_KEY = '_base_'
DELETE_KEY = '_delete_'
DEPRECATION_KEY = '_deprecation_'
RESERVED_KEYS = ['filename', 'text', 'pretty_text']

if platform.system() == 'Windows':
    import regex as re
else:
    import re  # type: ignore


class ConfigDict(Dict):
    """A dictionary for config which has the same interface as python's built-
    in dictionary and can be used as a normal dictionary.

    The Config class would transform the nested fields (dictionary-like fields)
    in config file into ``ConfigDict``.
    """

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no "
                                 f"attribute '{name}'")
        except Exception as e:
            raise e
        else:
            return value


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
            print(f'cannot parse key {prefix + k} of type {type(v)}')
    return parser


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
    """

    def __init__(self,
                 cfg_dict: dict = None,
                 cfg_text: Optional[str] = None,
                 filename: Optional[Union[str, Path]] = None):
        filename = str(filename) if isinstance(filename, Path) else filename
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f'{key} is reserved for config file')

        super().__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super().__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, encoding='utf-8') as f:
                text = f.read()
        else:
            text = ''
        super().__setattr__('_text', text)

    @staticmethod
    def fromfile(filename: Union[str, Path],
                 use_predefined_variables: bool = True,
                 import_custom_modules: bool = True) -> 'Config':
        """Build a Config instance from config file.

        Args:
            filename (str or Path): Name of config file.
            use_predefined_variables (bool, optional): Whether to use
                predefined variables. Defaults to True.
            import_custom_modules (bool, optional): Whether to support
                importing custom modules in config. Defaults to True.

        Returns:
            Config: Config instance built from config file.
        """
        filename = str(filename) if isinstance(filename, Path) else filename
        cfg_dict, cfg_text = Config._file2dict(filename,
                                               use_predefined_variables)
        if import_custom_modules and cfg_dict.get('custom_imports', None):
            try:
                import_modules_from_strings(**cfg_dict['custom_imports'])
            except ImportError as e:
                raise ImportError('Failed to custom import!') from e
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

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
    def _substitute_predefined_vars(filename: str, temp_config_name: str):
        """Substitute predefined variables in config with actual values.

        Sometimes we want some variables in the config to be related to the
        current path or file name, etc.

        Here is an example of a typical usage scenario. When training a model,
        we define a working directory in the config that save the models and
        logs. For different configs, we expect to define different working
        directories. A common way for users is to use the config file name
        directly as part of the working directory name, e.g. for the config
        ``config_setting1.py``, the working directory is
        ``. /work_dir/config_setting1``.

        This can be easily achieved using predefined variables, which can be
        written in the config `config_setting1.py` as follows

        .. code-block:: python

           work_dir = '. /work_dir/{{ fileBasenameNoExtension }}'


        Here `{{ fileBasenameNoExtension }}` indicates the file name of the
        config (without the extension), and when the config class reads the
        config file, it will automatically parse this double-bracketed string
        to the corresponding actual value.

        .. code-block:: python

           cfg = Config.fromfile('. /config_setting1.py')
           cfg.work_dir # ". /work_dir/config_setting1"


        For details, Please refer to docs/zh_cn/tutorials/config.md .

        Args:
            filename (str): Filename of config.
            temp_config_name (str): Temporary filename to save substituted
                config.
        """
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname)
        with open(filename, encoding='utf-8') as f:
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
            value = value.replace('\\', '/')
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, 'w', encoding='utf-8') as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def _pre_substitute_base_vars(filename: str,
                                  temp_config_name: str) -> dict:
        """Preceding step for substituting variables in base config with actual
        value.

        Args:
            filename (str): Filename of config.
            temp_config_name (str): Temporary filename to save substituted
                config.

        Returns:
            dict: A dictionary contains variables in base config.
        """
        with open(filename, encoding='utf-8') as f:
            config_file = f.read()
        base_var_dict = {}
        regexp = r'\{\{\s*' + BASE_KEY + r'\.([\w\.]+)\s*\}\}'
        base_vars = set(re.findall(regexp, config_file))
        for base_var in base_vars:
            randstr = f'_{base_var}_{uuid.uuid4().hex.lower()[:6]}'
            base_var_dict[randstr] = base_var
            regexp = r'\{\{\s*' + BASE_KEY + r'\.' + base_var + r'\s*\}\}'
            config_file = re.sub(regexp, f'"{randstr}"', config_file)
        with open(temp_config_name, 'w', encoding='utf-8') as tmp_config_file:
            tmp_config_file.write(config_file)
        return base_var_dict

    @staticmethod
    def _substitute_base_vars(cfg: Any, base_var_dict: dict,
                              base_cfg: dict) -> Any:
        """Substitute base variables from strings to their actual values.

        Args:
            Any : Config dictionary.
            base_var_dict (dict): A dictionary contains variables in base
                config.
            base_cfg (dict): Base config dictionary.

        Returns:
            Any : A dictionary with origin base variables
                substituted with actual values.
        """
        cfg = copy.deepcopy(cfg)

        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if isinstance(v, str) and v in base_var_dict:
                    new_v = base_cfg
                    for new_k in base_var_dict[v].split('.'):
                        new_v = new_v[new_k]
                    cfg[k] = new_v
                elif isinstance(v, (list, tuple, dict)):
                    cfg[k] = Config._substitute_base_vars(
                        v, base_var_dict, base_cfg)
        elif isinstance(cfg, tuple):
            cfg = tuple(
                Config._substitute_base_vars(c, base_var_dict, base_cfg)
                for c in cfg)
        elif isinstance(cfg, list):
            cfg = [
                Config._substitute_base_vars(c, base_var_dict, base_cfg)
                for c in cfg
            ]
        elif isinstance(cfg, str) and cfg in base_var_dict:
            new_v = base_cfg
            for new_k in base_var_dict[cfg].split('.'):
                new_v = new_v[new_k]
            cfg = new_v

        return cfg

    @staticmethod
    def _file2dict(filename: str,
                   use_predefined_variables: bool = True) -> Tuple[dict, str]:
        """Transform file to variables dictionary.

        Args:
            filename (str): Name of config file.
            use_predefined_variables (bool, optional): Whether to use
                predefined variables. Defaults to True.

        Returns:
            Tuple[dict, str]: Variables dictionary and text of Config.
        """
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in ['.py', '.json', '.yaml', '.yml']:
            raise OSError('Only py/yml/yaml/json type are supported now!')

        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix=fileExtname)
            if platform.system() == 'Windows':
                temp_config_file.close()

            # Substitute predefined variables
            if use_predefined_variables:
                Config._substitute_predefined_vars(filename,
                                                   temp_config_file.name)
            else:
                shutil.copyfile(filename, temp_config_file.name)
            # Substitute base variables from placeholders to strings
            base_var_dict = Config._pre_substitute_base_vars(
                temp_config_file.name, temp_config_file.name)

            # Handle base files
            base_cfg_dict = ConfigDict()
            cfg_text_list = list()
            for base_cfg_path in Config._get_base_files(temp_config_file.name):
                base_cfg_path, scope = Config._get_cfg_path(
                    base_cfg_path, filename)
                _cfg_dict, _cfg_text = Config._file2dict(base_cfg_path)
                cfg_text_list.append(_cfg_text)
                duplicate_keys = base_cfg_dict.keys() & _cfg_dict.keys()
                if len(duplicate_keys) > 0:
                    raise KeyError('Duplicate key is not allowed among bases. '
                                   f'Duplicate keys: {duplicate_keys}')

                # _dict_to_config_dict will do the following things:
                # 1. Recursively converts ``dict`` to :obj:`ConfigDict`.
                # 2. Set `_scope_` for the outer dict variable for the base
                # config.
                # 3. Set `scope` attribute for each base variable. Different
                # from `_scope_`， `scope` is not a key of base dict,
                # `scope` attribute will be parsed to key `_scope_` by
                # function `_parse_scope` only if the base variable is
                # accessed by the current config.
                _cfg_dict = Config._dict_to_config_dict(_cfg_dict, scope)
                base_cfg_dict.update(_cfg_dict)

            if filename.endswith('.py'):
                with open(temp_config_file.name) as f:
                    codes = ast.parse(f.read())
                    codes = RemoveAssignFromAST(BASE_KEY).visit(codes)
                codeobj = compile(codes, '', mode='exec')
                # Support load global variable in nested function of the
                # config.
                global_locals_var = {'_base_': base_cfg_dict}
                ori_keys = set(global_locals_var.keys())
                eval(codeobj, global_locals_var, global_locals_var)
                cfg_dict = {
                    key: value
                    for key, value in global_locals_var.items()
                    if (key not in ori_keys and not key.startswith('__'))
                }
            elif filename.endswith(('.yml', '.yaml', '.json')):
                cfg_dict = load(temp_config_file.name)
            # close temp file
            for key, value in list(cfg_dict.items()):
                if isinstance(value, (types.FunctionType, types.ModuleType)):
                    cfg_dict.pop(key)
            temp_config_file.close()

            # If the current config accesses a base variable of base
            # configs, The ``scope`` attribute of corresponding variable
            # will be converted to the `_scope_`.
            Config._parse_scope(cfg_dict)

        # check deprecation information
        if DEPRECATION_KEY in cfg_dict:
            deprecation_info = cfg_dict.pop(DEPRECATION_KEY)
            warning_msg = f'The config file {filename} will be deprecated ' \
                'in the future.'
            if 'expected' in deprecation_info:
                warning_msg += f' Please use {deprecation_info["expected"]} ' \
                    'instead.'
            if 'reference' in deprecation_info:
                warning_msg += ' More information can be found at ' \
                    f'{deprecation_info["reference"]}'
            warnings.warn(warning_msg, DeprecationWarning)

        cfg_text = filename + '\n'
        with open(filename, encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()

        # Substitute base variables from strings to their actual values
        cfg_dict = Config._substitute_base_vars(cfg_dict, base_var_dict,
                                                base_cfg_dict)
        cfg_dict.pop(BASE_KEY, None)

        cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
        cfg_dict = {
            k: v
            for k, v in cfg_dict.items() if not k.startswith('__')
        }

        # merge cfg_text
        cfg_text_list.append(cfg_text)
        cfg_text = '\n'.join(cfg_text_list)

        return cfg_dict, cfg_text

    @staticmethod
    def _dict_to_config_dict(cfg: dict,
                             scope: Optional[str] = None,
                             has_scope=True):
        """Recursively converts ``dict`` to :obj:`ConfigDict`.

        Args:
            cfg (dict): Config dict.
            scope (str, optional): Scope of instance.
            has_scope (bool): Whether to add `_scope_` key to config dict.

        Returns:
            ConfigDict: Converted dict.
        """
        # Only the outer dict with key `type` should have the key `_scope_`.
        if isinstance(cfg, dict):
            if has_scope and 'type' in cfg:
                has_scope = False
                if scope is not None and cfg.get('_scope_', None) is None:
                    cfg._scope_ = scope  # type: ignore
            cfg = ConfigDict(cfg)
            dict.__setattr__(cfg, 'scope', scope)
            for key, value in cfg.items():
                cfg[key] = Config._dict_to_config_dict(
                    value, scope=scope, has_scope=has_scope)
        elif isinstance(cfg, tuple):
            cfg = tuple(
                Config._dict_to_config_dict(_cfg, scope, has_scope=has_scope)
                for _cfg in cfg)
        elif isinstance(cfg, list):
            cfg = [
                Config._dict_to_config_dict(_cfg, scope, has_scope=has_scope)
                for _cfg in cfg
            ]
        return cfg

    @staticmethod
    def _parse_scope(cfg: dict) -> None:
        """Adds ``_scope_`` to :obj:`ConfigDict` instance, which means a base
        variable.

        If the config dict already has the scope, scope will not be
        overwritten.

        Args:
            cfg (dict): Config needs to be parsed with scope.
        """
        if isinstance(cfg, ConfigDict):
            cfg._scope_ = cfg.scope
        elif isinstance(cfg, (tuple, list)):
            [Config._parse_scope(value) for value in cfg]
        else:
            return

    @staticmethod
    def _get_base_files(filename: str) -> list:
        """Get the base config file.

        Args:
            filename (str): The config file.

        Raises:
            TypeError: Name of config file.

        Returns:
            list: A list of base config
        """
        file_format = filename.partition('.')[-1]
        if file_format == 'py':
            Config._validate_py_syntax(filename)
            with open(filename) as f:
                codes = ast.parse(f.read()).body

                def is_base_line(c):
                    return (isinstance(c, ast.Assign)
                            and isinstance(c.targets[0], ast.Name)
                            and c.targets[0].id == BASE_KEY)

                base_code = next((c for c in codes if is_base_line(c)), None)
                if base_code is not None:
                    base_code = ast.Expression(  # type: ignore
                        body=base_code.value)  # type: ignore
                    base_files = eval(compile(base_code, '', mode='eval'))
                else:
                    base_files = []
        elif file_format in ('yml', 'yaml', 'json'):
            import mmengine
            cfg_dict = mmengine.load(filename)
            base_files = cfg_dict.get(BASE_KEY, [])
        else:
            raise TypeError('The config type should be py, json, yaml or '
                            f'yml, but got {file_format}')
        base_files = base_files if isinstance(base_files,
                                              list) else [base_files]
        return base_files

    @staticmethod
    def _get_cfg_path(cfg_path: str,
                      filename: str) -> Tuple[str, Optional[str]]:
        """Get the config path from the current or external package.

        Args:
            cfg_path (str): Relative path of config.
            filename (str): The config file being parsed.

        Returns:
            Tuple[str, str or None]: Path and scope of config. If the config
            is not an external config, the scope will be `None`.
        """
        if '::' in cfg_path:
            # `cfg_path` startswith '::' means an external config path.
            # Get package name and relative config path.
            scope = cfg_path.partition('::')[0]
            package, cfg_path = _get_package_and_cfg_path(cfg_path)

            if not is_installed(package):
                raise ModuleNotFoundError(
                    f'{package} is not installed, please install {package} '
                    f'manually')

            # Get installed package path.
            package_path = get_installed_path(package)
            try:
                # Get config path from meta file.
                cfg_path = _get_external_cfg_path(package_path, cfg_path)
            except ValueError:
                # Since base config does not have a metafile, it should be
                # concatenated with package path and relative config path.
                cfg_path = _get_external_cfg_base_path(package_path, cfg_path)
            except FileNotFoundError as e:
                raise e
            return cfg_path, scope
        else:
            # Get local config path.
            cfg_dir = osp.dirname(filename)
            cfg_path = osp.join(cfg_dir, cfg_path)
            return cfg_path, None

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

        def _format_list(k, v, use_mapping=False):
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = '[\n'
                v_str += '\n'.join(
                    f'dict({_indent(_format_dict(v_), indent)}),'
                    for v_ in v).rstrip(',')
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f'{k_str}: {v_str}'
                else:
                    attr_str = f'{str(k)}={v_str}'
                attr_str = _indent(attr_str, indent) + ']'
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
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
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += '\n'.join(s)
            if use_mapping:
                r += '}'
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        # copied from setup.cfg
        yapf_style = dict(
            based_on_style='pep8',
            blank_line_before_nested_class_or_def=True,
            split_before_expression_after_opening_paren=True)
        text, _ = FormatCode(text, style_config=yapf_style, verify=True)

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

    def __getstate__(self) -> Tuple[dict, Optional[str], Optional[str]]:
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

    def __setstate__(self, state: Tuple[dict, Optional[str], Optional[str]]):
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
        options = {}
        if values is not None:
            for kv in values:
                key, val = kv.split('=', maxsplit=1)
                options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)
