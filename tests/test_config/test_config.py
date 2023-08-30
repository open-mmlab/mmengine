# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import pickle
import platform
import sys
import tempfile
from importlib import import_module
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import pytest

import mmengine
from mmengine import Config, ConfigDict, DictAction
from mmengine.config.lazy import LazyObject
from mmengine.fileio import dump, load
from mmengine.registry import MODELS, DefaultScope, Registry
from mmengine.utils import is_installed


class TestConfig:
    data_path = osp.join(osp.dirname(osp.dirname(__file__)), 'data/')

    @pytest.mark.parametrize('file_format', ['py', 'json', 'yaml'])
    def test_init(self, file_format):
        # test init Config by __init__
        cfg = Config()
        assert cfg.filename is None
        assert cfg.text == ''
        assert len(cfg) == 0
        assert cfg._cfg_dict == {}

        # test `cfg_dict` parameter
        # `cfg_dict` is either dict or None
        with pytest.raises(TypeError, match='cfg_dict must be a dict'):
            Config([0, 1])

        # test `filename` parameter
        cfg_dict = dict(
            item1=[1, 2], item2=dict(a=0), item3=True, item4='test')
        cfg_file = osp.join(
            self.data_path,
            f'config/{file_format}_config/simple_config.{file_format}')
        cfg = Config(cfg_dict, filename=cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        assert cfg.text == open(cfg_file).read()

        cfg_file = osp.join(
            self.data_path,
            f'config/{file_format}_config/test_reserved_key.{file_format}')
        # reserved keys cannot be set in config
        with pytest.raises(
                KeyError, match='filename is reserved for config '
                'file'):
            Config.fromfile(cfg_file)

    def test_fromfile(self):
        # test whether import `custom_imports` from cfg_file.
        cfg_file = osp.join(self.data_path, 'config',
                            'py_config/test_custom_import.py')
        sys.path.append(osp.join(self.data_path, 'config/py_config'))
        cfg = Config.fromfile(cfg_file, import_custom_modules=True)
        assert isinstance(cfg, Config)
        # If import successfully, os.environ[''TEST_VALUE''] will be
        # set to 'test'
        assert os.environ.pop('TEST_VALUE') == 'test'
        sys.path.pop()

        Config.fromfile(cfg_file, import_custom_modules=False)
        assert 'TEST_VALUE' not in os.environ
        sys.modules.pop('test_custom_import_module')
        with pytest.raises(
                ImportError, match='Failed to import custom modules from'):
            Config.fromfile(cfg_file, import_custom_modules=True)

    @pytest.mark.parametrize('file_format', ['py', 'json', 'yaml'])
    def test_fromstring(self, file_format):
        filename = f'{file_format}_config/simple_config.{file_format}'
        cfg_file = osp.join(self.data_path, 'config', filename)
        file_format = osp.splitext(filename)[-1]
        in_cfg = Config.fromfile(cfg_file)

        cfg_str = open(cfg_file).read()
        out_cfg = Config.fromstring(cfg_str, file_format)
        assert in_cfg._cfg_dict == out_cfg._cfg_dict

        # test pretty_text only supports py file format
        # in_cfg.pretty_text is .py format, cannot be parsed to .json
        if file_format != '.py':
            with pytest.raises(Exception):
                Config.fromstring(in_cfg.pretty_text, file_format)

        # error format
        with pytest.raises(IOError):
            Config.fromstring(cfg_str, '.xml')

    def test_magic_methods(self):
        cfg_dict = dict(
            item1=[1, 2], item2=dict(a=0), item3=True, item4='test')
        filename = 'py_config/simple_config.py'
        cfg_file = osp.join(self.data_path, 'config', filename)
        cfg = Config.fromfile(cfg_file)
        # len(cfg)
        assert len(cfg) == 4
        # cfg.keys()
        assert set(cfg.keys()) == set(cfg_dict.keys())
        assert set(cfg._cfg_dict.keys()) == set(cfg_dict.keys())
        # cfg.values()
        for value in cfg.values():
            assert value in cfg_dict.values()
        # cfg.items()
        for name, value in cfg.items():
            assert name in cfg_dict
            assert value in cfg_dict.values()
        # cfg.field
        assert cfg.item1 == cfg_dict['item1']
        assert cfg.item2 == cfg_dict['item2']
        assert cfg.item2.a == 0
        assert cfg.item3 == cfg_dict['item3']
        assert cfg.item4 == cfg_dict['item4']
        # accessing keys that do not exist will cause error
        with pytest.raises(AttributeError):
            cfg.not_exist
        # field in cfg, cfg[field], cfg.get()
        for name in ['item1', 'item2', 'item3', 'item4']:
            assert name in cfg
            assert cfg[name] == cfg_dict[name]
            assert cfg.get(name) == cfg_dict[name]
            assert cfg.get('not_exist') is None
            assert cfg.get('not_exist', 0) == 0
            # accessing keys that do not exist will cause error
            with pytest.raises(KeyError):
                cfg['not_exist']
        assert 'item1' in cfg
        assert 'not_exist' not in cfg
        # cfg.update()
        cfg.update(dict(item1=0))
        assert cfg.item1 == 0
        cfg.update(dict(item2=dict(a=1)))
        assert cfg.item2.a == 1
        # test __setattr__
        cfg = Config()
        cfg.item1 = [1, 2]
        cfg.item2 = {'a': 0}
        cfg['item5'] = {'a': {'b': None}}
        assert cfg._cfg_dict['item1'] == [1, 2]
        assert cfg.item1 == [1, 2]
        assert cfg._cfg_dict['item2'] == {'a': 0}
        assert cfg.item2.a == 0
        assert cfg._cfg_dict['item5'] == {'a': {'b': None}}
        assert cfg.item5.a.b is None

    def test_merge_from_dict(self):
        cfg_file = osp.join(self.data_path,
                            'config/py_config/simple_config.py')
        cfg = Config.fromfile(cfg_file)
        input_options = {'item2.a': 1, 'item2.b': 0.1, 'item3': False}
        cfg.merge_from_dict(input_options)
        assert cfg.item2 == dict(a=1, b=0.1)
        assert cfg.item3 is False

        cfg_file = osp.join(self.data_path,
                            'config/py_config/test_merge_from_dict.py')
        cfg = Config.fromfile(cfg_file)

        # Allow list keys
        input_options = {'item.0.a': 1, 'item.1.b': 1}
        cfg.merge_from_dict(input_options, allow_list_keys=True)
        assert cfg.item == [{'a': 1}, {'b': 1, 'c': 0}]

        # allow_list_keys is False
        input_options = {'item.0.a': 1, 'item.1.b': 1}
        with pytest.raises(TypeError):
            cfg.merge_from_dict(input_options, allow_list_keys=False)

        # Overflowed index number
        input_options = {'item.2.a': 1}
        with pytest.raises(KeyError):
            cfg.merge_from_dict(input_options, allow_list_keys=True)

    def test_diff(self):
        cfg1 = Config(dict(a=1, b=2))
        cfg2 = Config(dict(a=1, b=3))

        diff_str = \
            '--- \n\n+++ \n\n@@ -1,3 +1,3 @@\n\n a = 1\n-b = 2\n+b = 3\n \n\n'

        assert Config.diff(cfg1, cfg2) == diff_str

        cfg1_file = osp.join(self.data_path, 'config/py_config/test_diff_1.py')
        cfg1 = Config.fromfile(cfg1_file)

        cfg2_file = osp.join(self.data_path, 'config/py_config/test_diff_2.py')
        cfg2 = Config.fromfile(cfg2_file)

        assert Config.diff(cfg1, cfg2) == diff_str

    def test_auto_argparser(self):
        # Temporarily make sys.argv only has one argument and keep backups
        tmp = sys.argv[1:]
        sys.argv = sys.argv[:2]
        sys.argv[1] = osp.join(
            self.data_path,
            'config/py_config/test_merge_from_multiple_bases.py')
        parser, cfg = Config.auto_argparser()
        args = parser.parse_args()
        assert args.config == sys.argv[1]
        for key in cfg._cfg_dict.keys():
            if not isinstance(cfg[key], ConfigDict):
                assert not getattr(args, key)
        # TODO currently do not support nested keys, bool args will be
        #  overwritten by int
        sys.argv.extend(tmp)

    def test_dict_to_config_dict(self):
        cfg_dict = dict(
            a=1, b=dict(c=dict()), d=[dict(e=dict(f=(dict(g=1), [])))])
        cfg_dict = Config._dict_to_config_dict(cfg_dict)
        assert isinstance(cfg_dict, ConfigDict)
        assert isinstance(cfg_dict.a, int)
        assert isinstance(cfg_dict.b, ConfigDict)
        assert isinstance(cfg_dict.b.c, ConfigDict)
        assert isinstance(cfg_dict.d, list)
        assert isinstance(cfg_dict.d[0], ConfigDict)
        assert isinstance(cfg_dict.d[0].e, ConfigDict)
        assert isinstance(cfg_dict.d[0].e.f, tuple)
        assert isinstance(cfg_dict.d[0].e.f[0], ConfigDict)
        assert isinstance(cfg_dict.d[0].e.f[1], list)

    def test_dump(self, tmp_path):
        file_path = 'config/py_config/test_merge_from_multiple_bases.py'
        cfg_file = osp.join(self.data_path, file_path)
        cfg = Config.fromfile(cfg_file)
        dump_py = tmp_path / 'simple_config.py'

        cfg.dump(dump_py)
        assert cfg.dump() == cfg.pretty_text
        assert open(dump_py).read() == cfg.pretty_text

        # test dump json/yaml.
        file_path = 'config/json_config/simple.config.json'
        cfg_file = osp.join(self.data_path, file_path)
        cfg = Config.fromfile(cfg_file)
        dump_json = tmp_path / 'simple_config.json'
        cfg.dump(dump_json)

        with open(dump_json) as f:
            assert f.read() == cfg.dump()

        # test pickle
        file_path = 'config/py_config/test_dump_pickle_support.py'
        cfg_file = osp.join(self.data_path, file_path)
        cfg = Config.fromfile(cfg_file)

        text_cfg_filename = tmp_path / '_text_config.py'
        cfg.dump(text_cfg_filename)
        text_cfg = Config.fromfile(text_cfg_filename)
        assert text_cfg.str_item_7 == osp.join(osp.expanduser('~'), 'folder')
        assert text_cfg.str_item_8 == 'string with \tescape\\ characters\n'
        assert text_cfg._cfg_dict == cfg._cfg_dict

        cfg_file = osp.join(self.data_path,
                            'config/py_config/test_dump_pickle_support.py')
        cfg = Config.fromfile(cfg_file)

        pkl_cfg_filename = tmp_path / '_pickle.pkl'
        dump(cfg, pkl_cfg_filename)
        pkl_cfg = load(pkl_cfg_filename)
        assert pkl_cfg._cfg_dict == cfg._cfg_dict
        # Test dump config from dict.
        cfg_dict = dict(a=1, b=2)
        cfg = Config(cfg_dict)
        assert cfg.pretty_text == cfg.dump()
        # Test dump python format config.
        dump_file = tmp_path / 'dump_from_dict.py'
        cfg.dump(dump_file)
        with open(dump_file) as f:
            assert f.read() == 'a = 1\nb = 2\n'
        # Test dump json format config.
        dump_file = tmp_path / 'dump_from_dict.json'
        cfg.dump(dump_file)
        with open(dump_file) as f:
            assert f.read() == '{"a": 1, "b": 2}'
        # Test dump yaml format config.
        dump_file = tmp_path / 'dump_from_dict.yaml'
        cfg.dump(dump_file)
        with open(dump_file) as f:
            assert f.read() == 'a: 1\nb: 2\n'

    def test_pretty_text(self, tmp_path):
        cfg_file = osp.join(
            self.data_path,
            'config/py_config/test_merge_from_multiple_bases.py')
        cfg = Config.fromfile(cfg_file)
        text_cfg_filename = tmp_path / '_text_config.py'
        with open(text_cfg_filename, 'w') as f:
            f.write(cfg.pretty_text)
        text_cfg = Config.fromfile(text_cfg_filename)
        assert text_cfg._cfg_dict == cfg._cfg_dict

    def test_repr(self, tmp_path):
        cfg_file = osp.join(self.data_path,
                            'config/py_config/simple_config.py')
        cfg = Config.fromfile(cfg_file)
        tmp_txt = tmp_path / 'tmp.txt'
        with open(tmp_txt, 'w') as f:
            print(cfg, file=f)
        with open(tmp_txt) as f:
            assert f.read().strip() == f'Config (path: {cfg.filename}): ' \
                                       f'{cfg._cfg_dict.__repr__()}'

    def test_dict_action(self):
        parser = argparse.ArgumentParser(description='Train a detector')
        parser.add_argument(
            '--options', nargs='+', action=DictAction, help='custom options')
        # Nested brackets
        args = parser.parse_args(
            ['--options', 'item2.a=a,b', 'item2.b=[(a,b), [1,2], false]'])
        out_dict = {
            'item2.a': ['a', 'b'],
            'item2.b': [('a', 'b'), [1, 2], False]
        }
        assert args.options == out_dict
        # Single Nested brackets
        args = parser.parse_args(['--options', 'item2.a=[[1]]'])
        out_dict = {'item2.a': [[1]]}
        assert args.options == out_dict
        # Imbalance bracket will cause error
        with pytest.raises(AssertionError):
            parser.parse_args(['--options', 'item2.a=[(a,b), [1,2], false'])
        # Normal values
        args = parser.parse_args([
            '--options', 'item2.a=1', 'item2.b=0.1', 'item2.c=x', 'item3=false'
        ])
        out_dict = {
            'item2.a': 1,
            'item2.b': 0.1,
            'item2.c': 'x',
            'item3': False
        }
        assert args.options == out_dict
        cfg_file = osp.join(self.data_path,
                            'config/py_config/simple_config.py')
        cfg = Config.fromfile(cfg_file)
        cfg.merge_from_dict(args.options)
        assert cfg.item2 == dict(a=1, b=0.1, c='x')
        assert cfg.item3 is False

        # test multiple options
        args = parser.parse_args([
            '--options', 'item1.a=1', 'item2.a=2', '--options', 'item2.a=1',
            'item3=false'
        ])
        out_dict = {'item1.a': 1, 'item2.a': 1, 'item3': False}
        assert args.options == out_dict

    def test_validate_py_syntax(self, tmp_path):
        tmp_cfg = tmp_path / 'tmp_config.py'
        with open(tmp_cfg, 'w') as f:
            f.write('dict(a=1,b=2.c=3)')
        # Incorrect point in dict will cause error
        with pytest.raises(SyntaxError):
            Config._validate_py_syntax(tmp_cfg)
        with open(tmp_cfg, 'w') as f:
            f.write('[dict(a=1, b=2, c=(1, 2)]')
        # Imbalance bracket will cause error
        with pytest.raises(SyntaxError):
            Config._validate_py_syntax(tmp_cfg)
        with open(tmp_cfg, 'w') as f:
            f.write('dict(a=1,b=2\nc=3)')
        # Incorrect feed line in dict will cause error
        with pytest.raises(SyntaxError):
            Config._validate_py_syntax(tmp_cfg)

    def test_substitute_predefined_vars(self, tmp_path):
        cfg_text = 'a={{fileDirname}}\n' \
                   'b={{fileBasename}}\n' \
                   'c={{fileBasenameNoExtension}}\n' \
                   'd={{fileExtname}}\n'

        cfg = tmp_path / 'tmp_cfg1.py'
        substituted_cfg = tmp_path / 'tmp_cfg2.py'

        file_dirname = osp.dirname(cfg)
        file_basename = osp.basename(cfg)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(cfg)[1]

        expected_text = f'a={file_dirname}\n' \
                        f'b={file_basename}\n' \
                        f'c={file_basename_no_extension}\n' \
                        f'd={file_extname}\n'
        expected_text = expected_text.replace('\\', '/')
        with open(cfg, 'w') as f:
            f.write(cfg_text)
        Config._substitute_predefined_vars(cfg, substituted_cfg)

        with open(substituted_cfg) as f:
            assert f.read() == expected_text

    def test_substitute_environment_vars(self, tmp_path):
        cfg = tmp_path / 'tmp_cfg1.py'
        substituted_cfg = tmp_path / 'tmp_cfg2.py'

        cfg_text = 'a={{$A:}}\n'
        with open(cfg, 'w') as f:
            f.write(cfg_text)
        with pytest.raises(KeyError):
            Config._substitute_env_variables(cfg, substituted_cfg)

        os.environ['A'] = 'text_A'
        Config._substitute_env_variables(cfg, substituted_cfg)
        with open(substituted_cfg) as f:
            assert f.read() == 'a=text_A\n'
        os.environ.pop('A')

        cfg_text = 'b={{$B:80}}\n'
        with open(cfg, 'w') as f:
            f.write(cfg_text)
        Config._substitute_env_variables(cfg, substituted_cfg)
        with open(substituted_cfg) as f:
            assert f.read() == 'b=80\n'

        os.environ['B'] = '100'
        Config._substitute_env_variables(cfg, substituted_cfg)
        with open(substituted_cfg) as f:
            assert f.read() == 'b=100\n'
        os.environ.pop('B')

        cfg_text = 'c={{"$C:80"}}\n'
        with open(cfg, 'w') as f:
            f.write(cfg_text)
        Config._substitute_env_variables(cfg, substituted_cfg)
        with open(substituted_cfg) as f:
            assert f.read() == 'c=80\n'

    def test_pre_substitute_base_vars(self, tmp_path):
        cfg_path = osp.join(self.data_path, 'config',
                            'py_config/test_pre_substitute_base_vars.py')
        tmp_cfg = tmp_path / 'tmp_cfg.py'
        base_var_dict = Config._pre_substitute_base_vars(cfg_path, tmp_cfg)
        assert 'item6' in base_var_dict.values()
        assert 'item10' in base_var_dict.values()
        assert 'item11' in base_var_dict.values()
        sys.path.append(str(tmp_path))
        cfg_module_dict = import_module(tmp_cfg.name.strip('.py')).__dict__
        assert cfg_module_dict['item22'].startswith('_item11')
        assert cfg_module_dict['item23'].startswith('_item10')
        assert cfg_module_dict['item25']['c'][1].startswith('_item6')
        sys.path.pop()

        cfg_path = osp.join(self.data_path, 'config',
                            'json_config/test_base.json')
        tmp_cfg = tmp_path / 'tmp_cfg.json'
        Config._pre_substitute_base_vars(cfg_path, tmp_cfg)
        cfg_module_dict = load(tmp_cfg)
        assert cfg_module_dict['item9'].startswith('_item2')
        assert cfg_module_dict['item10'].startswith('_item7')

        cfg_path = osp.join(self.data_path, 'config',
                            'yaml_config/test_base.yaml')
        tmp_cfg = tmp_path / 'tmp_cfg.yaml'
        Config._pre_substitute_base_vars(cfg_path, tmp_cfg)
        cfg_module_dict = load(tmp_cfg)
        assert cfg_module_dict['item9'].startswith('_item2')
        assert cfg_module_dict['item10'].startswith('_item7')

    def test_substitute_base_vars(self):
        cfg = dict(
            item4='_item1.12345',
            item5=dict(item3='1', item2='_item2_.fswf'),
            item0=('_item0_.12ed21wq', 1))
        cfg_base = dict(item1=0, item2=[1, 2, 3], item0=(1, 2, 3))
        base_var_dict = {
            '_item1.12345': 'item1',
            '_item2_.fswf': 'item2',
            '_item0_.12ed21wq': 'item0'
        }
        cfg = Config._substitute_base_vars(cfg, base_var_dict, cfg_base)
        assert cfg['item4'] == cfg_base['item1']
        assert cfg['item5']['item2'] == cfg_base['item2']

    def test_file2dict(self, tmp_path):

        # test error format config
        tmp_cfg = tmp_path / 'tmp_cfg.xml'
        tmp_cfg.write_text('exist')
        # invalid config format
        with pytest.raises(IOError):
            Config.fromfile(tmp_cfg)
        # invalid config file path
        with pytest.raises(FileNotFoundError):
            Config.fromfile('no_such_file.py')

        self._simple_load()
        self._predefined_vars()
        self._environment_vars()
        self._base_variables()
        self._merge_from_base()
        self._code_in_config()
        self._merge_from_multiple_bases()
        self._merge_delete()
        self._merge_intermediate_variable()
        self._merge_recursive_bases()
        self._deprecation()

    def test_get_cfg_path_local(self):
        filename = 'py_config/simple_config.py'
        filename = osp.join(self.data_path, 'config', filename)
        cfg_name = './base.py'
        cfg_path, scope = Config._get_cfg_path(cfg_name, filename)
        assert scope is None
        osp.isfile(cfg_path)

    @pytest.mark.skipif(
        not is_installed('mmdet') or not is_installed('mmcls'),
        reason='mmdet and mmcls should be installed')
    def test_get_cfg_path_external(self):
        filename = 'py_config/simple_config.py'
        filename = osp.join(self.data_path, 'config', filename)

        cfg_name = 'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
        cfg_path, scope = Config._get_cfg_path(cfg_name, filename)
        assert scope == 'mmdet'
        osp.isfile(cfg_path)

        cfg_name = 'mmcls::cspnet/cspresnet50_8xb32_in1k.py'
        cfg_path, scope = Config._get_cfg_path(cfg_name, filename)
        assert scope == 'mmcls'
        osp.isfile(cfg_path)

    def _simple_load(self):
        # test load simple config
        for file_format in ['py', 'json', 'yaml']:
            for name in ['simple.config', 'simple_config']:
                filename = f'{file_format}_config/{name}.{file_format}'

                cfg_file = osp.join(self.data_path, 'config', filename)
                cfg_dict, cfg_text, env_variables = Config._file2dict(cfg_file)
                assert isinstance(cfg_text, str)
                assert isinstance(cfg_dict, dict)
                assert isinstance(env_variables, dict)

    def _get_file_path(self, file_path):
        if platform.system() == 'Windows':
            return file_path.replace('\\', '/')
        else:
            return file_path

    def _predefined_vars(self):
        # test parse predefined_var in config
        cfg_file = osp.join(self.data_path,
                            'config/py_config/test_predefined_var.py')
        path = osp.join(self.data_path, 'config/py_config')

        path = Path(path).as_posix()
        cfg_dict_dst = dict(
            item1='test_predefined_var.py',
            item2=path,
            item3='abc_test_predefined_var')

        assert Config._file2dict(cfg_file)[0]['item1'] == cfg_dict_dst['item1']
        assert Config._file2dict(cfg_file)[0]['item2'] == cfg_dict_dst['item2']
        assert Config._file2dict(cfg_file)[0]['item3'] == cfg_dict_dst['item3']

        # test `use_predefined_variable=False`
        cfg_dict_ori = dict(
            item1='{{fileBasename}}',
            item2='{{ fileDirname}}',
            item3='abc_{{ fileBasenameNoExtension }}')

        assert Config._file2dict(cfg_file,
                                 False)[0]['item1'] == cfg_dict_ori['item1']
        assert Config._file2dict(cfg_file,
                                 False)[0]['item2'] == cfg_dict_ori['item2']
        assert Config._file2dict(cfg_file,
                                 False)[0]['item3'] == cfg_dict_ori['item3']

        # test test_predefined_var.yaml
        cfg_file = osp.join(self.data_path,
                            'config/yaml_config/test_predefined_var.yaml')

        # test `use_predefined_variable=False`
        assert Config._file2dict(cfg_file,
                                 False)[0]['item1'] == '{{ fileDirname }}'
        assert Config._file2dict(cfg_file)[0]['item1'] == self._get_file_path(
            osp.dirname(cfg_file))

        # test test_predefined_var.json
        cfg_file = osp.join(self.data_path,
                            'config/json_config/test_predefined_var.json')

        assert Config.fromfile(cfg_file, False)['item1'] == '{{ fileDirname }}'
        assert Config.fromfile(cfg_file)['item1'] == self._get_file_path(
            osp.dirname(cfg_file))

    def _environment_vars(self):
        # test parse predefined_var in config
        cfg_file = osp.join(self.data_path,
                            'config/py_config/test_environment_var.py')

        with pytest.raises(KeyError):
            Config._file2dict(cfg_file)

        os.environ['ITEM1'] = '60'
        cfg_dict_dst = dict(item1='60', item2='default_value', item3=80)
        assert Config._file2dict(cfg_file)[0]['item1'] == cfg_dict_dst['item1']
        assert Config._file2dict(cfg_file)[0]['item2'] == cfg_dict_dst['item2']
        assert Config._file2dict(cfg_file)[0]['item3'] == cfg_dict_dst['item3']

        os.environ['ITEM2'] = 'new_value'
        os.environ['ITEM3'] = '50'
        cfg_dict_dst = dict(item1='60', item2='new_value', item3=50)
        assert Config._file2dict(cfg_file)[0]['item1'] == cfg_dict_dst['item1']
        assert Config._file2dict(cfg_file)[0]['item2'] == cfg_dict_dst['item2']
        assert Config._file2dict(cfg_file)[0]['item3'] == cfg_dict_dst['item3']

        os.environ.pop('ITEM1')
        os.environ.pop('ITEM2')
        os.environ.pop('ITEM3')

    def _merge_from_base(self):
        cfg_file = osp.join(self.data_path,
                            'config/py_config/test_merge_from_base_single.py')
        cfg_dict = Config._file2dict(cfg_file)[0]

        assert cfg_dict['item1'] == [2, 3]
        assert cfg_dict['item2']['a'] == 1
        assert cfg_dict['item3'] is False
        assert cfg_dict['item4'] == 'test_base'
        # item3 is a dict in the child config but a boolean in base config
        with pytest.raises(TypeError):
            Config.fromfile(
                osp.join(self.data_path,
                         'config/py_config/test_merge_from_base_error.py'))

    def _merge_from_multiple_bases(self):
        cfg_file = osp.join(
            self.data_path,
            'config/py_config/test_merge_from_multiple_bases.py')
        cfg_dict = Config._file2dict(cfg_file)[0]

        # cfg.fcfg_dictd
        assert cfg_dict['item1'] == [1, 2]
        assert cfg_dict['item2']['a'] == 0
        assert cfg_dict['item3'] is False
        assert cfg_dict['item4'] == 'test'
        assert cfg_dict['item5'] == dict(a=0, b=1)
        assert cfg_dict['item6'] == [dict(a=0), dict(b=1)]
        assert cfg_dict['item7'] == dict(
            a=[0, 1, 2], b=dict(c=[3.1, 4.2, 5.3]))
        # Redefine key
        with pytest.raises(KeyError):
            Config.fromfile(
                osp.join(self.data_path,
                         'config/py_config/test_merge_from_multiple_error.py'))

    def _base_variables(self):
        for file in [
                'py_config/test_base_variables.py',
                'json_config/test_base.json', 'yaml_config/test_base.yaml'
        ]:
            cfg_file = osp.join(self.data_path, 'config', file)
            cfg_dict = Config._file2dict(cfg_file)[0]

            assert cfg_dict['item1'] == [1, 2]
            assert cfg_dict['item2']['a'] == 0
            assert cfg_dict['item3'] is False
            assert cfg_dict['item4'] == 'test'
            assert cfg_dict['item5'] == dict(a=0, b=1)
            assert cfg_dict['item6'] == [dict(a=0), dict(b=1)]
            assert cfg_dict['item7'] == dict(
                a=[0, 1, 2], b=dict(c=[3.1, 4.2, 5.3]))
            assert cfg_dict['item8'] == file.split('/')[-1]
            assert cfg_dict['item9'] == dict(a=0)
            assert cfg_dict['item10'] == [3.1, 4.2, 5.3]

        # test nested base
        for file in [
                'py_config/test_base_variables_nested.py',
                'json_config/test_base_variables_nested.json',
                'yaml_config/test_base_variables_nested.yaml'
        ]:
            cfg_file = osp.join(self.data_path, 'config', file)
            cfg_dict = Config._file2dict(cfg_file)[0]

            assert cfg_dict['base'] == '_base_.item8'
            assert cfg_dict['item1'] == [1, 2]
            assert cfg_dict['item2']['a'] == 0
            assert cfg_dict['item3'] is False
            assert cfg_dict['item4'] == 'test'
            assert cfg_dict['item5'] == dict(a=0, b=1)
            assert cfg_dict['item6'] == [dict(a=0), dict(b=1)]
            assert cfg_dict['item7'] == dict(
                a=[0, 1, 2], b=dict(c=[3.1, 4.2, 5.3]))
            assert cfg_dict['item8'] == 'test_base_variables.py'
            assert cfg_dict['item9'] == dict(a=0)
            assert cfg_dict['item10'] == [3.1, 4.2, 5.3]
            assert cfg_dict['item11'] == 'test_base_variables.py'
            assert cfg_dict['item12'] == dict(a=0)
            assert cfg_dict['item13'] == [3.1, 4.2, 5.3]
            assert cfg_dict['item14'] == [1, 2]
            assert cfg_dict['item15'] == dict(
                a=dict(b=dict(a=0)),
                b=[False],
                c=['test'],
                d=[[{
                    'e': 0
                }], [{
                    'a': 0
                }, {
                    'b': 1
                }]],
                e=[1, 2])

        # test reference assignment for py
        cfg_file = osp.join(
            self.data_path,
            'config/py_config/test_pre_substitute_base_vars.py')
        cfg_dict = Config._file2dict(cfg_file)[0]

        assert cfg_dict['item21'] == 'test_base_variables.py'
        assert cfg_dict['item22'] == 'test_base_variables.py'
        assert cfg_dict['item23'] == [3.1, 4.2, 5.3]
        assert cfg_dict['item24'] == [3.1, 4.2, 5.3]
        assert cfg_dict['item25'] == dict(
            a=dict(b=[3.1, 4.2, 5.3]),
            b=[[3.1, 4.2, 5.3]],
            c=[[{
                'e': 'test_base_variables.py'
            }], [{
                'a': 0
            }, {
                'b': 1
            }]],
            e='test_base_variables.py')

        cfg_file = osp.join(self.data_path, 'config/py_config/test_py_base.py')
        cfg = Config.fromfile(cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        # cfg.field
        assert cfg.item1 == [1, 2]
        assert cfg.item2.a == 0
        assert cfg.item2.b == [5, 6]
        assert cfg.item3 is False
        assert cfg.item4 == 'test'
        assert cfg.item5 == dict(a=0, b=1)
        assert cfg.item6 == [dict(c=0), dict(b=1)]
        assert cfg.item7 == dict(a=[0, 1, 2], b=dict(c=[3.1, 4.2, 5.3]))
        assert cfg.item8 == 'test_py_base.py'
        assert cfg.item9 == 3.1
        assert cfg.item10 == 4.2
        assert cfg.item11 == 5.3

        # test nested base
        cfg_file = osp.join(self.data_path,
                            'config/py_config/test_py_nested_path.py')
        cfg = Config.fromfile(cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        # cfg.field
        assert cfg.item1 == [1, 2]
        assert cfg.item2.a == 0
        assert cfg.item2.b == [5, 6]
        assert cfg.item3 is False
        assert cfg.item4 == 'test'
        assert cfg.item5 == dict(a=0, b=1)
        assert cfg.item6 == [dict(c=0), dict(b=1)]
        assert cfg.item7 == dict(a=[0, 1, 2], b=dict(c=[3.1, 4.2, 5.3]))
        assert cfg.item8 == 'test_py_base.py'
        assert cfg.item9 == 3.1
        assert cfg.item10 == 4.2
        assert cfg.item11 == 5.3
        assert cfg.item12 == 'test_py_base.py'
        assert cfg.item13 == 3.1
        assert cfg.item14 == [1, 2]
        assert cfg.item15 == dict(
            a=dict(b=dict(a=0, b=[5, 6])),
            b=[False],
            c=['test'],
            d=[[{
                'e': 0
            }], [{
                'c': 0
            }, {
                'b': 1
            }]],
            e=[1, 2])

        # Test use global variable in config function
        cfg_file = osp.join(self.data_path,
                            'config/py_config/test_py_function_global_var.py')
        cfg = Config._file2dict(cfg_file)[0]
        assert cfg['item1'] == 1
        assert cfg['item2'] == 2

        # Test support modifying the value of dict without defining base
        # config.
        cfg_file = osp.join(self.data_path,
                            'config/py_config/test_py_modify_key.py')
        cfg = Config._file2dict(cfg_file)[0]
        assert cfg == dict(item1=dict(a=1))

        # Simulate the case that the temporary directory includes `.`, etc.
        # /tmp/test.axsgr12/. This patch is to check the issue
        # https://github.com/open-mmlab/mmengine/issues/788 has been solved.
        class PatchedTempDirectory(tempfile.TemporaryDirectory):

            def __init__(self, *args, prefix='test.', **kwargs):
                super().__init__(*args, prefix=prefix, **kwargs)

        with patch('mmengine.config.config.tempfile.TemporaryDirectory',
                   PatchedTempDirectory):
            cfg_file = osp.join(self.data_path,
                                'config/py_config/test_py_modify_key.py')
            cfg = Config._file2dict(cfg_file)[0]
            assert cfg == dict(item1=dict(a=1))

    def _merge_recursive_bases(self):
        cfg_file = osp.join(self.data_path,
                            'config/py_config/test_merge_recursive_bases.py')
        cfg_dict = Config._file2dict(cfg_file)[0]

        assert cfg_dict['item1'] == [2, 3]
        assert cfg_dict['item2']['a'] == 1
        assert cfg_dict['item3'] is False
        assert cfg_dict['item4'] == 'test_recursive_bases'

    def _merge_delete(self):
        cfg_file = osp.join(self.data_path,
                            'config/py_config/test_merge_delete.py')
        cfg_dict = Config._file2dict(cfg_file)[0]
        # cfg.field
        assert cfg_dict['item1'] == dict(a=0)
        assert cfg_dict['item2'] == dict(a=0, b=0)
        assert cfg_dict['item3'] is True
        assert cfg_dict['item4'] == 'test'
        assert '_delete_' not in cfg_dict['item1']

        assert type(cfg_dict['item1']) == ConfigDict
        assert type(cfg_dict['item2']) == ConfigDict

    def _merge_intermediate_variable(self):

        cfg_file = osp.join(
            self.data_path,
            'config/py_config/test_merge_intermediate_variable_child.py')
        cfg_dict = Config._file2dict(cfg_file)[0]
        # cfg.field
        assert cfg_dict['item1'] == [1, 2]
        assert cfg_dict['item2'] == dict(a=0)
        assert cfg_dict['item3'] is True
        assert cfg_dict['item4'] == 'test'
        assert cfg_dict['item_cfg'] == dict(b=2)
        assert cfg_dict['item5'] == dict(cfg=dict(b=1))
        assert cfg_dict['item6'] == dict(cfg=dict(b=2))

    def _code_in_config(self):
        cfg_file = osp.join(self.data_path,
                            'config/py_config/test_code_in_config.py')
        cfg = Config.fromfile(cfg_file)
        # cfg.field
        assert cfg.cfg.item1 == [1, 2]
        assert cfg.cfg.item2 == dict(a=0)
        assert cfg.cfg.item3 is True
        assert cfg.cfg.item4 == 'test'
        assert cfg.item5 == 1

    def _deprecation(self):
        deprecated_cfg_files = [
            osp.join(self.data_path, 'config', 'py_config/test_deprecated.py'),
            osp.join(self.data_path, 'config',
                     'py_config/test_deprecated_base.py')
        ]

        for cfg_file in deprecated_cfg_files:
            with pytest.warns(DeprecationWarning):
                cfg = Config.fromfile(cfg_file)
            assert cfg.item1 == [1, 2]

    def test_deepcopy(self):
        cfg_file = osp.join(self.data_path, 'config',
                            'py_config/test_dump_pickle_support.py')
        cfg = Config.fromfile(cfg_file)
        new_cfg = copy.deepcopy(cfg)

        assert isinstance(new_cfg, Config)
        assert new_cfg._cfg_dict == cfg._cfg_dict
        assert new_cfg._cfg_dict is not cfg._cfg_dict
        assert new_cfg._filename == cfg._filename
        assert new_cfg._text == cfg._text

    def test_copy(self):
        cfg_file = osp.join(self.data_path, 'config',
                            'py_config/test_dump_pickle_support.py')
        cfg = Config.fromfile(cfg_file)
        new_cfg = copy.copy(cfg)

        assert isinstance(new_cfg, Config)
        assert new_cfg._cfg_dict == cfg._cfg_dict
        assert new_cfg._filename == cfg._filename
        assert new_cfg._text == cfg._text

        new_cfg = cfg.copy()
        assert isinstance(new_cfg, Config)
        assert new_cfg._cfg_dict == cfg._cfg_dict
        assert new_cfg._filename == cfg._filename
        assert new_cfg._text == cfg._text

    @pytest.mark.skipif(
        not is_installed('mmdet'), reason='mmdet should be installed')
    def test_get_external_cfg(self):
        ext_cfg_path = osp.join(self.data_path,
                                'config/py_config/test_get_external_cfg.py')
        ext_cfg = Config.fromfile(ext_cfg_path)
        assert ext_cfg._cfg_dict.model.neck == dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
        )
        assert '_scope_' in ext_cfg._cfg_dict.model

    @pytest.mark.skipif(
        not is_installed('mmdet'), reason='mmdet should be installed')
    def test_build_external_package(self):
        # Test load base config.
        ext_cfg_path = osp.join(self.data_path,
                                'config/py_config/test_get_external_cfg.py')
        ext_cfg = Config.fromfile(ext_cfg_path)

        LOCAL_MODELS = Registry('local_model', parent=MODELS, scope='test')
        LOCAL_MODELS.build(ext_cfg.model)

        # Test load non-base config
        ext_cfg_path = osp.join(self.data_path,
                                'config/py_config/test_get_external_cfg2.py')
        ext_cfg = Config.fromfile(ext_cfg_path)
        LOCAL_MODELS.build(ext_cfg.model)

        # Test override base variable.
        ext_cfg_path = osp.join(self.data_path,
                                'config/py_config/test_get_external_cfg3.py')
        ext_cfg = Config.fromfile(ext_cfg_path)

        @LOCAL_MODELS.register_module()
        class ToyLoss:
            pass

        @LOCAL_MODELS.register_module()
        class ToyModel:
            pass

        DefaultScope.get_instance('test1', scope_name='test')
        assert ext_cfg.model._scope_ == 'mmdet'
        model = LOCAL_MODELS.build(ext_cfg.model)

        # Local base config should not have scope.
        assert '_scope_' not in ext_cfg.toy_model
        toy_model = LOCAL_MODELS.build(ext_cfg.toy_model)
        assert isinstance(toy_model, ToyModel)
        assert model.backbone.style == 'pytorch'
        assert isinstance(model.roi_head.bbox_head.loss_cls, ToyLoss)
        DefaultScope._instance_dict.pop('test1')

    def test_pickle(self):
        # Text style config
        cfg_path = osp.join(self.data_path, 'config/py_config/test_py_base.py')
        cfg = Config.fromfile(cfg_path)
        pickled = pickle.loads(pickle.dumps(cfg))
        assert pickled.__dict__ == cfg.__dict__

        cfg_path = osp.join(self.data_path,
                            'config/lazy_module_config/toy_model.py')
        cfg = Config.fromfile(cfg_path)
        pickled = pickle.loads(pickle.dumps(cfg))
        assert pickled.__dict__ == cfg.__dict__

    def test_lazy_import(self, tmp_path):
        lazy_import_cfg_path = osp.join(
            self.data_path, 'config/lazy_module_config/toy_model.py')
        cfg = Config.fromfile(lazy_import_cfg_path)
        cfg_dict = cfg.to_dict()
        assert (cfg_dict['train_dataloader']['dataset']['type'] ==
                'mmengine.testing.runner_test_case.ToyDataset')
        assert (
            cfg_dict['custom_hooks'][0]['type'] == 'mmengine.hooks.EMAHook')
        # Dumped config
        dumped_cfg_path = tmp_path / 'test_dump_lazy.py'
        cfg.dump(dumped_cfg_path)
        dumped_cfg = Config.fromfile(dumped_cfg_path)

        copied_cfg_path = tmp_path / 'test_dump_copied_lazy.py'
        cfg_copy = cfg.copy()
        cfg_copy.dump(copied_cfg_path)
        copied_cfg = Config.fromfile(copied_cfg_path)

        def _compare_dict(a, b):
            if isinstance(a, dict):
                assert len(a) == len(b)
                for k, v in a.items():
                    _compare_dict(v, b[k])
            elif isinstance(a, list):
                assert len(a) == len(b)
                for item_a, item_b in zip(a, b):
                    _compare_dict(item_a, item_b)
            else:
                assert str(a) == str(b)

        _compare_dict(cfg.to_dict(), dumped_cfg.to_dict())
        _compare_dict(cfg.to_dict(), copied_cfg.to_dict())

        # TODO reimplement this part of unit test when mmdetection adds the
        # new config.
        # if find_spec('mmdet') is not None:
        #     cfg = Config.fromfile(
        #         osp.join(self.data_path,
        #                  'config/lazy_module_config/load_mmdet_config.py'))
        #     assert cfg.model.backbone.depth == 101
        #     cfg.work_dir = str(tmp_path)
        # else:
        #     pytest.skip('skip testing loading config from mmdet since mmdet '
        #                 'is not installed or mmdet version is too low')

        # catch import error correctly
        error_obj = tmp_path / 'error_obj.py'
        error_obj.write_text("""from mmengine.fileio import error_obj""")
        # match pattern should be double escaped
        match = str(error_obj).encode('unicode_escape').decode()
        with pytest.raises(ImportError, match=match):
            cfg = Config.fromfile(str(error_obj))
            cfg.error_obj

        error_attr = tmp_path / 'error_attr.py'
        error_attr.write_text("""
import mmengine
error_attr = mmengine.error_attr
""")  # noqa: E122
        match = str(error_attr).encode('unicode_escape').decode()
        with pytest.raises(ImportError, match=match):
            cfg = Config.fromfile(str(error_attr))
            cfg.error_attr

        error_module = tmp_path / 'error_module.py'
        error_module.write_text("""import error_module""")
        match = str(error_module).encode('unicode_escape').decode()
        with pytest.raises(ImportError, match=match):
            cfg = Config.fromfile(str(error_module))
            cfg.error_module

        # lazy-import and non-lazy-import should not be used mixed.
        # current text config, base lazy-import config
        with pytest.raises(RuntimeError, match='with read_base()'):
            Config.fromfile(
                osp.join(self.data_path,
                         'config/lazy_module_config/error_mix_using1.py'))

        # Force to import in non-lazy-import mode
        Config.fromfile(
            osp.join(self.data_path,
                     'config/lazy_module_config/error_mix_using1.py'),
            lazy_import=False)

        # current lazy-import config, base text config
        with pytest.raises(RuntimeError, match='_base_ ='):
            Config.fromfile(
                osp.join(self.data_path,
                         'config/lazy_module_config/error_mix_using2.py'))

        cfg = Config.fromfile(
            osp.join(self.data_path,
                     'config/lazy_module_config/test_mix_builtin.py'))
        assert cfg.path == osp.join('a', 'b')
        assert cfg.name == 'a/b'
        assert cfg.suffix == '.py'
        assert cfg.chained == [1, 2, 3, 4]
        assert cfg.existed
        assert cfg.cfgname == 'test_mix_builtin.py'

        cfg_dict = cfg.to_dict()
        dumped_cfg_path = tmp_path / 'test_dump_lazy.py'
        cfg.dump(dumped_cfg_path)
        dumped_cfg = Config.fromfile(dumped_cfg_path)

        assert set(dumped_cfg.keys()) == {
            'path', 'name', 'suffix', 'chained', 'existed', 'cfgname'
        }
        assert dumped_cfg.to_dict() == cfg.to_dict()


class TestConfigDict(TestCase):

    def test_keep_custom_dict(self):

        class CustomDict(dict):
            ...

        cfg_dict = ConfigDict(dict(a=CustomDict(b=1)))
        self.assertIsInstance(cfg_dict.a, CustomDict)
        self.assertIsInstance(cfg_dict['a'], CustomDict)
        self.assertIsInstance(cfg_dict.values()[0], CustomDict)
        self.assertIsInstance(cfg_dict.items()[0][1], CustomDict)

    def test_build_lazy(self):
        # This unit test are divide into two parts:
        # I. ConfigDict will never return a `LazyObject` instance. Only the
        #    built will be returned. The `LazyObject` can be accessed after
        #    `to_dict` is called.

        # II. LazyObject will always be kept in the ConfigDict no matter what
        #    operation is performed, such as ``update``, ``setitem``, or
        #    building another ConfigDict from the current one. The updated
        #    ConfigDict also follow the rule of Part I

        # Part I
        # Keep key-value the same
        raw = dict(a=1, b=dict(c=2, e=[dict(f=(2, ))]))
        cfg_dict = ConfigDict(raw)

        assert len(cfg_dict) == 2
        assert len(cfg_dict.items()) == 2
        assert len(cfg_dict.keys()) == 2
        assert len(cfg_dict.values()) == 2

        self.assertDictEqual(cfg_dict, raw)

        # Check `items` and `values` will only return the build object
        raw = dict(
            a=LazyObject('mmengine'),
            b=dict(
                c=2,
                e=[
                    dict(
                        f=dict(h=LazyObject('mmengine')),
                        g=LazyObject('mmengine'))
                ]))
        cfg_dict = ConfigDict(raw)
        # check `items` and values
        self.assertDictEqual(cfg_dict._to_lazy_dict(), raw)
        self._check(cfg_dict)

        # check getattr
        self.assertIs(cfg_dict.a, mmengine)
        self.assertIs(cfg_dict.b.e[0].f.h, mmengine)
        self.assertIs(cfg_dict.b.e[0].g, mmengine)

        # check get
        self.assertIs(cfg_dict.get('a'), mmengine)
        self.assertIs(
            cfg_dict.get('b').get('e')[0].get('f').get('h'), mmengine)
        self.assertIs(cfg_dict.get('b').get('e')[0].get('g'), mmengine)

        # check pop
        a = cfg_dict.pop('a')
        b = cfg_dict.pop('b')
        e = b.pop('e')
        h = e[0].pop('f')['h']
        g = e[0].pop('g')
        self.assertIs(a, mmengine)
        self.assertIs(h, mmengine)
        self.assertIs(g, mmengine)
        self.assertEqual(cfg_dict, {})
        self.assertEqual(b, {'c': 2})

        # Part II
        # check update with dict and ConfigDict
        for dict_type in (dict, ConfigDict):
            cfg_dict = ConfigDict(x=LazyObject('mmengine'))
            cfg_dict.update(dict_type(raw))
            self._check(cfg_dict)

        # Create a new ConfigDict
        new_dict = ConfigDict(cfg_dict)
        self._check(new_dict)

        # Update the ConfigDict by __setitem__ and __setattr__
        new_dict['b']['h'] = LazyObject('mmengine')
        new_dict['b']['k'] = dict(l=dict(n=LazyObject('mmengine')))
        new_dict.b.e[0].i = LazyObject('mmengine')
        new_dict.b.e[0].j = dict(l=dict(n=LazyObject('mmengine')))
        self._check(new_dict)

    def _check(self, cfg_dict):
        self._recursive_check_lazy(cfg_dict,
                                   lambda x: not isinstance(x, LazyObject))
        self._recursive_check_lazy(cfg_dict._to_lazy_dict(),
                                   lambda x: x is not mmengine)
        self._recursive_check_lazy(
            cfg_dict._to_lazy_dict(), lambda x: not isinstance(x, ConfigDict)
            if isinstance(x, dict) else True)
        self._recursive_check_lazy(
            cfg_dict, lambda x: isinstance(x, ConfigDict)
            if isinstance(x, dict) else True)

    def _recursive_check_lazy(self, cfg, expr):
        if isinstance(cfg, dict):
            {
                key: self._recursive_check_lazy(value, expr)
                for key, value in cfg.items()
            }
            [self._recursive_check_lazy(value, expr) for value in cfg.values()]
        elif isinstance(cfg, (tuple, list)):
            [self._recursive_check_lazy(value, expr) for value in cfg]
        else:
            self.assertTrue(expr(cfg))
