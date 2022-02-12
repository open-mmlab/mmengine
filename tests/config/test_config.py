# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp
import platform
import sys
from importlib import import_module
from pathlib import Path

import pytest
import yaml

from mmengine.config import Config, ConfigDict, DictAction, dump, load


class TestConfig:
    data_path = osp.join(osp.dirname(osp.dirname(__file__)), 'tests_data/')

    def test_init(self, tmp_path):
        # test init Config by __init__
        cfg = Config()
        assert cfg.filename is None
        assert cfg.text == ''
        assert len(cfg) == 0
        assert cfg._cfg_dict == {}

        with pytest.raises(TypeError):
            Config([0, 1])

        cfg_dict = dict(
            item1=[1, 2], item2=dict(a=0), item3=True, item4='test')
        # test simple_config.py
        cfg_file = osp.join(self.data_path,
                            'config/py_config/simple_config.py')
        cfg = Config(cfg_dict, filename=cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        assert cfg.text == open(cfg_file, 'r').read()

        # test simple_config.json
        cfg_file = osp.join(self.data_path,
                            'config/json_config/simple_config.json')
        cfg = Config(cfg_dict, filename=cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        assert cfg.text == open(cfg_file, 'r').read()

        # test simple_config.yaml
        cfg_file = osp.join(self.data_path,
                            'config/yml_config/simple_config.yaml')
        cfg = Config(cfg_dict, filename=cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        assert cfg.text == open(cfg_file, 'r').read()

        dump_file = tmp_path / 'simple_config.yaml'
        cfg.dump(dump_file)
        assert cfg.dump() == open(dump_file, 'r').read()
        assert Config.fromfile(dump_file)

    def test_validate_py_syntax(self, tmp_path):
        tmp_cfg = tmp_path / 'tmp_config.py'
        with open(tmp_cfg, 'w') as f:
            f.write('dict(a=1,b=2.c=3)')
        with pytest.raises(SyntaxError):
            Config._validate_py_syntax(tmp_cfg)

    def test_substitute_predefined_vars(self, tmp_path):
        ori = 'a={{fileDirname}}\n' \
              'b={{fileBasename}}\n' \
              'c={{fileBasenameNoExtension}}\n' \
              'd={{fileExtname}}\n'

        tmp_cfg1 = tmp_path / 'tmp_cfg1.py'
        tmp_cfg2 = tmp_path / 'tmp_cfg2.py'

        file_dirname = osp.dirname(tmp_cfg1)
        file_basename = osp.basename(tmp_cfg1)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(tmp_cfg1)[1]

        if platform.system() == 'Windows':
            file_dirname = file_dirname.replace('\\', '/')

        target = f'a={file_dirname}\n' \
                 f'b={file_basename}\n' \
                 f'c={file_basename_no_extension}\n' \
                 f'd={file_extname}\n'

        with open(tmp_cfg1, 'w') as f:
            f.write(ori)
        Config._substitute_predefined_vars(tmp_cfg1, tmp_cfg2)

        with open(tmp_cfg2, 'r') as f:
            assert f.read() == target

    def test_pre_substitute_base_vars(self, tmp_path):
        cfg_path = osp.join(self.data_path, 'config',
                            'py_config/basevar_assign.py')
        tmp_cfg = tmp_path / 'tmp_cfg.py'
        Config._pre_substitute_base_vars(cfg_path, tmp_cfg)
        sys.path.append(str(tmp_path))
        cfg_module_dict = import_module(tmp_cfg.name.strip('.py')).__dict__
        assert cfg_module_dict['item22'].startswith('_item11')
        assert cfg_module_dict['item23'].startswith('_item10')
        assert cfg_module_dict['item25']['c'][1].startswith('_item6')
        sys.path.pop()

        cfg_path = osp.join(self.data_path, 'config',
                            'json_config/use_base_var.json')
        tmp_cfg = tmp_path / 'tmp_cfg.json'
        Config._pre_substitute_base_vars(cfg_path, tmp_cfg)
        cfg_module_dict = load(tmp_cfg)
        assert cfg_module_dict['item9'].startswith('_item2')
        assert cfg_module_dict['item10'].startswith('_item7')

        cfg_path = osp.join(self.data_path, 'config',
                            'yml_config/use_base_var.yaml')
        tmp_cfg = tmp_path / 'tmp_cfg.yaml'
        Config._pre_substitute_base_vars(cfg_path, tmp_cfg)
        cfg_module_dict = load(tmp_cfg)
        assert cfg_module_dict['item9'].startswith('_item2')
        assert cfg_module_dict['item10'].startswith('_item7')

    def test_substitute_base_vars(self, tmp_path):
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

    def test_property(self):
        cfg_file = osp.join(self.data_path,
                            'config/py_config/predefined_var.py')
        path = osp.join(osp.dirname(__file__), 'data', 'config')
        path = Path(path).as_posix()
        cfg_dict = dict(item1='predefined_var.py', item2=path, item3='abc_h')
        cfg = Config(cfg_dict, filename=cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        assert cfg.text == open(cfg_file, 'r').read()
        assert cfg.dump() == cfg.pretty_text

    def test_fromfile(self, tmp_path):
        for filename in [
                'py_config/simple_config.py', 'py_config/simple.config.py',
                'json_config/simple_config.json',
                'yml_config/simple_config.yaml'
        ]:
            cfg_file = osp.join(self.data_path, 'config', filename)
            cfg = Config.fromfile(cfg_file)
            assert isinstance(cfg, Config)
            assert cfg.filename == cfg_file
            assert cfg.text == osp.abspath(osp.expanduser(cfg_file)) + '\n' + \
                open(cfg_file, 'r').read()

        tmp_cfg = tmp_path / 'tmp_cfg.xml'
        tmp_cfg.write_text('exist')
        with pytest.raises(IOError):
            Config.fromfile(tmp_cfg)

        with pytest.raises(FileNotFoundError):
            Config.fromfile('no_such_file.py')

        self._predefined_vars(tmp_path)
        self._base_variables()
        self._merge_from_base()
        self._code_in_config()
        self._merge_from_multiple_bases()
        self._custom_import()
        self._merge_from_dict()
        self._merge_delete()
        self._merge_intermediate_variable()
        self._merge_recursive_bases()
        self._syntax_error(tmp_path)
        self._reserved_key()

    def test_auto_argparser(self):
        tmp = sys.argv[1]
        sys.argv[1] = osp.join(self.data_path,
                               'config/py_config/merge_multi.py')
        parser, cfg = Config.auto_argparser()
        args = parser.parse_args()
        assert args.config == sys.argv[1]
        for key in cfg._cfg_dict.keys():
            if not isinstance(cfg[key], ConfigDict):
                assert getattr(args, key) is None
        # TODO currently do not support nested keys, bool args will be
        #  overwritten by int
        sys.argv[1] = tmp

    def test_fromstring(self):
        for filename in [
                'py_config/simple_config.py', 'py_config/simple.config.py',
                'json_config/simple_config.json',
                'yml_config/simple_config.yaml'
        ]:
            cfg_file = osp.join(self.data_path, 'config', filename)
            file_format = osp.splitext(filename)[-1]
            in_cfg = Config.fromfile(cfg_file)

            out_cfg = Config.fromstring(in_cfg.pretty_text, '.py')
            assert in_cfg._cfg_dict == out_cfg._cfg_dict

            cfg_str = open(cfg_file, 'r').read()
            out_cfg = Config.fromstring(cfg_str, file_format)
            assert in_cfg._cfg_dict == out_cfg._cfg_dict

        # test pretty_text only supports py file format
        cfg_file = osp.join(self.data_path, 'config',
                            'json_config/simple_config.json')
        in_cfg = Config.fromfile(cfg_file)
        with pytest.raises(Exception):
            Config.fromstring(in_cfg.pretty_text, '.json')

        # test file format error
        cfg_str = open(cfg_file, 'r').read()
        with pytest.raises(Exception):
            Config.fromstring(cfg_str, '.py')

        with pytest.raises(IOError):
            Config.fromstring(cfg_str, '.xml')

    def test_dump(self, tmp_path):
        cfg_file = osp.join(self.data_path,
                            'config/py_config/simple_config.py')
        cfg = Config.fromfile(cfg_file)
        dump_py = tmp_path / 'simple_config.py'
        dump_json = tmp_path / 'simple_config.json'
        dump_yaml = tmp_path / 'simple_config.yaml'

        cfg.dump(dump_py)
        cfg.dump(dump_json)
        cfg.dump(dump_yaml)
        assert open(dump_py, 'r').read() == cfg.pretty_text
        assert open(dump_json, 'r').read() == cfg.pretty_text
        assert open(dump_yaml, 'r').read() == cfg.pretty_text

        # test pickle
        cfg_file = osp.join(self.data_path,
                            'config/py_config/pickle_support.py')
        cfg = Config.fromfile(cfg_file)

        text_cfg_filename = tmp_path / '_text_config.py'
        cfg.dump(text_cfg_filename)
        text_cfg = Config.fromfile(text_cfg_filename)

        assert text_cfg._cfg_dict == cfg._cfg_dict

        cfg_file = osp.join(self.data_path,
                            'config/py_config/pickle_support.py')
        cfg = Config.fromfile(cfg_file)

        pkl_cfg_filename = tmp_path / '_pickle.pkl'
        dump(cfg, pkl_cfg_filename)
        pkl_cfg = load(pkl_cfg_filename)

        assert pkl_cfg._cfg_dict == cfg._cfg_dict

    def _predefined_vars(self, tmp_path):
        # test parse predefined_var in config
        cfg_file = osp.join(self.data_path,
                            'config/py_config/predefined_var.py')
        path = osp.join(osp.dirname(__file__), 'data', 'config')

        path = Path(path).as_posix()
        cfg_dict = dict(item1='predefined_var.py', item2=path, item3='abc_h')
        cfg = Config(cfg_dict, filename=cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        assert cfg.text == open(cfg_file, 'r').read()
        assert cfg.dump() == cfg.pretty_text

        dump_file = tmp_path / 'predefined_var.py'
        cfg.dump(dump_file)
        assert cfg.dump() == open(dump_file, 'r').read()
        assert Config.fromfile(dump_file)
        assert Config.fromfile(dump_file)['item1'] == cfg_dict['item1']
        assert Config.fromfile(dump_file)['item2'] == cfg_dict['item2']
        assert Config.fromfile(dump_file)['item3'] == cfg_dict['item3']

        # test no use_predefined_variable
        cfg_dict = dict(
            item1='{{fileBasename}}',
            item2='{{ fileDirname}}',
            item3='abc_{{ fileBasenameNoExtension }}')
        assert Config.fromfile(cfg_file, False)
        assert Config.fromfile(cfg_file, False)['item1'] == cfg_dict['item1']
        assert Config.fromfile(cfg_file, False)['item2'] == cfg_dict['item2']
        assert Config.fromfile(cfg_file, False)['item3'] == cfg_dict['item3']

        # test predefined_var.yaml
        cfg_file = osp.join(self.data_path,
                            'config/yml_config/predefined_var.yaml')
        cfg_dict = dict(
            item1=osp.join(osp.dirname(__file__), 'data', 'config'))
        cfg = Config(cfg_dict, filename=cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        assert cfg.text == open(cfg_file, 'r').read()
        assert cfg.dump() == yaml.dump(cfg_dict)

        dump_file = tmp_path / 'predefined_var.yaml'
        cfg.dump(dump_file)
        assert cfg.dump() == open(dump_file, 'r').read()
        assert Config.fromfile(dump_file)
        assert Config.fromfile(dump_file)['item1'] == cfg_dict['item1']

        # test no use_predefined_variable
        assert Config.fromfile(cfg_file, False)
        assert Config.fromfile(cfg_file, False)['item1'] == '{{ fileDirname }}'

        # test predefined_var.json
        cfg_file = osp.join(self.data_path,
                            'config/json_config/predefined_var.json')
        cfg_dict = dict(
            item1=osp.join(osp.dirname(__file__), 'data', 'config'))
        cfg = Config(cfg_dict, filename=cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        assert cfg.text == open(cfg_file, 'r').read()
        assert cfg.dump() == json.dumps(cfg_dict)

        dump_file = tmp_path / 'predefined_var.json'
        cfg.dump(dump_file)
        assert cfg.dump() == open(dump_file, 'r').read()
        assert Config.fromfile(dump_file)
        assert Config.fromfile(dump_file)['item1'] == cfg_dict['item1']

        # test no use_predefined_variable
        assert Config.fromfile(cfg_file, False)
        assert Config.fromfile(cfg_file, False)['item1'] == '{{ fileDirname }}'

        # test custom_imports for Config.fromfile
    def _custom_import(self):
        cfg_file = osp.join(self.data_path, 'config',
                            'py_config/custom_imports.py')
        sys.path.append(osp.join(self.data_path, 'config/py_config'))

        # Since the imported config will be regarded as a tmp file
        # it should be copied to the directory at the same level

        Config.fromfile(cfg_file, import_custom_modules=True)

        assert os.environ.pop('TEST_VALUE') == 'test'

    def _merge_from_base(self):
        cfg_file = osp.join(self.data_path,
                            'config/py_config/merge_single_base.py')
        cfg = Config.fromfile(cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        base_cfg_file = osp.join(self.data_path, 'config/py_config/base.py')
        merge_text = osp.abspath(osp.expanduser(base_cfg_file)) + '\n' + \
            open(base_cfg_file, 'r').read()
        merge_text += '\n' + osp.abspath(osp.expanduser(cfg_file)) + '\n' + \
                      open(cfg_file, 'r').read()
        assert cfg.text == merge_text
        assert cfg.item1 == [2, 3]
        assert cfg.item2.a == 1
        assert cfg.item3 is False
        assert cfg.item4 == 'test_base'

        with pytest.raises(TypeError):
            Config.fromfile(
                osp.join(self.data_path, 'config/py_config/merge_error.py'))

    def _merge_from_multiple_bases(self):
        cfg_file = osp.join(self.data_path, 'config/py_config/merge_multi.py')
        cfg = Config.fromfile(cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        # cfg.field
        assert cfg.item1 == [1, 2]
        assert cfg.item2.a == 0
        assert cfg.item3 is False
        assert cfg.item4 == 'test'
        assert cfg.item5 == dict(a=0, b=1)
        assert cfg.item6 == [dict(a=0), dict(b=1)]
        assert cfg.item7 == dict(a=[0, 1, 2], b=dict(c=[3.1, 4.2, 5.3]))

        with pytest.raises(KeyError):
            Config.fromfile(
                osp.join(self.data_path,
                         'config/py_config/merge_multi_error.py'))

    def _base_variables(self):
        for file in [
                'py_config/use_basevar.py', 'json_config/use_base_var.json',
                'yml_config/use_base_var.yaml'
        ]:
            cfg_file = osp.join(self.data_path, 'config', file)
            cfg = Config.fromfile(cfg_file)
            assert isinstance(cfg, Config)
            assert cfg.filename == cfg_file
            # cfg.field
            assert cfg.item1 == [1, 2]
            assert cfg.item2.a == 0
            assert cfg.item3 is False
            assert cfg.item4 == 'test'
            assert cfg.item5 == dict(a=0, b=1)
            assert cfg.item6 == [dict(a=0), dict(b=1)]
            assert cfg.item7 == dict(a=[0, 1, 2], b=dict(c=[3.1, 4.2, 5.3]))
            assert cfg.item8 == file.split('/')[-1]
            assert cfg.item9 == dict(a=0)
            assert cfg.item10 == [3.1, 4.2, 5.3]

        # test nested base
        for file in [
                'py_config/test_nested.py', 'json_config/test_nested.json',
                'yml_config/test_nested.yaml'
        ]:
            cfg_file = osp.join(self.data_path, 'config', file)
            cfg = Config.fromfile(cfg_file)
            assert isinstance(cfg, Config)
            assert cfg.filename == cfg_file
            # cfg.field
            assert cfg.base == '_base_.item8'
            assert cfg.item1 == [1, 2]
            assert cfg.item2.a == 0
            assert cfg.item3 is False
            assert cfg.item4 == 'test'
            assert cfg.item5 == dict(a=0, b=1)
            assert cfg.item6 == [dict(a=0), dict(b=1)]
            assert cfg.item7 == dict(a=[0, 1, 2], b=dict(c=[3.1, 4.2, 5.3]))
            assert cfg.item8 == 'use_basevar.py'
            assert cfg.item9 == dict(a=0)
            assert cfg.item10 == [3.1, 4.2, 5.3]
            assert cfg.item11 == 'use_basevar.py'
            assert cfg.item12 == dict(a=0)
            assert cfg.item13 == [3.1, 4.2, 5.3]
            assert cfg.item14 == [1, 2]
            assert cfg.item15 == dict(
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
        cfg_file = osp.join(self.data_path,
                            'config/py_config/basevar_assign.py')
        cfg = Config.fromfile(cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        assert cfg.item21 == 'use_basevar.py'
        assert cfg.item22 == 'use_basevar.py'
        assert cfg.item23 == [3.1, 4.2, 5.3]
        assert cfg.item24 == [3.1, 4.2, 5.3]
        assert cfg.item25 == dict(
            a=dict(b=[3.1, 4.2, 5.3]),
            b=[[3.1, 4.2, 5.3]],
            c=[[{
                'e': 'use_basevar.py'
            }], [{
                'a': 0
            }, {
                'b': 1
            }]],
            e='use_basevar.py')

    def _merge_recursive_bases(self):
        cfg_file = osp.join(self.data_path,
                            'config/py_config/recursive_bases.py')
        cfg = Config.fromfile(cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        # cfg.field
        assert cfg.item1 == [2, 3]
        assert cfg.item2.a == 1
        assert cfg.item3 is False
        assert cfg.item4 == 'test_recursive_bases'

    def _merge_from_dict(self):
        cfg_file = osp.join(self.data_path,
                            'config/py_config/simple_config.py')
        cfg = Config.fromfile(cfg_file)
        input_options = {'item2.a': 1, 'item2.b': 0.1, 'item3': False}
        cfg.merge_from_dict(input_options)
        assert cfg.item2 == dict(a=1, b=0.1)
        assert cfg.item3 is False

        cfg_file = osp.join(self.data_path, 'config/py_config/merge_dict.py')
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

    def _merge_delete(self):
        cfg_file = osp.join(self.data_path, 'config/py_config/delete.py')
        cfg = Config.fromfile(cfg_file)
        # cfg.field
        assert cfg.item1 == dict(a=0)
        assert cfg.item2 == dict(a=0, b=0)
        assert cfg.item3 is True
        assert cfg.item4 == 'test'
        assert '_delete_' not in cfg.item2

        # related issue: https://github.com/open-mmlab/mmengine/issues/1570
        assert type(cfg.item1) == ConfigDict
        assert type(cfg.item2) == ConfigDict

    def _merge_intermediate_variable(self):

        cfg_file = osp.join(self.data_path, 'config/py_config/i_child.py')
        cfg = Config.fromfile(cfg_file)
        # cfg.field
        assert cfg.item1 == [1, 2]
        assert cfg.item2 == dict(a=0)
        assert cfg.item3 is True
        assert cfg.item4 == 'test'
        assert cfg.item_cfg == dict(b=2)
        assert cfg.item5 == dict(cfg=dict(b=1))
        assert cfg.item6 == dict(cfg=dict(b=2))

    def _code_in_config(self):
        cfg_file = osp.join(self.data_path, 'config/py_config/code.py')
        cfg = Config.fromfile(cfg_file)
        # cfg.field
        assert cfg.cfg.item1 == [1, 2]
        assert cfg.cfg.item2 == dict(a=0)
        assert cfg.cfg.item3 is True
        assert cfg.cfg.item4 == 'test'
        assert cfg.item5 == 1

    def test_dict(self):
        cfg_dict = dict(
            item1=[1, 2], item2=dict(a=0), item3=True, item4='test')

        for filename in [
                'py_config/simple_config.py', 'json_config/simple_config.json',
                'yml_config/simple_config.yaml'
        ]:
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
            with pytest.raises(AttributeError):
                cfg.not_exist
            # field in cfg, cfg[field], cfg.get()
            for name in ['item1', 'item2', 'item3', 'item4']:
                assert name in cfg
                assert cfg[name] == cfg_dict[name]
                assert cfg.get(name) == cfg_dict[name]
                assert cfg.get('not_exist') is None
                assert cfg.get('not_exist', 0) == 0
                with pytest.raises(KeyError):
                    cfg['not_exist']
            assert 'item1' in cfg
            assert 'not_exist' not in cfg
            # cfg.update()
            cfg.update(dict(item1=0))
            assert cfg.item1 == 0
            cfg.update(dict(item2=dict(a=1)))
            assert cfg.item2.a == 1

    def test_setattr(self):
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

        with pytest.raises(TypeError):
            cfg[[1, 2]] = 0

    def test_pretty_text(self, tmp_path):
        cfg_file = osp.join(self.data_path, 'config/py_config/merge_multi.py')
        cfg = Config.fromfile(cfg_file)
        text_cfg_filename = tmp_path / '_text_config.py'
        with open(text_cfg_filename, 'w') as f:
            f.write(cfg.pretty_text)
        text_cfg = Config.fromfile(text_cfg_filename)
        assert text_cfg._cfg_dict == cfg._cfg_dict

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
        # Imbalance bracket
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

    def _reserved_key(self):
        cfg_file = osp.join(self.data_path, 'config/py_config/reserved_key.py')
        with pytest.raises(KeyError):
            Config.fromfile(cfg_file)

    def _syntax_error(self, tmp_path):
        temp_cfg_path = tmp_path / 'tmp_file.py'
        # write a file with syntax error
        with open(temp_cfg_path, 'w') as f:
            f.write('a=0b=dict(c=1)')
        with pytest.raises(
                SyntaxError, match='There are syntax errors in config file'):
            Config.fromfile(temp_cfg_path)

    def test_repr(self):
        cfg_file = osp.join(self.data_path,
                            'config/py_config/simple_config.py')
        cfg = Config.fromfile(cfg_file)
        print(cfg)
