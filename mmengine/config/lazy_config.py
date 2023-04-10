# Copyright (c) OpenMMLab. All rights reserved.
import ast
import copy
import os.path as osp
import re
from collections import OrderedDict

from mmengine.utils import get_installed_path
from .lazy import LazyCall, LazyModule
from .lazy_ast import Transform, _gather_abs_import_lazymodule


class Config(OrderedDict):
    base_key = '_base_'

    def setdefault(self, key, default):
        if key not in self.keys():
            setattr(self, key, default)

    @classmethod
    def _parse(cls, filepath):
        with open(filepath) as f:
            global_dict = OrderedDict({
                'LazyModule': LazyModule,
            })
            base_dict = {}

            code = ast.parse(f.read())
            base_code = []
            for inst in code.body:
                if (isinstance(inst, ast.Assign)
                        and isinstance(inst.targets[0], ast.Name)
                        and inst.targets[0].id == cls.base_key):
                    base_code.append(inst)
            variable_dict: dict = {}
            base_code = ast.Module(body=base_code, type_ignores=[])
            exec(
                compile(base_code, '', mode='exec'), variable_dict,
                variable_dict)
            base_modules = variable_dict.get('_base_', [])
            if not isinstance(base_modules, list):
                base_modules = [base_modules]
            for base_module in base_modules:
                # from .xxx import xxx
                # fron mmdet.config.xxx import xxx
                level = len(re.match(r'\.*', base_module).group())
                if level > 0:
                    # Relative import
                    base_dir = osp.dirname(filepath)
                    module_path = osp.join(
                        base_dir, *(['..'] * (level - 1)),
                        f'{base_module[level:].replace(".", "/")}.py')
                else:
                    # Absolute import
                    module_list = base_module.split('.')
                    if len(module_list) == 1:
                        # TODO
                        ...
                        # module_path = osp.abspath(f'{module_list[0]}.py')
                    else:
                        package = module_list[0]
                        root_path = get_installed_path(package)
                        module_path = f'{osp.join(root_path, *module_list[1:])}.py'
                base_cfg = cls._parse(module_path)
                base_dict[base_module] = base_cfg
            # TODO only support relative import now.
            transform = Transform(global_dict=global_dict, base_dict=base_dict)
            modified_code = transform.visit(code)
            modified_code = _gather_abs_import_lazymodule(modified_code)
            modified_code = ast.fix_missing_locations(modified_code)
            exec(
                compile(modified_code, filepath, mode='exec'), global_dict,
                global_dict)

            ret = OrderedDict()
            for key, value in global_dict.items():
                if key.startswith('__') or key in ['LazyModule']:
                    continue
                ret[key] = value

            cfg_dict = Config._to_config_dict(ret)
            return cls(cfg_dict, global_dict=global_dict)

    @classmethod
    def fromfile(cls, filepath: str):
        cfg_dict = cls._parse(filepath=filepath)
        return cfg_dict

    @classmethod
    def _to_config_dict(cls, cfg_dict):
        ordered_keys = cfg_dict.keys()
        result = OrderedDict()

        # Do not used generator-expression here for the building sequence.
        def _convert(cfg_dict):
            if isinstance(cfg_dict, dict):
                for key, value in cfg_dict.items():
                    cfg_dict[key] = _convert(value)
                return cls(cfg_dict)
            if isinstance(cfg_dict, (list, tuple)):
                return type(cfg_dict)(_convert(item) for item in cfg_dict)
            if isinstance(cfg_dict, LazyCall):
                cfg_dict.kwargs = _convert(cfg_dict.kwargs)
                cfg_dict.args = _convert(cfg_dict.args)
                return cfg_dict
            return cfg_dict

        cfg_dict = _convert(cfg_dict)
        # Keep the order of cfg_dict
        for key in ordered_keys:
            result[key] = cfg_dict[key]
        return result

    def build(self):
        built = dict()

        def _build(cfg_dict):
            if isinstance(cfg_dict, dict):
                for key, value in cfg_dict.items():
                    cfg_dict[key] = _build(value)
            if isinstance(cfg_dict, (list, tuple)):
                return type(cfg_dict)(_build(item) for item in cfg_dict)
            if isinstance(cfg_dict, LazyCall):
                if id(cfg_dict) in built:
                    return built[id(cfg_dict)]
                cfg_dict.kwargs = _build(cfg_dict.kwargs)
                cfg_dict.args = _build(cfg_dict.args)
                ret = cfg_dict.build()
                built[id(cfg_dict)] = ret
                return ret
            return cfg_dict

        ret = copy.deepcopy(self)
        for key in ret.keys():
            ret[key] = _build(self[key])
        return ret

    def pop(self, name, default=None):
        ret = getattr(self, name, default)
        try:
            self.__delitem__(name)
        except:
            pass
        return ret

    def copy(self, deep=False):
        if deep:
            return copy.deepcopy(self)
        return copy.copy(self)

    def __getattr__(self, key):
        if key in self.keys():
            return self[key]
        raise AttributeError(f'No attribute named {key}')

    def __setattr__(self, key, value):
        self[key] = value
