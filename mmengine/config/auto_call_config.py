import ast
import copy
import importlib
import os
import os.path as osp
import re
from abc import ABCMeta
from collections import OrderedDict
from typing import Any, Mapping, Optional, Union

from ml_collections import ConfigDict as _ConfigDict
from ml_collections.config_dict.config_dict import (FrozenConfigDict,
                                                    _configdict_fill_seed)

from mmengine.utils import get_installed_path


class LazyCall:
    def __init__(
        self,
        name_id,
        **kwargs
    ) -> None:
        super().__setattr__('name_id', name_id)
        super().__setattr__('kwargs', kwargs)
        # self.args = args

    def build(self):
        if self.name_id in self.module_dict:
            module = self.module_dict[self.name_id]
            module = importlib.import_module(module)
            func = getattr(module, self.name_id)
            return func(**self.kwargs)
        func_name_list = self.name_id.split('.')

        # Built by custom function: def xxx
        if self.name_id in self.global_dict:
            return self.global_dict[self.name_id](*self.args, **self.kwargs)
        for i in range(len(func_name_list)):
            func_name = '.'.join(func_name_list[:i + 1])
            if func_name in self.module_dict:
                try:
                    # import mmdet.models
                    # mmdet.models.xxx.xxx()
                    module = self.module_dict[func_name]
                    attrs = func_name_list[i + 1:]
                    func = importlib.import_module(module)
                    for attr in attrs:
                        func = getattr(func, attr)
                except:
                    # Relative import.
                    module = self.module_dict[func_name]
                    func = importlib.import_module(module)
                    for attr in func_name_list:
                        func = getattr(func, attr)

                return func(**self.kwargs)
        raise Exception()

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'kwargs':
            super().__setattr__(name, value)
            return
        self.kwargs[name] = value
        super().__setattr__(name, value)
    
    def setdefault(self, name, value):
        if name not in self.kwargs:
            self.kwargs[name] = value
    
    def __getattr__(self, name: str) -> Any:
        if name in self.kwargs:
            return self.kwargs[name]

    def __deepcopy__(self, memo):
        cls = self.__class__
        return cls(copy.deepcopy(self.name_id), **copy.deepcopy(self.kwargs))

    def set_build_variable(self, global_dict, module_dict):
        super().__setattr__('global_dict', global_dict)
        super().__setattr__('module_dict', module_dict)




class LazyNameCall(LazyCall):
    def build(self):
        module = self.module_dict[self.name_id]
        module = importlib.import_module(module)
        func = getattr(module, self.name_id)
        return func
    

    # def _addindent(self, s_, numSpaces):
    #     s = s_.split('\n')
    #     # don't do anything for single-line stuff
    #     if len(s) == 1:
    #         return s_
    #     first = s.pop(0)
    #     s = [(numSpaces * ' ') + line for line in s]
    #     s = '\n'.join(s)
    #     s = first + '\n' + s
    #     return s

    # def __repr__(self) -> str:
    #     ret = f'{self.name_id}:\n'
    #     for key, value in self.kwargs.items():
    #         if not isinstance(value, LazyCall):
    #             ret += self._addindent(f'{key}: {value}\n', 2)
    #         else:
    #             ret += self._addindent(f'{key}:\n', 2)
    #             ret += self._addindent(value.__repr__(), 4)
    #     return ret
    # def __call__(self, *args, **kwds):
    #     return self.returned(*args, **kwds)
        

# from mmengine.config import ConfigDict


class Transform(ast.NodeTransformer):
    def __init__(self, base_dict, module_dict, global_dict) -> None:
        self.base_dict = base_dict
        self.module_dict = module_dict
        self.global_dict = global_dict
        super().__init__()

    def visit_Call(self, node):
        if hasattr(node.func, 'id'):
            func_name = node.func.id
        else:
            attr = []
            func = node.func
            while True:
                attr.insert(0, func.attr)
                func = func.value
                if isinstance(func, ast.Name):
                    attr.insert(0, func.id)
                    break

            func_name = '.'.join(attr)

        if func_name in __builtins__ or func_name in globals():
            return super().generic_visit(node)
        # node.func.id = 'ConfigNode'
        node.args.insert(0, ast.Constant(value=func_name, kind=None))
        # node.ctx = ast.Load()
        new_node = ast.Call(
            ast.Name(id='LazyCall', ctx=ast.Load()),
            args=node.args,
            keywords=node.keywords
        )
        return super().generic_visit(new_node)

    def visit_ImportFrom(self, node):
        # Relative improt
        module = f'{node.level*"."}{node.module}'
        if module in self.base_dict:
            for name in node.names:
                if name.name == '*':
                    self.global_dict.update(self.base_dict[module]._fields)
                    return None
                self.global_dict[name.name] = self.base_dict[module][name.name]

        for name in node.names:
            self.module_dict[name.name] = module
        return None
    
    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.module_dict:
            new_node = ast.Call(
                func=ast.Name(id='LazyNameCall', ctx=ast.Load()),
                args=[ast.Constant(value=node.id, kind=None)],
                keywords=[]
            )
            return new_node
        else:
            return node

    # def visit_Import(self, node: ast.Import):
    #     for name in node.names:
    #         module_dict[name.name] = module_dict
    #     return node
    
    # def visit_Assign(self, node):
    #     if (isinstance(node.targets[0], ast.Name)
    #             and node.targets[0].id == self.key):
    #         return None
    #     else:
    #         return node




class Config(_ConfigDict, dict):
    base_key = '_base_'

    def __init__(self, *args, module_dict=None, global_dict=None, type_safe=False, **kwargs):
        super().__init__(*args, type_safe=type_safe, **kwargs)
        object.__setattr__(self, 'module_dict', module_dict)
        object.__setattr__(self, 'global_dict', global_dict)
    
    def setdefault(self, key, default):
        if key not in self.keys():
            setattr(self, key, default)

    @classmethod
    def _parse(cls, filepath, module_dict):
        with open(filepath, 'r') as f:
            global_dict = OrderedDict({'LazyCall': LazyCall,
                                       'LazyNameCall': LazyNameCall})
            base_dict = {}
            module_dict = {}

            code = ast.parse(f.read())
            base_code = []
            for inst in code.body:
                if (isinstance(inst, ast.Assign) and
                    isinstance(inst.targets[0], ast.Name) and
                    inst.targets[0].id == cls.base_key
                ):
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
                    module_path = base_module[level:].replace('.', '/')
                    base_dir = osp.dirname(filepath)
                    module_path = osp.join(base_dir, '.' * level, f'{module_path}.py')
                else:
                    # Absolute import
                    module_list = base_module.split()
                    if len(module_list) == 1:
                        module_path = osp.abspath(f'{module_list[0]}.py')
                    else:
                        package = module_list[0]
                        root_path = get_installed_path(package)
                        module_path = osp.join(root_path, *module_list[1:])
                base_module_dict, base_cfg = cls._parse(module_path, module_dict)
                module_dict.update(base_module_dict)
                base_dict[base_module] = base_cfg
            # TODO only support relative import now.
            transform = Transform(
                base_dict=base_dict,
                module_dict=module_dict,
                global_dict=global_dict
            )
            modified_code = transform.visit(code)
            modified_code = ast.fix_missing_locations(modified_code)
            exec(
                compile(modified_code, '', mode='exec'),
                global_dict,
                global_dict
            )

            ret = OrderedDict()
            for key, value in global_dict.items():
                if key.startswith('__') or key in ['LazyCall', 'LazyNameCall']:
                    continue
                ret[key] = value

            cfg_dict = Config._to_config_dict(ret, global_dict, module_dict)
            return module_dict, cls(cfg_dict,
                                    module_dict=module_dict,
                                    global_dict=global_dict)

    @classmethod
    def fromfile(cls, filepath: str):
        module_dict = OrderedDict()
        module_dict, cfg_dict = cls._parse(filepath=filepath, module_dict=module_dict)
        return cfg_dict

    @classmethod
    def _to_config_dict(cls, cfg_dict, global_dict, module_dict):
        ordered_keys = cfg_dict.keys()
        result = OrderedDict()

        def _convert(cfg_dict):
            if isinstance(cfg_dict, (dict, _ConfigDict)):
                for key, value in cfg_dict.items():
                    cfg_dict[key] = _convert(value)
                return cls(cfg_dict)
            if isinstance(cfg_dict, (list, tuple)):
                return type(cfg_dict)(_convert(item) for item in cfg_dict)
            if isinstance(cfg_dict, LazyCall):
                cfg_dict.kwargs = _convert(cfg_dict.kwargs)
                cfg_dict.set_build_variable(global_dict, module_dict)
                return cfg_dict
            return cfg_dict

        cfg_dict = _convert(cfg_dict)
        # Keep the order of cfg_dict
        for key in ordered_keys:
            result[key] = cfg_dict[key]
        return result

    def build(self):
        def _build(cfg_dict):
            if isinstance(cfg_dict, (dict, _ConfigDict)):
                for key, value in cfg_dict.items():
                    cfg_dict[key] = _build(value)
            if isinstance(cfg_dict, (list, tuple)):
                return type(cfg_dict)(_build(item) for item in cfg_dict)
            if isinstance(cfg_dict, LazyCall):
                cfg_dict.kwargs = _build(cfg_dict.kwargs)
                return cfg_dict.build()
            return cfg_dict
        ret = copy.deepcopy(self)
        for key in ret._fields.keys():
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

