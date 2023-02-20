# Copyright (c) OpenMMLab. All rights reserved.
import ast
import copy
import importlib
import os
import os.path as osp
import re
from abc import ABCMeta
from collections import OrderedDict
from functools import partial
from itertools import accumulate
from typing import Any, Mapping, Optional, Union

from mmengine.utils import get_installed_path

# EAGER_IMPORTS = ['mmengine.config']


class LazyCall:

    def __init__(self, type, instance_id=None, *args, **kwargs) -> None:
        super().__setattr__('type', type)
        super().__setattr__('kwargs', kwargs)
        super().__setattr__('args', args)
        super().__setattr__('missed_args', {})
        for key, value in kwargs.items():
            if isinstance(value, LazyCall):
                if value.inner_built and ('inner_built' not in self.__dict__):
                    super().__setattr__('inner_built', True)
                if value == '???':
                    self.missed_args[key] = value
                    super().__setattr__('inner_built', True)
        super().__setattr__('inner_built', False)
        instance_id = id(self) if instance_id is None else instance_id
        super().__setattr__('instance_id', instance_id)
        # self.args = args

    def build(self, memo=None):
        if memo is None:
            memo = dict()
        # built is used for duplicated built.
        def _build_lazy_call(kwargs, global_built):
            if isinstance(kwargs, Mapping):
                return type(kwargs)({
                    key: _build_lazy_call(value, global_built)
                    for key, value in kwargs.items()
                })
                # for key, value in kwargs.items():
                #     kwargs[key] = _build_lazy_call(value, global_built)
                # return kwargs
            elif isinstance(kwargs, (list, tuple)):
                return type(kwargs)([
                    _build_lazy_call(value, global_built) for value in kwargs
                ])
                # for i, value in enumerate(kwargs):
                #     kwargs[i] = _build_lazy_call(kwargs[i], global_built)
                # # return kwargs
            elif isinstance(kwargs, LazyCall):
                if kwargs.instance_id not in global_built:
                    ret = kwargs.build(memo=global_built)
                    global_built[kwargs.instance_id] = ret
                    return ret
                else:
                    return global_built[kwargs.instance_id]
            else:
                return kwargs

        kwargs = _build_lazy_call(copy.deepcopy(self.kwargs), memo)
        args = _build_lazy_call(copy.deepcopy(self.args), memo)

        # Built by custom function: def xxx
        if self.type in self.global_dict:
            return self.global_dict[self.type](*args, **kwargs)

        if self.type in self.module_dict:
            module = self.module_dict[self.type]
            module = importlib.import_module(module)
            func = getattr(module, self.type)
            return func(*args, **kwargs)

        func_name_list = self.type.split('.')
        for func_name in accumulate(func_name_list):
            if func_name in self.module_dict:
                try:
                    # Absolute import.
                    # import mmdet.models
                    # mmdet.models.xxx.xxx()
                    module = self.module_dict[func_name]
                    attrs = self.type.rstrip(func_name).split('.')
                    func = importlib.import_module(module)
                    for attr in attrs:
                        func = getattr(func, attr)
                except:
                    # Relative import.
                    module = self.module_dict[func_name]
                    func = importlib.import_module(module)
                    for attr in func_name_list:
                        func = getattr(func, attr)

                return func(*args, **kwargs)
        raise Exception()

    def __setattr__(self, name: str, value: Any) -> None:
        assert not self.args, (
            f'If you want to set attribute, please build {self.type} with '
            'keyword args, but not positional args.')
        if name == 'kwargs' or name == 'type' or name == 'instance_id':
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
        return LazyAttr(name, self)
        # try:
        #     return super().__getattr__(name)
        # except:
        #     builder = partial(LazyCall, name)
        #     return builder

    def __contains__(self, key):
        return key in self.kwargs

    def __deepcopy__(self, memo):
        cls = self.__class__
        ret = cls(
            copy.deepcopy(self.type),
            instance_id=self.instance_id,
            **copy.deepcopy(self.kwargs))
        super(LazyCall, ret).__setattr__('global_dict', self.global_dict)
        super(LazyCall, ret).__setattr__('module_dict', self.module_dict)

        return ret

    def set_build_variable(self, global_dict, module_dict):
        super().__setattr__('global_dict', global_dict)
        super().__setattr__('module_dict', module_dict)


class LazyAttrCall:

    def __init__(self, attr, **kwargs) -> None:
        self.kwargs = kwargs
        self.attr = attr

    def build(self):
        return self.attr.build()(**self.kwargs)


class LazyAttr:

    def __init__(self, name, source) -> None:
        self.name = name
        self.source = source

    def __call__(self, **kwargs: Any) -> Any:
        return LazyAttrCall(self, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return LazyAttr(name, self)

    def build(self):
        return getattr(self.source.build(), self.name)

    # def build(self):
    #     if isinstance(self.lazy, LazyCall):
    #         return self.lazy_call.build(memo=None)
    #     else:
    #         return self.lazy.build()


class LazyNameCall(LazyCall):

    def build(self, memo=None):
        module = self.module_dict[self.type]
        module = importlib.import_module(module)
        func = getattr(module, self.type)
        return func

    def __getattr__(self, name: str) -> Any:
        return LazyAttr(name, self)

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
        # a(): LazyCall a
        # a.b(): LazyCall a.b, a is a module variable or previous built variable
        # not lazycall

        if not isinstance(node.func, ast.Name):
            return node

        def _visit_call(node):
            if hasattr(node.func, 'id'):
                func_name = node.func.id
            else:
                func_name = ''
                func = node.func
                while hasattr(func, 'attr'):
                    try:
                        func_name += func.attr
                        func = func.value
                    except:
                        break

            if func_name in __builtins__ or func_name in globals():
                return super().generic_visit(node)
            # node.func.id = 'ConfigNode'
            node.args.insert(0, ast.Constant(value=func_name, kind=None))
            # node.ctx = ast.Load()
            new_node = ast.Call(
                ast.Name(id='LazyCall', ctx=ast.Load()),
                args=node.args,
                keywords=node.keywords)
            return super().generic_visit(new_node)

        return _visit_call(node)

    def visit_ImportFrom(self, node):
        # Relative improt
        # for eager_import in EAGER_IMPORTS:
        #     if node.module.startswith(eager_import):
        #         return super().generic_visit(node)
        module = f'{node.level*"."}{node.module}'
        if module in self.base_dict:
            for name in node.names:
                if name.name == '*':
                    self.global_dict.update(self.base_dict[module])
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
                keywords=[])
            return new_node
        else:
            return node

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if hasattr(alias, 'asname'):
                self.module_dict[alias.asname] = alias.name
            else:
                self.module_dict[alias.name] = alias.name
        return None

    # def visit_Assign(self, node):
    #     if (isinstance(node.targets[0], ast.Name)
    #             and node.targets[0].id == self.key):
    #         return None
    #     else:
    #         return node


class Config(OrderedDict):
    base_key = '_base_'

    def setdefault(self, key, default):
        if key not in self.keys():
            setattr(self, key, default)

    @classmethod
    def _parse(cls, filepath, module_dict):
        with open(filepath) as f:
            global_dict = OrderedDict({
                'LazyCall': LazyCall,
                'LazyNameCall': LazyNameCall
            })
            base_dict = {}
            module_dict = {}

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
                base_module_dict, base_cfg = cls._parse(
                    module_path, module_dict)
                module_dict.update(base_module_dict)
                base_dict[base_module] = base_cfg
            # TODO only support relative import now.
            transform = Transform(
                base_dict=base_dict,
                module_dict=module_dict,
                global_dict=global_dict)
            modified_code = transform.visit(code)
            modified_code = ast.fix_missing_locations(modified_code)
            exec(
                compile(modified_code, filepath, mode='exec'), global_dict,
                global_dict)

            ret = OrderedDict()
            for key, value in global_dict.items():
                if key.startswith('__') or key in ['LazyCall', 'LazyNameCall']:
                    continue
                ret[key] = value

            cfg_dict = Config._to_config_dict(ret, global_dict, module_dict)
            return module_dict, cls(
                cfg_dict, module_dict=module_dict, global_dict=global_dict)

    @classmethod
    def fromfile(cls, filepath: str):
        module_dict = OrderedDict()
        module_dict, cfg_dict = cls._parse(
            filepath=filepath, module_dict=module_dict)
        return cfg_dict

    @classmethod
    def _to_config_dict(cls, cfg_dict, global_dict, module_dict):
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
                cfg_dict.set_build_variable(global_dict, module_dict)
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


def convert_to(ori: LazyCall, target: LazyNameCall):
    ori.type = target.type
    return ori
