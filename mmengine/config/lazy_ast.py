# Copyright (c) OpenMMLab. All rights reserved.
import ast
import os.path as osp
import sys
from collections import defaultdict
from importlib.util import find_spec
from typing import List

PYTHON_ROOT_DIR = osp.dirname(osp.dirname(sys.executable))


def _is_builtin_module(module_name: str) -> bool:
    """Check if a module is a built-in module."""
    if module_name.startswith('.'):
        return False
    spec = find_spec(module_name)
    if spec is None:
        raise ImportError(f'Cannot find module {module_name}')
    origin_path = osp.abspath(getattr(spec, 'origin', None))
    if origin_path is None:
        return True
    elif ('site-package' in origin_path
          or not origin_path.startswith(PYTHON_ROOT_DIR)):
        return False
    else:
        return True


class Transform(ast.NodeTransformer):

    def __init__(self, global_dict, base_dict=None) -> None:
        self.base_dict = base_dict if base_dict is not None else {}
        self.global_dict = global_dict
        super().__init__()

    def visit_ImportFrom(self, node):
        # Built-in modules will not be parsed as LazyModule
        module = f'{node.level*"."}{node.module}'
        if _is_builtin_module(module):
            return node

        if module in self.base_dict:
            for name in node.names:
                if name.name == '*':
                    self.global_dict.update(self.base_dict[module])
                    return None
                self.global_dict[name.name] = self.base_dict[module][name.name]
            return None

        # TODO: Support lazyimport module from relative path
        nodes = []
        for name in node.names:
            if name == '*':
                # TODO If user import * from a non-config module, it should
                # fallback to import the real module and raise a warning to
                # remind user the real module will be imported which will slows
                # donwn the parsing speed.
                raise RuntimeError(
                    'You cannot import * from a non-config module, please use')
            elif name.asname is not None:
                # case1:
                # from mmengine.dataset import BaseDataset as Dataset ->
                # Dataset = LazyModule('mmengine.dataset', 'BaseDataset')
                code = f'{name.asname} = LazyModule("{module}", "{name.name}")'
            else:
                # case2:
                # from mmengine.model import BaseModel
                # BaseModel = LazyModule('mmengine.model', 'BaseModel')
                code = f'{name.name} = LazyModule("{module}", "{name.name}")'
            try:
                nodes.append(ast.parse(code).body[0])
            except Exception as e:
                raise ImportError(
                    f'Cannot import {name} from {module}',
                    '1. Cannot import * from 3rd party lib in the config file',
                    '2. Please check if the module is a base config which '
                    'should be added to `_base_`',
                ) from e
        return nodes

    def visit_Import(self, node):
        # For absolute import like: `import mmdet.configs as configs`.
        # It will be parsed as:
        # configs = LazyModule('mmdet.configs')
        # For absolute import like:
        # `import mmdet.configs`
        # `import mmdet.configs.default_runtime`
        # This will be parsed as
        # mmdet = LazyModule(['mmdet.configs.default_runtime', 'mmdet.configs])
        # However, visit_Import cannot gather other import information, so
        # `_gather_abs_import_lazymodule` will gather all import information
        # from the same module and construct the LazyModule.
        alias_list = node.names
        assert len(alias_list) == 1, (
            'Does not support import multiple modules in one line')
        # TODO Support multiline import
        alias = alias_list[0]
        if alias.asname is not None:
            return ast.parse(
                f'{alias.asname} = LazyModule("{alias.name}")').body[0]
        return node


def _gather_abs_import_lazymodule(tree: ast.Module):
    imported = defaultdict(list)
    new_body: List[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Skip converting built-in module to LazyModule
                if _is_builtin_module(alias.name):
                    new_body.append(node)
                    continue
                module = alias.name.split('.')[0]
                imported[module].append(alias.name)
            continue
        new_body.append(node)

    for key, value in imported.items():
        lazy_module_assign = ast.parse(f'{key} = LazyModule({value})')
        new_body.insert(0, lazy_module_assign.body[0])
    tree.body = new_body
    return tree
