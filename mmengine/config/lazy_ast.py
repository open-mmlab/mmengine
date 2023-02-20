import ast
from collections import defaultdict


class Transform(ast.NodeTransformer):

    def __init__(self, global_dict, base_dict=None) -> None:
        self.base_dict = base_dict if base_dict is not None else {}
        self.global_dict = global_dict
        super().__init__()

    def visit_ImportFrom(self, node):
        module = f'{node.level*"."}{node.module}'
        if module in self.base_dict:
            for name in node.names:
                if name.name == '*':
                    self.global_dict.update(self.base_dict[module])
                    return None
                self.global_dict[name.name] = self.base_dict[module][name.name]

        # TODO: Support lazyimport module from relative path
        nodes = []
        for name in node.names:
            if name.asname is not None:
                name = name.asname
            try:
                nodes.append(
                    ast.parse(
                        f'{name.name} = LazyModule("{module}", "{name.name}")'
                    ).body[0])
            except Exception as e:
                raise ImportError(
                    f'Cannot import {name.name} from {module}\n',
                    '1. Cannot import * from 3rd party lib in the config '
                    'file\n',
                    '2. Please check if the module is a base config which '
                    'should be added to `_base_`',
                ) from e
        return nodes

    def visit_Import(self, node):
        alias_list = node.names
        assert len(alias_list) == 1, (
            'Does not support import muiltiple modules in one line')
        alias = alias_list[0]
        if alias.asname is not None:
            return ast.parse(
                f'{alias.asname} = LazyModule("{alias.name}")').body[0]
        return node


def import_to_lazymodule(tree):
    imported = defaultdict(list)
    new_body = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(name.asname is not None for name in node.names):
                new_body.append(node)
                # TODO: Support multiple import xxx as xx in the same line
                continue
            for alias in node.names:
                module = alias.name.split('.')[0]
                imported[module].append(alias.name)
            continue
        new_body.append(node)

    for key, value in imported.items():
        lazy_module_assign = ast.parse(f'{key} = LazyModule({value})')
        new_body.insert(0, lazy_module_assign.body[0])
    tree.body = new_body
    return tree
