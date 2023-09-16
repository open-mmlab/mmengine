# Copyright (c) OpenMMLab. All rights reserved.
import ast
import inspect
import logging
import os.path as osp
import textwrap
import types
from abc import ABCMeta, abstractmethod
from operator import attrgetter
from typing import Any, Dict, List, Optional

import torch

from mmengine.logging import MessageHub, print_log
from mmengine.registry import HOOKS, RECORDERS
from . import Hook


def function_with_index(function, index):
    return function + '@' + str(index)


class FunctionRecorderTransformer(ast.NodeTransformer):

    def __init__(self, target, target_index):
        super().__init__()
        self._target = target
        if isinstance(target_index, list):
            self._target_index = set(target_index)
        else:
            self._target_index = {target_index}
        self.count = -1

    def visit_Assign(self, node):
        if node.targets[0].id != self._target:
            return node
        self.count += 1
        if self.count not in self._target_index:
            return node
        update_messagehub_node = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='message_hub', ctx=ast.Load()),
                    attr='update_info',
                    ctx=ast.Load()),
                args=[
                    ast.Constant(
                        value=function_with_index(node.targets[0].id,
                                                  self.count)),
                    ast.Name(id=node.targets[0].id, ctx=ast.Load())
                ],
                keywords=[]))

        return [node, update_messagehub_node]


# Take "x = self.conv1(x)" as an example
# genertate "self_conv1 = self.conv1(x)"
# and "x = self_conv1"
# and "message_hub.update_info('self_conv1', self_conv1)"
def _get_tensor_key(target):
    target = target.replace('.', '_')
    return target


class AttributeRecorderTransformer(ast.NodeTransformer):

    def __init__(self, target):
        super().__init__()
        self._target = target
        self._visited = False

    def _get_target_attribute(self):
        func_chain = self._target.split('.')
        assert len(func_chain) >= 2
        attr = ast.Attribute(
            value=ast.Name(id=func_chain[0], ctx=ast.Load()),
            attr=func_chain[1],
            ctx=ast.Load())
        for ele in func_chain[2:]:
            attr = ast.Attribute(value=attr, attr=ele, ctx=ast.Load())
        return attr

    def _deepcopy_varname(self):
        return f'_deep_copy_{self._target.replace(".", "_")}'

    def _get_deep_copy_node(self, var_node):
        if_node = ast.If(
            test=ast.Call(
                func=ast.Name(id='isinstance', ctx=ast.Load()),
                args=[
                    var_node,
                    ast.Attribute(
                        value=ast.Name(id='torch', ctx=ast.Load()),
                        attr='Tensor',
                        ctx=ast.Load())
                ],
                keywords=[]),
            body=[
                ast.Assign(
                    targets=[
                        ast.Name(id=self._deepcopy_varname(), ctx=ast.Store())
                    ],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Call(
                                func=ast.Attribute(
                                    var_node, attr='detach', ctx=ast.Load()),
                                args=[],
                                keywords=[]),
                            attr='clone',
                            ctx=ast.Load()),
                        args=[],
                        keywords=[]))
            ],
            orelse=[
                ast.Assign(
                    targets=[
                        ast.Name(id=self._deepcopy_varname(), ctx=ast.Store())
                    ],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='copy', ctx=ast.Load()),
                            attr='deepcopy',
                            ctx=ast.Load()),
                        args=[var_node],
                        keywords=[]))
            ])
        return if_node

    def visit_Assign(self, node):
        if self._visited:
            return node
        # insert update attribute node after message_hub assign node
        if node.targets[0].id == 'message_hub':
            self._visited = True

        attribute_node = self._get_target_attribute()
        if_node = self._get_deep_copy_node(attribute_node)
        deep_copy_attribute_node = ast.Name(
            id=self._deepcopy_varname(), ctx=ast.Load())
        update_messagehub_node = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='message_hub', ctx=ast.Load()),
                    attr='update_info',
                    ctx=ast.Load()),
                args=[
                    ast.Constant(value=_get_tensor_key(self._target)),
                    deep_copy_attribute_node
                ],
                keywords=[]))
        return [node, if_node, update_messagehub_node]


class Recorder(metaclass=ABCMeta):

    def __init__(self, model, target: str):
        self.model = model
        self._target = target

    @abstractmethod
    def rewrite(self, ast_tree):
        pass


@RECORDERS.register_module()
class FunctionRecorder(Recorder):

    def __init__(self, model: str, target: str, index: list):
        super().__init__(model, target)
        self.index = index
        self.visit_assign = FunctionRecorderTransformer(
            self._target, self.index)

    def rewrite(self, ast_tree):
        return self.visit_assign.visit(ast_tree)


@RECORDERS.register_module()
class AttributeRecorder(Recorder):

    def __init__(self, model: str, target: str):
        super().__init__(model, target)
        self.visit_assign = AttributeRecorderTransformer(self._target)

    def rewrite(self, ast_tree):
        return self.visit_assign.visit(ast_tree)


@HOOKS.register_module()
class RecorderHook(Hook):
    priority = 'LOWEST'

    def __init__(
        self,
        recorders: Optional[List[Dict]] = None,
        print_modification: bool = True,
        save_dir: str = None,
        filename_tmpl: Optional[str] = None,
    ):
        self.tensor_dict: Dict[str, Any] = {}
        self.origin_forward = None
        self.origin_func: Dict[Any, Any] = {}
        self._recorders: Dict[str, Recorder] = {}
        self.print_modification = print_modification
        self.save_dir = save_dir  # type: ignore
        if filename_tmpl is None:
            self.filename_tmpl = 'record_epoch_{}.pth'

        if recorders is None or len(recorders) == 0:
            raise ValueError('recorders not initialized')
        for recorder in recorders:
            model = recorder.get('model')
            if model is None:
                recorder['model'] = ''
            target = recorder.get('target')
            tensor_key = _get_tensor_key(target)

            if target is None:
                print_log(
                    '`RecorderHook` cannot be initialized '
                    'because recorder has no target',
                    logger='current',
                    level=logging.WARNING)
            if recorder.get('type') == 'FunctionRecorder':
                index = recorder.get('index', 0)
                if isinstance(index, list):
                    for i in index:
                        self.tensor_dict[function_with_index(target,
                                                             i)] = list()
                elif isinstance(index, int):
                    self.tensor_dict[function_with_index(target,
                                                         index)] = list()
            elif recorder.get('type') == 'AttributeRecorder':
                self.tensor_dict[tensor_key] = list()
            self._recorders[tensor_key] = RECORDERS.build(recorder)

    def _modify_forward_func(self, func, recorders):
        # Gets the source code for the function
        source = inspect.getsource(func)
        source = textwrap.dedent(source)

        # Parse source code as ast
        tree = ast.parse(source)

        func_body = tree.body[0].body
        # import mmengine.logging.MessageHub
        import_messagehub_node = ast.ImportFrom(
            module='mmengine.logging',
            names=[ast.alias(name='MessageHub')],
            level=0)
        import_copy_node = ast.Import(names=[ast.alias(name='copy')])
        # get messagehub instance
        get_messagehub_node = ast.Assign(
            targets=[ast.Name(id='message_hub', ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='MessageHub', ctx=ast.Load()),
                    attr='get_current_instance',
                    ctx=ast.Load()),
                args=[],
                keywords=[]))

        tree.body[0].body = [
            import_messagehub_node, import_copy_node, get_messagehub_node
        ] + func_body

        for recorder in recorders:
            tree = recorder.rewrite(tree)
            if self.print_modification:
                new_tree = ast.fix_missing_locations(tree)
                modified_source_code = ast.unparse(new_tree)
                print_log(
                    f'After modification, the source code is:\n'
                    f'{modified_source_code}',
                    logger='current',
                    level=logging.INFO)
        tree = ast.fix_missing_locations(tree)

        # Compile the modified ast as a new function
        namespace = {}
        exec(
            compile(tree, filename='<ast>', mode='exec'), func.__globals__,
            namespace)
        return namespace[func.__name__]

    def _get_model(self, model_name):
        if not model_name:
            return self.base_model
        model = self.base_model
        model = attrgetter(model_name)(model)
        return model

    def _group_recorder_by_model(self):
        group_dict = {}
        for recorder in self._recorders.values():
            key = recorder.model
            if key not in group_dict:
                group_dict[key] = []
            group_dict[key].append(recorder)
        return group_dict

    def before_run(self, runner) -> None:
        if not self.save_dir:
            self.save_dir = runner.work_dir

        # get messagehub instance and store it.
        self.message_hub = MessageHub.get_current_instance()
        # get model and modify its forward function
        model = runner.model
        self.base_model = model
        self.grouped_recorders = self._group_recorder_by_model()
        for model_name, recorders in self.grouped_recorders.items():
            try:
                model = self._get_model(model_name)
            except AttributeError:
                print_log(
                    f'Can not record {model_name} in runner.model'
                    'because it doesn\'t exist',
                    logger='current',
                    level=logging.WARNING)
                continue
            self.origin_func[model] = model.forward
            model.forward = types.MethodType(
                self._modify_forward_func(model.forward, recorders), model)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
        for key in self.tensor_dict.keys():
            self.tensor_dict[key].append(self.message_hub.get_info(key))

    def _save_record(self, step):
        recorder_file_name = self.filename_tmpl.format(step)
        path = osp.join(self.save_dir, recorder_file_name)
        torch.save(self.tensor_dict, path)

    def _init_tensor_dict(self):
        for k in self.tensor_dict.keys():
            self.tensor_dict[k] = list()

    def after_train_epoch(self, runner) -> None:
        step = runner.epoch + 1
        runner.logger.info(f'Saving record at {runner.epoch + 1} epochs')
        self._save_record(step)
        self._init_tensor_dict()

    def after_train(self, runner) -> None:
        # restore forward function after train
        for m, f in self.origin_func.items():
            m.forward = f
