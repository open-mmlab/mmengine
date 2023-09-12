# Copyright (c) OpenMMLab. All rights reserved.
import ast
import inspect
import logging
import os.path as osp
import textwrap
import types
from abc import ABCMeta, abstractmethod
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
# genertate "tmp_func_self_conv1 = self.conv1(x)"
# and "x = tmp_func_self_conv1"
# and "message_hub.update_info('conv1', tmp_func_conv1)"
def _get_tensor_key(target, attribute=None):
    target = target.replace('.', '_')
    if attribute:
        target = target + '_' + attribute
    return target


class FuncCallVisitor(ast.NodeTransformer):

    def __init__(self, func_name):
        self.func_name = func_name
        self.call_nodes = []

    # judge if the ast.Call node is user wanted
    def _is_target_call(self, call_node):
        assert isinstance(call_node, ast.Call)
        call_chain_list = self.func_name.split('.')
        call_node = call_node.func
        if len(call_chain_list) == 1:
            return isinstance(
                call_node.func,
                ast.Name) and call_node.func.id == call_chain_list[0]
        else:
            # Traversal call_chain_list in reverse order
            for i in range(len(call_chain_list) - 1, 0, -1):
                if isinstance(call_node, ast.Attribute
                              ) and call_node.attr == call_chain_list[i]:
                    call_node = call_node.value
                else:
                    return False
            return isinstance(call_node,
                              ast.Name) and call_node.id == call_chain_list[0]

    def visit_Call(self, node):
        if not self._is_target_call(node):
            return node
        new_node = ast.Name(id=_get_tensor_key(self.func_name), ctx=ast.Load())
        self.call_nodes.append(node)
        return new_node


class AttributeRecorderTransformer(ast.NodeTransformer):

    def __init__(self, target, attribute):
        super().__init__()
        self._target = target
        self._attribute = attribute
        self.function_visitor = FuncCallVisitor(target)

    def _get_target_attribute(self):
        func_chain = self._target.split('.')
        func_chain.append(self._attribute)
        assert len(func_chain) >= 2
        attr = ast.Attribute(
            value=ast.Name(id=func_chain[0], ctx=ast.Load()),
            attr=func_chain[1],
            ctx=ast.Load())
        for ele in func_chain[2:]:
            attr = ast.Attribute(value=attr, attr=ele, ctx=ast.Load())
        return attr

    def visit_Assign(self, node):
        self.function_visitor.visit(node)
        if self.function_visitor.call_nodes:
            assign_right_node = self.function_visitor.call_nodes[0]
            assign_node_name = _get_tensor_key(self._target, None)
            assign_left_node = ast.Assign(
                targets=[ast.Name(id=assign_node_name, ctx=ast.Store())],
                value=assign_right_node)
            if self._attribute:
                assign_node_name = _get_tensor_key(self._target,
                                                   self._attribute)
                ast_arg2 = self._get_target_attribute()
            else:
                ast_arg2 = ast.Name(id=assign_node_name, ctx=ast.Load())
            update_messagehub_node = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='message_hub', ctx=ast.Load()),
                        attr='update_info',
                        ctx=ast.Load()),
                    args=[ast.Constant(value=assign_node_name), ast_arg2],
                    keywords=[]))
            self.function_visitor.call_nodes.clear()
            return [assign_left_node, update_messagehub_node, node]
        return node


class Recorder(metaclass=ABCMeta):

    def __init__(self, target: str):
        self._target = target

    @abstractmethod
    def rewrite(self, ast_tree):
        pass


@RECORDERS.register_module()
class FunctionRecorder(Recorder):

    def __init__(self, target: str, index: list):
        super().__init__(target)
        self.index = index
        self.visit_assign = self._get_transformer_class()

    def _get_transformer_class(self):
        return FunctionRecorderTransformer(self._target, self.index)

    def rewrite(self, ast_tree):
        return self.visit_assign.visit(ast_tree)


@RECORDERS.register_module()
class AttributeRecorder(Recorder):

    def __init__(self, target: str, attribute: str = None):
        super().__init__(target)
        self.attribute = attribute
        self.visit_assign = self._get_transformer_class()

    def _get_transformer_class(self):
        return AttributeRecorderTransformer(self._target, self.attribute)

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
        self._recorders: Dict[str, Recorder] = {}
        self.print_modification = print_modification
        self.save_dir = save_dir  # type: ignore
        if filename_tmpl is None:
            self.filename_tmpl = 'record_epoch_{}.pth'

        if recorders is None or len(recorders) == 0:
            raise ValueError('recorders not initialized')
        for recorder in recorders:
            target = recorder.get('target')
            attribute = recorder.get('attribute')
            tensor_key = _get_tensor_key(target, attribute)

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

    def _modify_func(self, func):
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

        tree.body[0].body = [import_messagehub_node, get_messagehub_node
                             ] + func_body

        for recorder in self._recorders.values():
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

    def before_run(self, runner) -> None:
        if not self.save_dir:
            self.save_dir = runner.work_dir

        # get messagehub instance and store it.
        self.message_hub = MessageHub.get_current_instance()
        # get model and modify its forward function
        model = runner.model
        self.origin_forward = model.forward
        model.forward = types.MethodType(
            self._modify_func(model.forward), model)

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
        runner.model.forward = self.origin_forward
