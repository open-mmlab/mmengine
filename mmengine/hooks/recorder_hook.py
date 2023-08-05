# Copyright (c) OpenMMLab. All rights reserved.
import ast
import dis
import inspect
import textwrap
import types
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Any, List, Optional, Tuple, Union

from mmengine.logging import HistoryBuffer, MessageHub
from mmengine.registry import HOOKS
from . import Hook


class AttributeRecorderAdder(ast.NodeTransformer):

    def __init__(self, target):
        super().__init__()
        self._target = target

    def visit_Assign(self, node):
        if node.targets[0].id != self._target:
            return node
        add2messagehub = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='message_hub', ctx=ast.Load()),
                    attr='update_info',
                    ctx=ast.Load()),
                args=[
                    ast.Constant(value=node.targets[0].id),
                    ast.Name(id=node.targets[0].id, ctx=ast.Load())
                ],
                keywords=[]))

        # 插入print语句
        return [node, add2messagehub]


def get_node_name(func_name):
    return 'tmp_func_' + func_name


class FuncCallVisitor(ast.NodeTransformer):

    def __init__(self, func_name):
        self.func_name = func_name
        self.call_nodes = []

    def is_target_call(self, call_node):
        assert isinstance(call_node, ast.Call)
        call_node = call_node.func
        call_chain_list = self.func_name.split('.')
        if len(call_chain_list) == 1:
            return isinstance(
                call_node.func,
                ast.Name) and call_node.func.id == call_chain_list[0]
        else:
            # 倒序遍历call_chain_list
            for i in range(len(call_chain_list) - 1, 0, -1):
                print(ast.dump(call_node))
                if isinstance(call_node, ast.Attribute
                              ) and call_node.attr == call_chain_list[i]:
                    call_node = call_node.value
                else:
                    return False
            return isinstance(call_node,
                              ast.Name) and call_node.id == call_chain_list[0]

    def visit_Call(self, node):
        if not self.is_target_call(node):
            return node
        new_node = ast.Name(
            id=get_node_name(self.func_name.replace('.', '_')), ctx=ast.Load())
        self.call_nodes.append(node)
        return new_node


class FunctionRecorderAdder(ast.NodeTransformer):

    def __init__(self, target):
        super().__init__()
        self._target = target
        self.function_visitor = FuncCallVisitor(target)

    def visit_Assign(self, node):
        self.function_visitor.visit(node)
        if self.function_visitor.call_nodes:
            assign_node = self.function_visitor.call_nodes[0]
            assign_node_name = get_node_name(self._target.replace('.', '_'))
            # test = assign node
            assign = ast.Assign(
                targets=[
                    ast.Name(
                        id=assign_node_name,
                        ctx=ast.Store())
                ],
                value=assign_node)
            add2messagehub = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='message_hub', ctx=ast.Load()),
                        attr='update_info',
                        ctx=ast.Load()),
                    args=[
                        ast.Constant(value=self._target),
                        ast.Name(id=assign_node_name, ctx=ast.Load())
                    ],
                    keywords=[]))
            self.function_visitor.call_nodes.clear()
            return [assign, add2messagehub, node]
        return node


class Recorder(metaclass=ABCMeta):

    def __init__(self, target: str):
        self._target = target

    @abstractmethod
    def rewrite(self, ast_tree):
        pass


# AttributeRecorder
class AttributeRecorder(Recorder):

    def __init__(self, target: str):
        super().__init__(target)
        self.visit_assign = self._get_adder_class()

    # super.__init__()

    def _get_adder_class(self):
        return AttributeRecorderAdder(self._target)

    def rewrite(self, ast_tree):
        new_ast_tree = self.visit_assign.visit(ast_tree)
        new_ast_tree = ast.fix_missing_locations(new_ast_tree)

        modified_source_code = ast.unparse(new_ast_tree)
        print(modified_source_code)

        return new_ast_tree


class FunctionRecorder(Recorder):

    def __init__(self, target: str):
        super().__init__(target)
        self.visit_assign = self._get_adder_class()

    def _get_adder_class(self):
        return FunctionRecorderAdder(self._target)

    def rewrite(self, ast_tree):
        new_ast_tree = self.visit_assign.visit(ast_tree)
        new_ast_tree = ast.fix_missing_locations(new_ast_tree)

        modified_source_code = ast.unparse(new_ast_tree)
        print(modified_source_code)

        return new_ast_tree


@HOOKS.register_module()
class RecorderHook(Hook):
    priority = 'LOWEST'

    # RECORDER_MESSAGEHUB_NAME = "_recorder"

    # recorder = AttributeRecorder()

    def __init__(self, ):
        self.tensor_dict = defaultdict(list)
        self.origin_forward = None
        pass

    def _get_ast(source_code):
        return ast.parse(source_code)

    def _modify_func(self, func):
        # 获取函数的源代码
        source = inspect.getsource(func)
        source = textwrap.dedent(source)

        # 解析源代码为AST
        tree = ast.parse(source)

        import_from_statement = ast.ImportFrom(
            module='mmengine.logging.MessageHub',
            names=[ast.alias(name='RecorderHook', asname=None)],
            level=0)

        func_body = tree.body[0].body
        import_messagehub_statement = ast.ImportFrom(
            module='mmengine.logging',
            names=[ast.alias(name='MessageHub')],
            level=0)
        add_message_hub = ast.Assign(
            targets=[ast.Name(id='message_hub', ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='MessageHub', ctx=ast.Load()),
                    attr='get_current_instance',
                    ctx=ast.Load()),
                args=[],
                keywords=[]))

        tree.body[0].body = [import_messagehub_statement, add_message_hub
                             ] + func_body
        # tree.body[0].body.insert(0, import_statement)

        # 修改AST
        # tree = AttributeRecorder('x').rewrite(tree)
        tree = FunctionRecorder('self.resnet').rewrite(tree)
        tree = ast.fix_missing_locations(tree)

        # 编译修改后的AST为一个新的函数
        namespace = {}
        exec(
            compile(tree, filename='<ast>', mode='exec'), func.__globals__,
            namespace)
        return namespace[func.__name__]

    def before_run(self, runner) -> None:
        """Check `stop_training` variable in `runner.train_loop`.

        Args:
            runner (Runner): The runner of the training process.
        """
        log_scalars = dict(loss=HistoryBuffer())
        runtime_info = dict()
        resumed_keys = dict(loss=True)
        self.message_hub2 = MessageHub.get_current_instance()

        model = runner.model
        print('---------------------------')
        # breakpoint()
        self.origin_forward = model.forward

        model.forward = types.MethodType(
            self._modify_func(model.forward), model)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:

        # print(self.message_hub2.__dict__)
        # print(self.message_hub2.get_info("task"))
        self.tensor_dict['task'].append(self.message_hub2.get_info('task'))

    # def before_train(self, runner) -> None:
    #     model = runner.model

    #     model.forward = types.MethodType(
    #         self._modify_func(model.forward), model)

    def after_train(self, runner) -> None:
        runner.model.forward = self.origin_forward
