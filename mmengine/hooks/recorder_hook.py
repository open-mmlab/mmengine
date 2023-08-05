# Copyright (c) OpenMMLab. All rights reserved.
import ast
import inspect
import textwrap
import types
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional

from mmengine.logging import MessageHub
from mmengine.registry import HOOKS, RECORDERS
from . import Hook


class AttributeRecorderAdder(ast.NodeTransformer):

    def __init__(self, target):
        super().__init__()
        self._target = target

    def visit_Assign(self, node):
        if node.targets[0].id != self._target:
            return node
        update_messagehub_node = ast.Expr(
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

        return [node, update_messagehub_node]


# Take "x = self.conv1(x)" as an example
# genertate "tmp_func_self_conv1 = self.conv1(x)"
# and "x = tmp_func_self_conv1"
# and "message_hub.update_info('conv1', tmp_func_conv1)"
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
            assign_right_node = self.function_visitor.call_nodes[0]
            assign_node_name = get_node_name(self._target.replace('.', '_'))
            assign_left_node = ast.Assign(
                targets=[ast.Name(id=assign_node_name, ctx=ast.Store())],
                value=assign_right_node)
            update_messagehub_node = ast.Expr(
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
            return [assign_left_node, update_messagehub_node, node]
        return node


class Recorder(metaclass=ABCMeta):

    def __init__(self, target: str):
        self._target = target

    @abstractmethod
    def rewrite(self, ast_tree):
        pass


@RECORDERS.register_module()
class AttributeRecorder(Recorder):

    def __init__(self, target: str):
        super().__init__(target)
        self.visit_assign = self._get_adder_class()

    def _get_adder_class(self):
        return AttributeRecorderAdder(self._target)

    def rewrite(self, ast_tree):
        new_ast_tree = self.visit_assign.visit(ast_tree)
        new_ast_tree = ast.fix_missing_locations(new_ast_tree)

        modified_source_code = ast.unparse(new_ast_tree)
        print(modified_source_code)

        return new_ast_tree


@RECORDERS.register_module()
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

    def __init__(self, recorders: Optional[List[Dict]] = None):
        self.tensor_dict: Dict[str, Any] = {}
        self.origin_forward = None
        self._recorders: Dict[str, Recorder] = {}
        if recorders is None:
            raise ValueError('recorders not initialized')
        for recorder in recorders:
            assert recorder.get('target') is not None
            self.tensor_dict[recorder['target']] = list()
            self._recorders[recorder['target']] = RECORDERS.build(recorder)

    def _get_ast(source_code):
        return ast.parse(source_code)

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
        tree = ast.fix_missing_locations(tree)

        # Compile the modified ast as a new function
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

    def after_train(self, runner) -> None:
        runner.model.forward = self.origin_forward
