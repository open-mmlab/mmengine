# Copyright (c) OpenMMLab. All rights reserved.
import ast
import dis
import types
import inspect
import textwrap
from typing import Any, List, Optional, Tuple, Union

from mmengine.registry import HOOKS
from mmengine.logging import MessageHub, HistoryBuffer
from . import Hook

# model的 存到 runner的 message_hub
class RecorderAdder(ast.NodeTransformer):

    def visit_Assign(self, node):
        add2messagehub = ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='message_hub', ctx=ast.Load()), attr='update_info', ctx=ast.Load()),
                                         args=[ast.Constant(value='task'), ast.Name(id=node.targets[0].id, ctx=ast.Load())], keywords=[]))

        # 插入print语句
        return [node, add2messagehub]


# class RecorderAdder(ast.NodeTransformer):
#     def visit_Assign(self, node):
#         # 这将创建一个新的print调用节点
#         print_call = ast.Expr(
#             value=ast.Call(
#                 func=ast.Name(id='RecorderHook', ctx=ast.Load()),
#                 attr='add2buffer',
#                 args=[
#                     ast.Name(id=node.targets[0].id, ctx=ast.Load())
#                 ],
#                 keywords=[]
#             )
#         )
#
#         # 插入print语句
#         return [node, print_call]


@HOOKS.register_module()
class RecorderHook(Hook):
    priority = 'LOWEST'

    # recorder = FunctionRecorder()

    def __init__(self, ):
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
        import_statement = ast.ImportFrom(module='mmengine.logging', names=[ast.alias(name='MessageHub')], level=0)
        add_message_hub = ast.Assign(targets=[ast.Name(id='message_hub', ctx=ast.Store())], value=ast.Call(func=ast.Attribute(value=ast.Name(id='MessageHub', ctx=ast.Load()), attr='get_instance', ctx=ast.Load()), args=[ast.Constant(value='mmengine')], keywords=[]))
        tree.body[0].body = [import_statement, add_message_hub] + func_body
        # tree.body[0].body.insert(0, import_statement)

        # 修改AST
        tree = RecorderAdder().visit(tree)
        tree = ast.fix_missing_locations(tree)

        print(ast.dump(tree, indent=4))

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
        runtime_info = dict(task='task')
        resumed_keys = dict(loss=True)
         # create `MessageHub` from data.
        message_hub2 = MessageHub(
            name = 'name',
            log_scalars = log_scalars,
            runtime_info = runtime_info,
            resumed_keys = resumed_keys)
        model = runner.model
        print('---------------------------')
        # breakpoint()

        model.forward = types.MethodType(
            self._modify_func(model.forward), model)
