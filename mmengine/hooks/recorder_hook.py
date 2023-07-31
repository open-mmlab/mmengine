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
        outer_class = self

        class AttributeRecorderAdder(ast.NodeTransformer):

            def visit_Assign(self, node):
                if node.targets[0].id != outer_class._target:
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

        return AttributeRecorderAdder()

    def rewrite(self, ast_tree):
        return self.visit_assign.visit(ast_tree)


# model的 存到 runner的 message_hub
class RecorderAdder(ast.NodeTransformer):

    def visit_Assign(self, node):
        add2messagehub = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='message_hub', ctx=ast.Load()),
                    attr='update_info',
                    ctx=ast.Load()),
                args=[
                    ast.Constant(value='task'),
                    ast.Name(id=node.targets[0].id, ctx=ast.Load())
                ],
                keywords=[]))

        # 插入print语句
        return [node, add2messagehub]


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
        # breakpoint()
        tree = AttributeRecorder('x').rewrite(tree)
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
        runtime_info = dict()
        resumed_keys = dict(loss=True)
        # # create `MessageHub` from data.
        # self.message_hub2 = MessageHub(
        #     name=RecorderHook.RECORDER_MESSAGEHUB_NAME,
        #     log_scalars=log_scalars,
        #     runtime_info=runtime_info,
        #     resumed_keys=resumed_keys)
        # self.message_hub2.update_info("task", "1111")
        # self.message_hub2.update_info("task", {1231312: "dfasfd"})
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

    def before_train(self, runner) -> None:
        model = runner.model

        model.forward = types.MethodType(
            self._modify_func(model.forward), model)

    def after_train(self, runner) -> None:
        runner.model.forward = self.origin_forward