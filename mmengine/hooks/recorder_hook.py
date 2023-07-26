# Copyright (c) OpenMMLab. All rights reserved.
import ast
import dis
import inspect
import textwrap
from typing import Any, List, Optional, Tuple, Union

from mmengine.registry import HOOKS
from . import Hook


class FunctionRecorder():

    def __init__(self):
        self._data_buffer: List = list()
        self.now_epoch = 0

    @property
    def data_buffer(self) -> List:
        """list: data buffer."""
        return self._data_buffer

    def func_after_assign(self):
        pass

    def next_epoch(self):
        self.now_epoch += 1

    def get_record_data(self,
                        record_idx: int = 0,
                        data_idx: Optional[int] = None) -> Any:
        """Get data from ``data_buffer``.

        Args:
            record_idx (int): The index of the record saved in
                ``data_buffer``. If a source is executed N times during
                forward, there will be N records in ``data_buffer``.
            data_index (int, optional):  The index of target data in
                a record. A record may be a tuple or a list, if data_idx is
                None, the whole list or tuple is returned. Defaults to None.

        Returns:
            Any: The type of the return value is undefined, and different
                source data may have different types.
        """
        assert record_idx < len(self._data_buffer), \
            'record_idx is illegal. The length of data_buffer is ' \
            f'{len(self._data_buffer)}, but record_idx is ' \
            f'{record_idx}.'

        record = self._data_buffer[record_idx]

        if data_idx is None:
            target_data = record
        else:
            if isinstance(record, (list, tuple)):
                assert data_idx < len(record), \
                    'data_idx is illegal. The length of record is ' \
                    f'{len(record)}, but data_idx is {data_idx}.'
                target_data = record[data_idx]
            else:
                raise TypeError('When data_idx is not None, record should be '
                                'a list or tuple instance, but got '
                                f'{type(record)}.')

        return target_data

    def reset_data_buffer(self) -> None:
        """Clear data in data_buffer."""

        self._data_buffer = list()


# model的 存到 runner的 message_hub
class RecorderAdder(ast.NodeTransformer):

    def visit_Assign(self, node):
        # 这将创建一个新的print调用节点
        print_call = ast.Expr(
            value=ast.Call(
                func=ast.Name(id='print', ctx=ast.Load()),
                args=[
                    ast.Str(s='Assigning to variable '),
                    ast.Name(id=node.targets[0].id, ctx=ast.Load())
                ],
                keywords=[]))

        # 插入print语句
        return [node, print_call]


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

    recorder = FunctionRecorder()

    def __init__(self, ):
        pass

    def _modify_func(self, func):
        # 获取函数的源代码
        source = inspect.getsource(func)
        source = textwrap.dedent(source)

        # 解析源代码为AST
        tree = ast.parse(source)

        import_from_statement = ast.ImportFrom(
            module='mmengine.hooks',
            names=[ast.alias(name='RecorderHook', asname=None)],
            level=0)

        tree.body[0].body.insert(0, import_from_statement)

        # 修改AST
        tree = RecorderAdder().visit(tree)
        tree = ast.fix_missing_locations(tree)

        # print(ast.dump(tree, indent=4))

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
        import dis
        model = runner.model
        print('---------------------------')
        # breakpoint()
        import types

        model.forward = types.MethodType(
            self._modify_func(model.forward), model)