# Copyright (c) OpenMMLab. All rights reserved.
import ast
import inspect
from ast import *
from importlib import import_module
from typing import Optional

from mmengine import MessageHub
from .hook import Hook


class RecorderVisitor(NodeTransformer):
    def __init__(self,
                 func_name: str,
                 var: Optional[List[str]] = None,
                 resume: bool = False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.func_name = func_name
        self.recorded_buffer = dict()
        MessageHub.get_current_instance().update_info(
            func_name, self.recorded_buffer, resumed=resume)
        self.vars = var if var is not None else []

    def visit_FunctionDef(self, node: FunctionDef):
        if node.name == 'hello':
            self.flag = True
        return self.generic_visit(node)

    def generic_visit(self, node):
        print(node)
        return super().generic_visit(node)

    def visit_Return(self, node: Return):
        if self.flag:
            result = [
                eval(ast.dump(ast.parse(
                    'from mmengine import MessageHub').body[0])),
                eval(ast.dump(ast.parse(
                    'recorded_buffer = '
                    f'MessageHub.get_current_instance().get_info('
                    f'{self.func_name})').body[0]))
            ]

            for var in self.vars:
                result.append(
                eval(ast.dump(ast.parse(
                    f'recorded_buffer.update({var}=locals()["{var}"])'
                ).body[0])))

            result.append(Expr(
                value=Call(
                    func=Attribute(
                        value=Name(
                            id='recorded_buffer', ctx=Load()),
                        attr='update',
                        ctx=Load()),
                    args=[],
                    keywords=[
                        keyword(arg='output',
                                value=node.value)]))
            )
            result.append(super().generic_visit(node))
            return result


class FuncRewriter:
    def __init__(
            self,
            module_name: str,
            function_name: str,
            target_instance: Optional[str] = None,
            target_variable: Optional[List[str]] = None,
            resume: bool = False):
        self.module = import_module(module_name)
        self.ori_func = getattr(self.module, function_name)
        # TODO check orifunc should not accept "enable_rewrite"

        # Get modified function.
        act_module_path = inspect.getmodule(self.module).__file__
        self.function_name = function_name
        self.target_instance = target_instance
        visitor = RecorderVisitor(
            func_name=function_name,
            var=target_variable,
            resume=resume
        )
        with open(act_module_path) as f:
            ast_tree = ast.parse(f.read())
        ast_tree = visitor.visit(ast_tree)
        ast.fix_missing_locations(ast_tree)
        code = compile(ast_tree, '', mode='exec')
        global_dict = dict()
        eval(code, global_dict, global_dict)

        self.modified_func = global_dict[function_name]

    def patch(self, runner, *args, **kwargs):
        if self.target_instance:
            target_instance = self._get_instance_from_runner(runner)
            setattr(target_instance, self.function_name, self.modified_func)
        else:
            setattr(self.module, self.function_name, self.modified_func)

    def _get_instance_from_runner(self, runner):
        target_instance = self.target_instance.split('.')
        result = runner
        for instance in target_instance:
            result = getattr(result, instance)
        return result


class RecorderHook(Hook):
    def __init__(self,
                 rewrited_funcs: List[dict] = []
                 ):
        self.rewrite_funcs = []
        for cfg in rewrited_funcs:
            self.rewrite_funcs.append(FuncRewriter(**cfg))

    def before_run(self, runner) -> None:
