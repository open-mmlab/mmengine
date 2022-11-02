# Copyright (c) OpenMMLab. All rights reserved.
import ast
import inspect
import os
import sys
from abc import ABCMeta, abstractmethod
from ast import *
from functools import partial
from importlib import import_module
from typing import List, Optional, Union

from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS, TASK_UTILS
from mmengine.utils import ManagerMixin
from .hook import DATA_BATCH, Hook


class RecorderVisitor(NodeTransformer):

    def __init__(self,
                 func_name: str,
                 recorded_name: Optional[str] = None,
                 var: Optional[List[str]] = None,
                 class_name: Optional[str] = None,
                 resume: bool = False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.func_name = func_name
        self.recorded_name = recorded_name if recorded_name is not None else func_name
        self.var = var
        self.class_name = class_name
        self.resume = resume
        self.vars = var if var is not None else []
        self.function_flag = False
        self.class_flag = False
        self.class_name = class_name

    def visit_FunctionDef(self, node: FunctionDef):
        # function_name could be 'class_name.function_name'
        if self.func_name.endswith(node.name):
            self.function_flag = True
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ClassDef):
        if node.name == self.class_name:
            self.class_flag = True
        return self.generic_visit(node)

    def visit_Return(self, node: Return):
        if not self.function_flag:
            return super().generic_visit(node)

        if self.class_name is not None and not self.class_flag:
            return super().generic_visit(node)

        result = []
        result.append(
            eval(
                ast.dump(
                    ast.parse('from mmengine import MessageHub').body[0])))
        result.append(
            eval(
                ast.dump(
                    ast.parse('recorded_buffer = '
                              f'MessageHub.get_current_instance().get_info('
                              f'"{self.recorded_name}")').body[0])))

        for var in self.vars:
            result.extend([
                eval(
                    ast.dump(
                        ast.parse(f'var = recorded_buffer["{var}"]').body[0])),
                eval(
                    ast.dump(
                        ast.parse(f'var.append(locals()["{var}"])').body[0]))
            ])

        result.append(
            Expr(
                value=Call(
                    func=Attribute(
                        value=Subscript(
                            value=Name(id='recorded_buffer', ctx=Load()),
                            slice=Index(
                                value=Constant(value='output', kind=None)),
                            ctx=Load()),
                        attr='append',
                        ctx=Load()),
                    args=[node.value],
                    keywords=[])))
        result.append(super().generic_visit(node))
        return result


class BaseRecorder(metaclass=ABCMeta):

    def __init__(self, recorded_name=None):
        self.recorded_name = recorded_name
        self.message_hub_cls = import_module('mmengine').MessageHub

    @abstractmethod
    def initialize(self, instance):
        pass

    @abstractmethod
    def deinitialize(self, instance):
        pass

    @abstractmethod
    def clear(self):
        pass


class RecorderManager(ManagerMixin):

    def __init__(self, name, recorders: List[Union[dict, BaseRecorder]]):
        super().__init__(name=name)
        self.recorders: List[BaseRecorder] = []
        for recorder in recorders:
            if isinstance(recorder, dict):
                self.recorders.append(TASK_UTILS.build(recorder))
            elif isinstance(recorder, BaseRecorder):
                self.recorders.append(recorder)
            else:
                raise TypeError()

    def initialize(self, instance):
        for recorder in self.recorders:
            recorder.initialize(instance)

    def deinitialize(self, instance):
        for recorder in self.recorders:
            recorder.deinitialize(instance)

    def clear(self):
        for recorder in self.recorders:
            recorder.clear()


@TASK_UTILS.register_module()
class AttributeGetterRecorder(BaseRecorder):

    def __init__(self, target_attributes, **kwargs):
        super().__init__(**kwargs)
        self.target_attributes_list = target_attributes
        if not isinstance(target_attributes, list):
            self.target_attributes_list = [target_attributes]
        if self.recorded_name is None:
            self.recorded_name = 'attributes'

    def initialize(self, instance):
        results = []
        for target_attributes in self.target_attributes_list:
            result = instance
            for target_attribute in target_attributes.split('.'):
                result = getattr(result, target_attribute)
            results.append(result)
        self.message_hub_cls.get_current_instance().update_info(
            self.recorded_name, results)

    def deinitialize(self, instance):
        pass

    def clear(self):
        pass


@TASK_UTILS.register_module()
class FuncRewriterRecorder(BaseRecorder):

    def __init__(self,
                 function: str,
                 target_instance: Optional[str] = None,
                 target_variable: Optional[List[str]] = None,
                 resume: bool = False,
                 indices: int = 0,
                 **kwargs):
        super().__init__(**kwargs)
        self.module, self.class_type, self.ori_func = \
            self.get_module_and_function(function)
        target_variable = (
            target_variable if target_variable is not None else [])
        self.function_name = self.ori_func.__name__
        self.target_instance = target_instance

        if self.class_type is not None:
            self.act_module = inspect.getmodule(self.class_type)
            self.class_name = self.class_type.__name__
            self.recoded_func_name = f'{self.class_name}.{self.function_name}'

        else:
            self.act_module = inspect.getmodule(self.ori_func)
            self.class_name = self.class_type
            self.recoded_func_name = self.function_name

        if self.recorded_name is None:
            self.recorded_name = self.recoded_func_name

        act_module_path = self.act_module.__file__

        visitor = RecorderVisitor(
            func_name=self.function_name,
            recorded_name=self.recorded_name,
            var=target_variable,
            class_name=self.class_name,
            resume=resume)
        with open(act_module_path) as f:
            ast_tree = ast.parse(f.read())
        ast_tree = visitor.visit(ast_tree)
        ast.fix_missing_locations(ast_tree)
        code = compile(ast_tree, '', mode='exec')
        # Overwrites package for relative imports
        global_dict = dict(__package__=self.act_module.__package__)
        eval(code, global_dict, global_dict)
        if self.class_type is not None:
            self.modified_func = getattr(global_dict[self.class_name],
                                         self.function_name)
        else:
            self.modified_func = global_dict[self.function_name]

        self.recorded_buffer = {var: [] for var in target_variable}
        self.recorded_buffer['output'] = []
        self.message_hub_cls.get_current_instance().update_info(
            self.recorded_name, self.recorded_buffer, resumed=resume)

    def get_module_and_function(self, full_function_name):
        try:
            module_name, function_name = full_function_name.rsplit('.', 1)
            module = import_module(module_name)
            function = getattr(module, function_name)
            class_type = None
        except (ModuleNotFoundError, AttributeError):
            module, class_type, function = None, None, None

        if module is None:
            try:
                module_name, class_name, function_name = \
                    full_function_name.rsplit('.', 2)
                module = import_module(module_name)
                class_type = getattr(module, class_name)
                function = getattr(class_type, function_name)
            except (ModuleNotFoundError, AttributeError) as e:
                raise e
        return module, class_type, function

    def initialize(self, instance):
        if self.target_instance:
            target_instance = self.get_target_instance(instance)
            setattr(target_instance, self.function_name,
                    partial(self.modified_func, target_instance))
        elif self.class_type is not None:
            # Both module and act_module  have a reference for target function.
            setattr(self.class_type, self.function_name, self.modified_func)
            setattr(self.act_module, self.class_name, self.class_type)
            setattr(self.module, self.class_name, self.class_type)
        else:
            setattr(self.act_module, self.function_name, self.modified_func)
            setattr(self.module, self.function_name, self.modified_func)

    def deinitialize(self, instance):
        if self.target_instance:
            target_instance = self.get_target_instance(instance)
            setattr(target_instance, self.function_name, self.ori_func)
        elif self.class_type is not None:
            setattr(self.class_type, self.function_name, self.ori_func)
        else:
            setattr(self.module, self.function_name, self.ori_func)
            setattr(self.act_module, self.function_name, self.ori_func)

    def get_target_instance(self, instance):
        target_instance = self.target_instance.split('.')
        result = instance
        for instance in target_instance:
            result = getattr(result, instance)
        return result

    def clear(self):
        message_hub = self.message_hub_cls.get_current_instance()
        for value in message_hub.get_info(self.recorded_name).values():
            value.clear()


@HOOKS.register_module()
class RecorderHook(Hook):

    priority = 'VERY_LOW'

    def __init__(self, recorders: List[Union[dict, BaseRecorder]] = []):
        self.recorder_manager = RecorderManager(
            name='mmengine', recorders=recorders)

    def before_train(self, runner) -> None:
        if is_model_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model
        self.recorder_manager.initialize(model)

    def after_train(self, runner) -> None:
        if is_model_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model
        self.recorder_manager.deinitialize(model)

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:
        return self.recorder_manager.clear()

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        return self.recorder_manager.clear()
