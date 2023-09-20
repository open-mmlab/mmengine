# Copyright (c) OpenMMLab. All rights reserved.
import ast
import inspect
import logging
import os.path as osp
import textwrap
import types
from abc import ABCMeta, abstractmethod
from operator import attrgetter
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from mmengine.logging import MessageHub, print_log
from mmengine.registry import HOOKS, RECORDERS
from . import Hook


class FunctionRecorderTransformer(ast.NodeTransformer):
    """Transformer that modifies the Abstract Syntax Tree (AST) for function-
    related record updates.

    The transformer is responsible for updating the AST to add the logic needed
    to record tensor data at specific indices when a function is called within
    the model's forward pass.

    Args:
        model (str): The name or identifier of the model.
        method (str): The method in which the transformer operates.
        target (str): The target function to be recorded.
        target_index (int or list): Index of var to record.
    """

    def __init__(self, model: str, method: str, target: str,
                 target_index: Union[int, List[int]]):
        super().__init__()
        self._model = model
        self._method = method
        self._target = target
        if isinstance(target_index, list):
            self._target_index = set(target_index)
        else:
            self._target_index = {target_index}
        self.count = -1

    def get_store_varname_with_index(self, index: int) -> str:
        """Generate and return the variable name with the specified index.

        Args:
            index (int): The index for which to generate the variable name.

        Returns:
            str: The variable name for the given index.
        """
        return f'{self._model}:{self._method}:{self._target}@{index}'

    def visit_Assign(self, node: ast.Assign) -> Union[Any, List[Any]]:
        """Visit and possibly transform an assignment node in the AST.

        Args:
            node: The AST node being visited.

        Returns:
            Modified AST node or a list of AST nodes.
        """
        assert isinstance(node.targets[0], ast.Name)
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
                        value=self.get_store_varname_with_index(self.count)),
                    ast.Name(id=node.targets[0].id, ctx=ast.Load())
                ],
                keywords=[]))

        return [node, update_messagehub_node]


class AttributeRecorderTransformer(ast.NodeTransformer):
    """Transformer that modifies the Abstract Syntax Tree (AST) for attribute-
    related record updates.

    The transformer is responsible for updating the AST to add the logic needed
    to record tensor data from model attributes during the forward pass.

    Args:
        model (str): The name or identifier of the model.
        method (str): The method in which the transformer operates.
        target (str): The target attribute to be recorded.
    """

    def __init__(self, model, method, target):
        super().__init__()
        self._model = model
        self._method = method
        self._target = target
        self._visited = False

    def _get_target_attribute(self) -> ast.Attribute:
        """Extract and return the target attribute from the AST as a node.

        Returns:
            ast.Attribute: The node representing the target attribute.
        """
        func_chain = self._target.split('.')
        assert len(func_chain) >= 2
        attr = ast.Attribute(
            value=ast.Name(id=func_chain[0], ctx=ast.Load()),
            attr=func_chain[1],
            ctx=ast.Load())
        for ele in func_chain[2:]:
            attr = ast.Attribute(value=attr, attr=ele, ctx=ast.Load())
        return attr

    def _deepcopy_varname(self) -> str:
        """Generate and return a variable name for the deep copy of the target
        attribute.

        Returns:
            str: The variable name for the deep copy of the target attribute.
        """
        return f'_deep_copy_{self._target.replace(".", "_")}'

    def _get_tensor_name(self) -> str:
        """Generate and return the tensor name for the target attribute.

        Returns:
            str: The tensor name for the target attribute.
        """
        return f'{self._model}:{self._method}:{self._target}'

    def _get_deep_copy_node(self, var_node) -> ast.If:
        """Generate and return the AST node for deep copying the target
        attribute.

        Args:
            var_node (ast.Name):
            The AST node representing the variable to be deep copied.

        Returns:
            ast.If: The `if` node for deep copying the target attribute.
        """
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

    def visit_Assign(self, node: ast.Assign) -> Union[Any, List[Any]]:
        """Visit and possibly transform an assignment node in the AST.

        Args:
            node: The AST node being visited.

        Returns:
            Modified AST node or a list of AST nodes.
        """
        if self._visited:
            return node
        # insert update attribute node after message_hub assign node
        if isinstance(node.targets[0],
                      ast.Name) and node.targets[0].id == 'message_hub':
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
                    ast.Constant(value=self._get_tensor_name()),
                    deep_copy_attribute_node
                ],
                keywords=[]))
        return [node, if_node, update_messagehub_node]


class Recorder(metaclass=ABCMeta):
    """Abstract base class for implementing tensor data recorders.

    The Recorder is intended to be a blueprint for creating specific recorder
    types to capture tensor data during model forward passes.

    Args:
        model: The name or identifier of the model.
        method: The method on which the Recorder is attached.
        target (str): The target layer or tensor to be recorded.
    """

    def __init__(self, model: str, method: str, target: str):
        self._model = model
        self._method = method
        self._target = target

    @abstractmethod
    def rewrite(self, ast_tree) -> Any:
        """Rewrite the AST tree to include recording logic.

        Args:
            ast_tree: The Abstract Syntax Tree to be rewritten.

        Returns:
            Modified AST tree.
        """
        pass

    @abstractmethod
    def get_store_varname(self) -> Union[str, List[str]]:
        """Get the variable name used for storing recorded data.

        Returns:
            Variable name or a list of variable names.
        """
        pass


@RECORDERS.register_module()
class FunctionRecorder(Recorder):
    """A Recorder implementation to capture output tensor data from function
    calls.

    This Recorder hooks into specific function calls within the model's forward
    pass and records tensor data at specified indices.

    Args:
        model (str): The name or identifier of the model.
        method (str): The method on which the Recorder is attached.
        target (str): The target function to be recorded.
        index (list): List of indices within the function call to record.
    """

    def __init__(self, model: str, method: str, target: str, index: list):
        super().__init__(model, method, target)
        self.index = index
        self.visit_assign = FunctionRecorderTransformer(
            self._model, self._method, self._target, self.index)

    def rewrite(self, ast_tree) -> Any:
        """Rewrite the AST tree to include recording logic for output of
        function calls."""
        return self.visit_assign.visit(ast_tree)

    def get_store_varname(self) -> List[str]:
        """Generate and return variable names based on output name.

        Outputs with the same name will be distinguished based on the index
        number.
        """
        return [
            f'{self._model}:{self._method}:{self._target}@{i}'
            for i in self.index
        ]


@RECORDERS.register_module()
class AttributeRecorder(Recorder):
    """A Recorder implementation to capture tensor data from model attributes.

    This Recorder hooks into model attributes and records their tensor data
    during the forward pass.

    Args:
        model (str): The name or identifier of the model.
        method (str): The method on which the Recorder is attached.
        target (str): The target attribute to be recorded.
    """

    def __init__(self, model: str, method: str, target: str):
        super().__init__(model, method, target)
        self.visit_assign = AttributeRecorderTransformer(
            self._model, self._method, self._target)

    def rewrite(self, ast_tree) -> Any:
        """Rewrite the AST tree to include recording logic for attributes."""
        return self.visit_assign.visit(ast_tree)

    def get_store_varname(self) -> str:
        """Generate and return variable name based on model attributes."""
        return f'{self._model}:{self._method}:{self._target}'


@HOOKS.register_module()
class RecorderHook(Hook):
    """A hook to record information during model training.

    This hook allows users to modify and record certain model variables
    during training iterations and save them for analysis purposes.
    It provides the ability to modify any function of a model
    using ast module in python.

    Args:
        recorders (Optional[List[Dict]]):
            Configurations for individual recorders.
            Each recorder dict should contain the target model and method.
        print_modification (bool): Whether to print the modified source code
            after it's been altered by a recorder. Defaults to True.
        save_dir (str): The directory where recorded data will be saved.
            If not specified, it will use the runner's work directory.
            Defaults to None.
        filename_tmpl (Optional[str]): The filename template used when saving
            recorded data. If not provided, a default template will be used.
            Defaults to None.

    Examples:
        >>> recorder_hook_cfg = dict(
        ...     recorders=[{'model': 'runner_model',
        ...                'target': 'layer1', 'method': 'forward'}],
        ...     print_modification=True,
        ...     save_dir='./records',
        ...     filename_tmpl='record_epoch_{}.pth'
        ... )
    """
    priority = 'LOWEST'

    def __init__(
        self,
        recorders: Optional[List[Dict]] = None,
        print_modification: bool = True,
        save_dir: str = None,
        filename_tmpl: Optional[str] = None,
    ):
        self.tensor_dict: Dict[str, Any] = {}
        self.origin_methods: Dict[Any, Any] = {}
        self._recorders: List[Recorder] = []
        self.print_modification: bool = print_modification
        self.save_dir: Optional[str] = save_dir  # type: ignore
        if filename_tmpl is None:
            self.filename_tmpl = 'record_epoch_{}.pth'
        else:
            self.filename_tmpl = filename_tmpl

        if recorders is None or len(recorders) == 0:
            raise ValueError('recorders not initialized')
        for recorder in recorders:
            model = recorder.get('model')
            if model is None:
                recorder['model'] = 'runner_model'
            target = recorder.get('target')
            method = recorder.get('method')
            if method is None:
                recorder['method'] = 'forward'

            if target is None:
                print_log(
                    '`RecorderHook` cannot be initialized '
                    'because recorder has no target',
                    logger='current',
                    level=logging.WARNING)
            self._recorders.append(RECORDERS.build(recorder))

    def _modify_forward_func(self, func: Callable,
                             recorders: List[Recorder]) -> Callable:
        """Modify the forward function to incorporate recording behaviors.

        Args:
            func (callable): Original forward function to modify.
            recorders (List[Recorder]): List of recorder instances.

        Returns:
            callable: Modified forward function.
        """
        # Gets the source code for the function
        source = inspect.getsource(func)
        source = textwrap.dedent(source)

        # Parse source code as ast
        tree = ast.parse(source)
        breakpoint()
        if isinstance(tree.body[0], ast.FunctionDef):
            func_body = tree.body[0].body
        else:
            raise ValueError(
                "Unexpected node type that doesn't have a body attribute.")
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
                    attr='get_instance',
                    ctx=ast.Load()),
                args=[ast.Constant(value='recorder_hook')],
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
        namespace: Dict[str, Any] = {}
        if isinstance(func, types.FunctionType):
            globals_dict = func.__globals__
            func_name = func.__name__
        else:
            raise TypeError('It is not a function')
        exec(
            compile(tree, filename='<ast>', mode='exec'), globals_dict,
            namespace)
        return namespace[func_name]

    def _get_model(self, model_name: str) -> Any:
        """Retrieve a specific model from runner.
        If model_name == 'runner_model', return runner.model.
        Else, return runner.model.model_name
        Args:
            model_name (str): Name of the model to retrieve.

        Returns:
            Model: Requested model instance.
        """
        if not model_name or model_name == 'runner_model':
            return self.base_model
        model = self.base_model
        model = attrgetter(model_name)(model)
        return model

    def _group_recorder_by_model_method(
            self) -> Dict[str, Dict[str, List[Recorder]]]:
        """Group recorders by model and method.

        Returns:
            dict: Grouped recorders.
        """
        group_model_dist = {}
        group_model_method_dict: Dict[str, Dict[str, List[Recorder]]] = {}
        for recorder in self._recorders:
            key = recorder._model
            if key not in group_model_dist:
                group_model_dist[key] = [recorder]
            else:
                group_model_dist[key].append(recorder)
        for model_name, recorders in group_model_dist.items():
            group_model_method_dict[
                model_name] = self._group_recorder_by_method(recorders)
        return group_model_method_dict

    def _group_recorder_by_method(
            self, recorders: List[Recorder]) -> Dict[str, List[Recorder]]:
        """Group recorders by method.

        Args:
            recorders (List[Recorder]): List of recorder instances.

        Returns:
            dict: Grouped recorders.
        """
        group_dict: Dict[str, List[Recorder]] = {}
        for recorder in recorders:
            key = recorder._method
            if key not in group_dict:
                group_dict[key] = [recorder]
            group_dict[key].append(recorder)
        return group_dict

    def _save_origin_method(self, model: Any, method_name: str,
                            origin_method: Callable) -> None:
        """Save reference to the original method of a model.

        Args:
            model (Model): Model instance.
            method_name (str): Name of the method to save.
            origin_method (callable): Original method to save.
        """
        if model not in self.origin_methods:
            self.origin_methods[model] = {}
        self.origin_methods[model][method_name] = origin_method

    def before_run(self, runner) -> None:
        """Prepare for training by modifying methods for recording.

        Args:
            runner (Runner): The runner of the training process.
        """
        if not self.save_dir:
            self.save_dir = runner.work_dir

        # get messagehub instance and store it.
        self.message_hub = MessageHub.get_instance('recorder_hook')
        # init_save_var_dict
        self._init_tensor_dict()
        # get model and modify its forward function
        self.base_model = runner.model
        self.grouped_recorders = self._group_recorder_by_model_method()
        for model_name, group_method_recorders in self.grouped_recorders.items(
        ):
            try:
                model = self._get_model(model_name)
            except AttributeError:
                print_log(
                    f'Can not record {model_name} in runner.model '
                    'because it doesn\'t exist',
                    logger='current',
                    level=logging.WARNING)
                continue
            for method_name, recorders in group_method_recorders.items():
                try:
                    method = getattr(model, method_name)
                except AttributeError:
                    print_log(
                        f'Can not record {method_name} in {model_name}'
                        'because it doesn\'t exist',
                        logger='current',
                        level=logging.WARNING)
                    continue
                # self.origin_methods[model][method_name] = method
                print_log(
                    f'Modify {method_name} in {model_name}',
                    logger='current',
                    level=logging.INFO)
                self._save_origin_method(model, method_name, method)
                new_method = types.MethodType(
                    self._modify_forward_func(method, recorders), model)
                setattr(model, method_name, new_method)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
        """Record specific tensors after each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): Index of the current batch.
            data_batch (Optional): Current data batch. Default is None.
            outputs (Optional): Outputs from the current iteration.
        """
        for key in self.tensor_dict.keys():
            self.tensor_dict[key].append(self.message_hub.get_info(key))

    def _save_record(self, step):
        """Save recorded tensors to disk.

        Args:
            step (int): Current training epoch.
        """
        recorder_file_name = self.filename_tmpl.format(step)
        path = osp.join(self.save_dir, recorder_file_name)
        torch.save(self.tensor_dict, path)

    def _init_tensor_dict(self):
        """Initialize the tensor dictionary for recording."""
        for recorder in self._recorders:
            varname = recorder.get_store_varname()
            if isinstance(varname, list):
                for name in varname:
                    self.tensor_dict[name] = list()
            else:
                self.tensor_dict[varname] = list()

    def after_train_epoch(self, runner) -> None:
        """Save recorded tensors after each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        step = runner.epoch + 1
        runner.logger.info(f'Saving record at {runner.epoch + 1} epochs')
        self._save_record(step)
        self._init_tensor_dict()

    def after_train(self, runner) -> None:
        """Restore original methods after training.

        Args:
            runner (Runner): The runner of the training process.
        """
        for model, v in self.origin_methods.items():
            for method_name, origin_method in v.items():
                setattr(model, method_name, origin_method)