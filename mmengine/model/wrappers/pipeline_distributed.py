# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from functools import partial
from queue import Queue
from threading import Event, Thread, current_thread
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from mmengine.analysis import FlopAnalyzer
from mmengine.registry import MODEL_WRAPPERS
from mmengine.utils import apply_to
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils.version_utils import digit_version


@dataclass
class TreeNode:
    """The tree node for the model tree。"""
    # All the attributes have default values for
    # empty initialization and later modification
    module: Optional[nn.Module] = None
    parameter_size: int = 0
    flops: int = 0
    exec_order: Optional[int] = None
    parameters: Optional[Dict[str, Optional[nn.Parameter]]] = None
    buffers: Optional[Dict[str, Optional[torch.Tensor]]] = None
    submodules: Optional[Dict[str, 'TreeNode']] = None
    tied_parameters: Optional[Dict[str, List[str]]] = None


@MODEL_WRAPPERS.register_module()
class MMPipelineParallel(nn.Module):
    """The model wrapper for pipeline parallelism.

    A model wrpper for pipeline parallelism, which is ONLY applicable to
    the model inference. The wrapper will build the model on the meta device,
    get the memory map of the devices including cpu and gpus, infer the device
    map to split the model, and dispatch the weights to the corresponding
    devices. Then, the wrapper can perform the model inference with pipeline
    parallelism.

    Args:
        module (nn.Module): The model to be wrapped.
            The model must implement the function `test_step`.
        weights (str, optional): The path of the weights.
            Defaults to None.
        num_pipelines (int, optional): The number of pipelines.
            Defaults to None.
        num_chunks (int, optional): The number of mini-batches.
            Defaults to None.
        memory_threshold (float): The memory threshold to avoid
            OOM. Defaults to 0.7.
        memory_map (dict, optional): The memory map of devices.
            Defaults to None.
        no_split_module_classes (list, optional): The module classes which
            contains skip connection so that they should not be split. If a
            module contains a skip connection but is split, it will result
            in a device error. Defaults to None.
        language_module_class (str, optional): The module class of language
            model, such as LlamaForCausalLM, OPTForCausalLM. If there is a
            language module whose function `generate` will be called, we should
            explicitly specify this parameter. Defaults to None.
        device_map (str or dict): The device map policy or the device
            map. Defaults to device_map_policy 'balanced'.
        offload_directory (str, optional): The directory to store offloaded
            weights when the disk offload is on. If disk offload is required
            but `offload_directory` is not specified, a temporaryfolder will
            be created. Defaults to None.
        input_key (str): The key of the input in the data dict, which depends
            on the data preprocessor. Most of models use `inputs` while the
            multimodal models of MMPreTrain use `images`. Defaults to 'inputs'.

    Examples:
        >>> # An example with memory map and device map
        >>> memory_map = {
        ...     'cpu': '16GB',
        ...     'cuda:0': '24GB',
        ...     'cuda:1': '12GB',
        ... }
        >>> device_map = {
        ...     'layer1': {
        ...         'part_id': 0,
        ...         'init_device': 'cuda:0',
        ...         'exec_device': 'cuda:0',
        ...     },
        ...     'layer2': {
        ...         ...
        ...     },
        ... }
        >>> model_wrapper_cfg = dict(
        ...     type='MMPipelineParallel',
        ...     weights='checkpoint.pth',
        ...     num_pipelines=2,
        ...     num_chunks=32,
        ...     memory_map=memory_map,
        ...     device_map=device_map)
        >>> # A common case
        >>> model_wrapper_cfg = dict(
        ...    type='MMPipelineParallel',
        ...    weights='checkpoint.pth',
        ...    num_pipelines=2)
    """

    def __init__(self,
                 module: nn.Module,
                 weights: Optional[str] = None,
                 num_pipelines: Optional[int] = None,
                 num_chunks: Optional[int] = None,
                 memory_threshold: float = 0.7,
                 memory_map: Optional[Dict[str, str]] = None,
                 no_split_module_classes: Optional[List[str]] = None,
                 language_module_class: Optional[str] = None,
                 device_map: Union[str, Dict[str, dict]] = 'balanced',
                 offload_directory: Optional[str] = None,
                 input_key: str = 'inputs'):

        if digit_version(TORCH_VERSION) < digit_version('2.0.0'):
            raise RuntimeError(
                'MMPipelineParallel should work with PyTorch >= 2.0.0')

        super().__init__()

        # init model
        self.module_state_dict: Dict[str, torch.Tensor] = {}
        self.module = self._init_model(module)
        self.weights = weights
        self.input_key = input_key
        if not hasattr(self.module, 'test_step'):
            raise NotImplementedError('The function `test_step`' +
                                      'must be implemented')

        # init pipeline parallelism
        if num_pipelines is not None:
            self.num_pipelines = num_pipelines
            if self.num_pipelines > torch.cuda.device_count():
                warnings.warn('The number of pipelines is larger than ' +
                              'the number of GPUs. ' +
                              'There may be some unpredictable bugs.')
        else:
            # TODO multi-node pipeline parallel
            self.num_pipelines = torch.cuda.device_count()

        if num_chunks is not None:
            self.num_chunks = num_chunks
        else:
            # because we need a value to init num_chunks
            self.num_chunks = self.num_pipelines * 32

        self.in_queues: Dict[str, Queue] = {}
        self.out_queues: Dict[str, Queue] = {}
        self.events: List[List[Event]] = []
        self.hook_visited_times: Dict[str, int] = {}

        # init memory map
        self.memory_threshold = memory_threshold
        self.memory_map = _init_memory_map(memory_map, self.memory_threshold)

        # init device map
        self.no_split_module_classes = no_split_module_classes or []
        self.language_module_class = language_module_class
        self.lm_offset = 0
        self.module_tree = self._get_model_tree()
        self.device_map_policy = device_map
        self.is_inited = False

        # init offload directory
        if offload_directory is not None:
            self.offload_directory = offload_directory
        else:
            self.offload_directory = tempfile.mkdtemp()
        self.offloaded_weights: Dict[str, Dict[str, Any]] = {}

        # init queues
        self.in_queues, self.out_queues = self._init_queues()

        # init events
        self.events = self._init_events()

    def _prepare_forward(self, chunked_data: dict):
        if isinstance(self.device_map_policy, dict):
            self.device_map = self.device_map_policy
        else:
            self._get_flops_and_exec_order(chunked_data)
            self._find_tied_weights()
            self.device_map = self._init_device_map(self.device_map_policy)
        self.offload_map = self._init_offload_map()
        self.module_map = self._init_module_map()
        self._load_and_dispatch()
        self._register_hooks()

    def _init_model(self, model: nn.Module) -> nn.Module:
        """Init the model on the meta device and store the current weight."""
        # store the current weight for later use
        # because some buffers are not stored in the weights
        for n, p in model.named_parameters():
            self.module_state_dict[n] = p
        for n, b in model.named_buffers():
            self.module_state_dict[n] = b
        return model.to('meta')

    def _get_model_tree(self) -> TreeNode:
        """Init the model tree for many usages."""

        def _generate_model_tree(module: Optional[nn.Module], prefix: str,
                                 info: TreeNode):
            """BFS the module to generate the model tree.

            First, register the module self as a node. Then, register the
            buffers as the children of the node. Next, register the submodules.
            Last, for every submodule, if it is not in no_split_module_classes,
            do the bfs recursively.
            """
            # None
            if module is None:
                return
            # self
            info.module = module
            info.parameter_size = _parameter_size(module)
            info.flops = 0
            info.exec_order = None
            # parameter
            if len(module._parameters) != 0:
                info.parameters = {}
                for name, param in module._parameters.items():
                    curr_name = name if prefix == '' else f'{prefix}.{name}'
                    info.parameters[curr_name] = param
            # buffer
            if len(module._buffers) != 0:
                info.buffers = {}
                for name, buffer in module._buffers.items():
                    curr_name = name if prefix == '' else f'{prefix}.{name}'
                    info.buffers[curr_name] = buffer
            # submodule
            module_class_name = module.__class__.__name__
            if not (len(module._modules) == 0
                    or module_class_name in self.no_split_module_classes):
                info.submodules = {}
                for name, submodule in module._modules.items():
                    curr_name = name if prefix == '' else f'{prefix}.{name}'
                    info.submodules[curr_name] = TreeNode()
                    _generate_model_tree(submodule, curr_name,
                                         info.submodules[curr_name])

        tree: TreeNode = TreeNode()
        _generate_model_tree(self.module, '', tree)
        return tree

    def _iter_tree(
        self, module_name: str
    ) -> Optional[Union[nn.Parameter, torch.Tensor, TreeNode]]:
        """Get the tree where the name of its root is module_name."""
        tree = self.module_tree
        if module_name == '':
            return tree
        else:
            name_split = module_name.split('.')
            for i in range(len(name_split)):
                curr_name = '.'.join(name_split[:i + 1])
                if i == len(name_split) - 1:
                    # leaf node
                    if tree.parameters is not None:
                        if curr_name in tree.parameters:
                            # it is parameter
                            return tree.parameters[curr_name]
                    if tree.buffers is not None:
                        if curr_name in tree.buffers:
                            # it is buffer
                            return tree.buffers[curr_name]
                    # it is submodule
                    if tree.submodules is None:
                        break
                    else:
                        return tree.submodules[curr_name]
                else:
                    # due to no_split_module_classes
                    if tree.submodules is None:
                        break
                    else:
                        tree = tree.submodules[curr_name]
        # if not found
        return None

    def _find_tied_weights(self):
        """Find the tied weights in the model."""

        def _find_tied_parameters(module: nn.Module,
                                  named_parameters: Dict[str, torch.Tensor],
                                  prefix: str = '',
                                  result: Dict[str, List[str]] = {}):
            # The tied parameters will not be in the full named_parameters
            # but in the named_parameters of the submodule.
            if named_parameters is None:
                named_parameters = {n: p for n, p in module.named_parameters()}
            else:
                for name, param in module.named_parameters():
                    full_name = name if prefix == '' else f'{prefix}.{name}'
                    if full_name in named_parameters:
                        continue
                    for new_name, new_param in named_parameters.items():
                        if new_param is param:
                            if new_name not in result:
                                result[new_name] = []
                            full_name_split = full_name.split('.')
                            module_name = '.'.join(full_name_split[:-1])
                            result[new_name].append(module_name)
            # handle submodule
            for name, submodule in module.named_children():
                sub_name = name if prefix == '' else f'{prefix}.{name}'
                _find_tied_parameters(submodule, named_parameters, sub_name,
                                      result)

        result = {}
        _find_tied_parameters(self.module, None, '', result)
        # remove suffix of keys
        new_result = {}
        for k, v in result.items():
            new_k = '.'.join(k.split('.')[:-1])
            new_result[new_k] = v
        # merge into model tree
        self.module_tree.tied_parameters = new_result

    def _get_flops_and_exec_order(self, data: dict):
        """Get the flops of each module."""
        # preprocess
        self.module.data_preprocessor.to_empty(device='cpu')  # type: ignore
        self.module.data_preprocessor.to('cpu')
        inputs = self.module.data_preprocessor(  # type: ignore
            data, training=False)
        # prepare exec order
        exec_order = []
        module_name_map = {m: n for n, m in self.module.named_modules()}

        def return_module_name_hook(module: nn.Module, args: tuple):
            if module in module_name_map:
                module_name = module_name_map[module]
                if module_name not in exec_order:
                    exec_order.append(module_name)

        handle = nn.modules.module.register_module_forward_pre_hook(
            return_module_name_hook)

        # prevent circular import
        from mmengine.model import BaseDataPreprocessor

        def empty_hook(module: nn.Module, input: Any):
            if not isinstance(module, BaseDataPreprocessor):
                module.to_empty(device='cpu')

        def meta_hook(module: nn.Module, input: Any, output: Any):
            # avoid meta device when post processing
            if not isinstance(module, BaseDataPreprocessor):
                module.to('meta')

        empty_handle = nn.modules.module.register_module_forward_pre_hook(
            empty_hook)

        meta_handle = nn.modules.module.register_module_forward_hook(meta_hook)

        # hack the forward
        old_forward = self.module.forward

        def new_forward():
            input_data = inputs[self.input_key]
            data_samples = inputs['data_samples']
            # data_sample or list of data_sample
            if isinstance(data_samples, list):
                data_sample_dict = data_samples[0].to_dict()
            else:
                data_sample_dict = data_samples.to_dict()
            result = old_forward(input_data, data_samples, 'predict')
            result = result[0].to_dict()
            results = []
            for k, v in result.items():
                if k in data_sample_dict:
                    # remove input keys
                    continue
                if isinstance(v, (tuple, list, dict, str, torch.Tensor)):
                    # JIT Trace supports these types
                    results.append(v)
            return results

        self.module.forward = new_forward

        with FakeTensorMode(allow_non_fake_inputs=True):  # type: ignore
            flop_analyzer = FlopAnalyzer(self.module, inputs=())
        flops = flop_analyzer.by_module()
        handle.remove()
        empty_handle.remove()
        meta_handle.remove()
        # merge exec order into model tree
        for order, name in enumerate(exec_order):
            tree = self._iter_tree(name)
            if isinstance(tree, TreeNode):
                tree.exec_order = order
        # merge flops into model tree
        for name, num_flops in flops.items():
            tree = self._iter_tree(name)
            if isinstance(tree, TreeNode):
                tree.flops = num_flops

        self.module.forward = old_forward

    def _init_device_map(self, device_map_policy: str) -> Dict[str, dict]:
        """Init the device map of the model."""
        if device_map_policy == 'balanced':
            return self._init_device_map_balanced()
        else:
            raise ValueError(
                f'Unsupported device map policy {device_map_policy}')

    def _init_device_map_balanced(self) -> Dict[str, dict]:
        """Init the device map of the model with balanced policy."""
        avg_flops = self.module_tree.flops / self.num_pipelines
        # To get approximate solution
        # we choose 1.2 * avg_flops as the upper bound
        # 1 * avg_flops as the lower bound
        upper_flops = 1.2 * avg_flops
        lower_flops = 1 * avg_flops
        modules_info: List[Dict[str, Any]] = []
        # every item in modules_info is a dict
        # which contains the following keys:
        # modules, init, exec, flops, parameter_size
        devices = list(self.memory_map.keys())
        cuda_devices = [d for d in devices if d.startswith('cuda')]
        # part_pointer is the pointer to the current part
        # cuda_pointer is the pointer to the current cuda device
        meta_info = {
            'part_pointer': 0,
            'cuda_pointer': 0,
        }

        for _ in range(self.num_pipelines):
            modules_info.append({
                'modules': [],
                'flops': 0,
                'parameter_size': 0,
                'init': None,
                'exec': None
            })

        def _get_device_map(tree: TreeNode, name: str, meta_info: Dict[str,
                                                                       int]):
            """DFS the model tree to get the device map.

            First, handle language model. Second, get the current module flops
            and size. If the current device can hold the current module and the
            current flops does not exceed the average flops, put the current
            module into the modules and update the meta info. Third, if the
            current module is not handled, but it has submodules, handle the
            buffers and do the dfs recursively.
            """
            # handle language model
            if tree.module.__class__.__name__ == self.language_module_class:
                if meta_info['part_pointer'] != 0:
                    meta_info['part_pointer'] += 1
                if meta_info['cuda_pointer'] != 0:
                    meta_info['cuda_pointer'] += 1
                    meta_info['cuda_pointer'] %= len(cuda_devices)
                    self.lm_offset = meta_info['part_pointer']
            # prepare
            part_pointer = meta_info['part_pointer']
            cuda_pointer = meta_info['cuda_pointer']
            # handle self
            module_flops = tree.flops
            curr_flops = modules_info[part_pointer]['flops']
            module_size = tree.parameter_size
            curr_size = modules_info[part_pointer]['parameter_size']
            curr_cuda_memory = self.memory_map[cuda_devices[cuda_pointer]]
            is_memory_enough = curr_size + module_size <= curr_cuda_memory
            # infer exec
            # if it is the last part
            if part_pointer == self.num_pipelines - 1:
                if is_memory_enough:
                    modules_info[part_pointer]['modules'].append(name)
                    modules_info[part_pointer]['flops'] += module_flops
                    modules_info[part_pointer]['parameter_size'] += module_size
                    modules_info[part_pointer]['exec'] = cuda_devices[
                        cuda_pointer]
                else:
                    raise RuntimeError('The model if too large to fit into' +
                                       f'{cuda_devices[cuda_pointer]}' +
                                       'Please use more GPUs')
                return
            # otherwise, handle self
            if is_memory_enough:
                if module_flops + curr_flops < upper_flops:
                    modules_info[part_pointer]['modules'].append(name)
                    modules_info[part_pointer]['flops'] += module_flops
                    modules_info[part_pointer]['parameter_size'] += module_size
                    modules_info[part_pointer]['exec'] = cuda_devices[
                        cuda_pointer]
                    if module_flops + curr_flops > lower_flops:
                        meta_info['part_pointer'] += 1
                        meta_info['cuda_pointer'] += 1
                        meta_info['cuda_pointer'] %= len(cuda_devices)
                    return
            # handle submodules when
            # memory is not enough / flops is too large
            if tree.submodules is not None:
                if tree.parameters is not None:
                    for name, _ in tree.parameters.items():
                        modules_info[part_pointer]['modules'].append(name)
                if tree.buffers is not None:
                    for name, _ in tree.buffers.items():
                        modules_info[part_pointer]['modules'].append(name)
                for name, submodule in tree.submodules.items():
                    _get_device_map(submodule, name, meta_info)
            else:
                # put it into the next part
                meta_info['part_pointer'] += 1
                meta_info['cuda_pointer'] += 1
                meta_info['cuda_pointer'] %= len(cuda_devices)
                _get_device_map(tree, name, meta_info)

        _get_device_map(self.module_tree, '', meta_info)
        # check part_pointer
        num_non_empty_parts = 0
        for i in range(self.num_pipelines):
            num_non_empty_parts += (len(modules_info[i]['modules']) != 0)
        if num_non_empty_parts != self.num_pipelines:
            raise RuntimeError('The model cannot be split into ' +
                               f'{self.num_pipelines} parts. ' +
                               'Try to reduce the number of pipelines to' +
                               f'less or equal than {num_non_empty_parts}')
        # infer init
        cpu_used = 0
        for i in range(self.num_pipelines):
            if i < len(cuda_devices):
                # One gpu can be used for inference for multiple parts
                # but only for initialization of one part
                modules_info[i]['init'] = cuda_devices[i]
            else:
                cpu_used += modules_info[i]['parameter_size']
                if cpu_used > self.memory_map['cpu']:
                    # if the cpu memory is not enough
                    # we can use disk offload
                    modules_info[i]['init'] = 'disk'
                else:
                    modules_info[i]['init'] = 'cpu'

        # format modules
        device_map = {}
        for i in range(self.num_pipelines):
            modules_i = modules_info[i]
            for module in modules_i['modules']:
                device_map[module] = {
                    'part_id': i,
                    'init_device': modules_i['init'],
                    'exec_device': modules_i['exec']
                }
        # handle tied weights
        tied_weights = self.module_tree.tied_parameters or {}
        for source, targets in tied_weights.items():
            source_info = device_map[source]
            for target in targets:
                target_info = device_map[target]
                target_info['init_device'] = source_info['init_device']
                target_info['exec_device'] = source_info['exec_device']
        import json
        with open(f'../{self.module.__class__.__name__}.json', 'w') as f:
            json.dump(device_map, f, indent=4)
        return device_map

    def _init_offload_map(self) -> Dict[int, int]:
        """Init the offload map of the model."""
        curr_part_id = -1
        offload_map = {}
        for info in self.device_map.values():
            if info['part_id'] != curr_part_id:
                curr_part_id = info['part_id']
            # 0 is offload, 1 is onload
            if info['init_device'] == 'cpu' or \
                    info['init_device'] == 'disk':
                offload_map[curr_part_id] = 0
            else:
                offload_map[curr_part_id] = 1
        return offload_map

    def _init_module_map(self) -> Dict[str, dict]:
        """Init the module map of the model."""
        module_map = {}
        for name, info in self.device_map.items():
            tree = self._iter_tree(name)
            if isinstance(tree, dict):
                # it is a submodule
                module = tree['self']
            else:
                # it is a buffer or parameter
                module = tree
            module_map[name] = {
                'module': module,
                'curr_device': info['init_device'],
                'part_id': info['part_id'],
            }
        return module_map

    def _init_queues(self) -> Tuple[Dict[str, Queue], Dict[str, Queue]]:
        """Init the move queues and execution queues."""
        in_queues, out_queues = {}, {}
        # init move queues
        for move in ['out', 'in']:
            in_queue: Queue = Queue()
            out_queue: Queue = Queue()
            # create the thread corresponding to the move queues
            thread = Thread(
                target=self._worker,
                args=(in_queue, out_queue),
                name=f'move-{move}',
                daemon=True)
            thread.start()

            in_queues[f'move-{move}'] = in_queue
            out_queues[f'move-{move}'] = out_queue
        # init execution queues
        for i in range(self.num_chunks):
            in_queue = Queue()
            out_queue = Queue()
            # create the thread for each chunk
            # because we cannot split the module
            thread = Thread(
                target=self._worker,
                args=(in_queue, out_queue),
                name=f'chunk-{i}',
                daemon=True,
            )
            thread.start()

            in_queues[f'chunk-{i}'] = in_queue
            out_queues[f'chunk-{i}'] = out_queue
        return in_queues, out_queues

    def _init_events(self) -> List[List[Event]]:
        """Init the events for synchronization."""
        events = []
        for _ in range(self.num_chunks):
            events.append([Event() for _ in range(self.num_pipelines)])
        return events

    def _load_and_dispatch(self):
        """Dispatch the weight to the corresponding devices."""
        # load weights
        if self.weights is not None:
            from mmengine.runner.checkpoint import CheckpointLoader
            ckpt = CheckpointLoader.load_checkpoint(self.weights)
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            # update weights
            for param_name in ckpt:
                # some parameters are not stored at ckpt
                # so we need module_state_dict
                # therefore there should be
                # ckpt <= module_state_dict
                self.module_state_dict[param_name] = ckpt[param_name]
        # dispatch weights
        modules_weights: Dict[str, Dict[str, torch.Tensor]]
        modules_weights = {k: {} for k in self.device_map.keys()}
        for weight_name, param in self.module_state_dict.items():
            name_split = weight_name.split('.')
            is_found = False
            # find the module that the weight belongs to
            for i in range(len(name_split)):
                curr_name = '.'.join(name_split[:i + 1])
                if curr_name in self.device_map:
                    is_found = True
                    init_device = self.device_map[curr_name]['init_device']
                    if init_device == 'disk':
                        # it should be offloaded
                        dtype = None
                        if param.dtype == torch.bfloat16:
                            param = param.to(torch.int16)
                            dtype = 'bfloat16'
                        array = param.cpu().numpy()
                        if array.ndim == 0:
                            array = array[None]
                        if not os.path.exists(self.offload_directory):
                            os.makedirs(self.offload_directory, exist_ok=True)
                        # save the param info
                        self.offloaded_weights[weight_name] = {
                            'dtype': array.dtype if dtype is None else dtype,
                            'shape': list(array.shape),
                        }
                        offload_path = os.path.join(self.offload_directory,
                                                    f'{weight_name}.npy')
                        # offload
                        file_array: np.memmap
                        file_array = np.memmap(
                            offload_path,
                            dtype=array.dtype,
                            mode='w+',
                            shape=array.shape)
                        file_array[:] = array[:]
                        file_array.flush()
                    else:
                        param = param.to(init_device)
                        modules_weights[curr_name][weight_name] = param
            if not is_found:
                # the keys of device map is ''
                init_device = self.device_map['']['init_device']
                param = param.to(init_device)
                modules_weights[''][weight_name] = param
        # load
        for module_name, module_weights in modules_weights.items():
            init_device = self.device_map[module_name]['init_device']
            module = self.module_map[module_name]['module']
            # update the curr_device
            self.module_map[module_name]['curr_device'] = init_device
            self._load_state_dict(module, module_weights, module_name,
                                  init_device)
        del self.module_state_dict

    def _register_hooks(self):
        """Register the hooks."""
        curr_part_id = -1
        for name, info in self.device_map.items():
            module = self.module_map[name]['module']
            if not isinstance(module, nn.Module):
                continue
            if info['part_id'] != curr_part_id:
                # new part
                curr_part_id = info['part_id']
                hook = _MMPipelineParallelHook(
                    part_id=curr_part_id,
                    num_parts=self.num_pipelines,
                    exec_device=info['exec_device'],
                    is_part_begin=True,
                    out_queues=self.out_queues,
                    events=self.events,
                    hook_visited_times=self.hook_visited_times)
            else:
                # curr part
                hook = _MMPipelineParallelHook(
                    part_id=curr_part_id,
                    num_parts=self.num_pipelines,
                    exec_device=info['exec_device'],
                    is_part_begin=False,
                    out_queues=self.out_queues,
                    events=self.events,
                    hook_visited_times=self.hook_visited_times)

            module.register_forward_pre_hook(hook, with_kwargs=True)
        # send back to data_preprocessor's device
        if 'data_preprocessor' in self.device_map:
            device = self.device_map['data_preprocessor']['exec_device']

            def send_back(module: nn.Module, input: Any, output: Any):
                output = apply_to(output, lambda x: hasattr(x, 'to'),
                                  lambda x: x.to(device))
                return output

            module.register_forward_hook(send_back)

    @staticmethod
    @torch.no_grad()
    def _worker(in_queue: Queue, out_queue: Queue):
        """The worker for execution and moving."""
        while True:
            task = in_queue.get()
            if task is None:
                break
            try:
                result = task()
            except Exception:
                info = sys.exc_info()
                out_queue.put((False, info))
                continue
            out_queue.put((True, result))
        done = (False, None)
        out_queue.put(done)

    def _clock(self, num_chunks: int) -> Iterator[Union[List[List[int]], str]]:
        """The clock for generating schedules."""
        chunk_id = 0
        schedules: List[List[int]] = []
        waiting_offload_chunks: Set[int] = set()
        schedule = 'exec'
        first_part_id = -1
        # blocked part id as resume point
        blocked_part_id = -1
        while True:
            if schedule == 'offload':
                # after offload, we should onload
                schedule = 'onload'
            else:
                if schedule == 'onload':
                    # after onload, we should resume
                    schedule = 'exec'
                    schedules = []
                    chunk_id = 0
                    first_part_id = blocked_part_id
                if chunk_id < num_chunks:
                    # create schedules
                    schedules.append([chunk_id, first_part_id])
                for i in range(len(schedules)):
                    # check whether schedules can be executed
                    curr_part_id = schedules[i][1]
                    next_part_id = (curr_part_id + 1) % self.num_pipelines
                    curr_chunk_id = schedules[i][0]
                    # the next part is offloaded
                    if self.offload_map[next_part_id] == 0:
                        waiting_offload_chunks.add(curr_chunk_id)
                        blocked_part_id = curr_part_id
                        schedules.pop(i)
                    else:
                        schedules[i][1] += 1
                chunk_id += 1
            if len(waiting_offload_chunks) == num_chunks:
                schedule = 'offload'
                waiting_offload_chunks = set()
            if schedule == 'exec':
                yield schedules
            else:
                yield schedule

    def _load_state_dict(self, module: nn.Module,
                         state_dict: Dict[str, torch.Tensor], module_name: str,
                         device: str):
        """Load the state dict to the module on meta device."""
        if len(state_dict) == 0:
            # for some data_preprocessors without params
            if hasattr(module, 'device'):
                module.to(device)
            return
        for name, param in state_dict.items():
            # remove prefix
            # when pipelines = 1, module_name is ''
            if name.startswith(module_name) and module_name != '':
                name = name[len(module_name) + 1:]
            if name == '':
                # because they are parameters or buffers
                # they cannot be stored by pointers
                # we need to get it
                # such as cls_token, pos_embed
                # but modules can be stored by pointers
                new_module = self.module
                name = module_name
            else:
                new_module = module
            name_split = name.split('.')
            # find the module
            for i in name_split[:-1]:
                new_module = getattr(new_module, i)
            param_name = name_split[-1]
            is_buffer = param_name in new_module._buffers
            if is_buffer:
                # just replace the old buffer
                new_module._buffers[param_name] = param
            else:
                # create a new tensor because the old tensor is
                # on the meta device, we cannot call `load_state_dict`
                # directly, so we should create a new tensor on the
                # target device and replace the old tensor
                new_tensor = nn.Parameter(param, requires_grad=False)
                new_module._parameters[param_name] = new_tensor
        # because data_preprocessor needs device to cast data
        # we have to set it
        if hasattr(new_module, 'device'):
            new_module.to(device)

    def _move_part(self, part_id: int, target_device: str):
        """Move the part to the target device."""
        for module_name, info in self.module_map.items():
            if info['part_id'] != part_id:
                continue
            if info['curr_device'] == target_device:
                continue
            if info['curr_device'] == 'disk':
                # load from disk
                param_names = [n for n, _ in info['module'].named_parameters()]
                state_dict = {}
                # prepare the state dict
                for param_name in param_names:
                    full_name = f'{module_name}.{param_name}'
                    param_info = self.offloaded_weights[full_name]
                    param_path = os.path.join(self.offload_directory,
                                              full_name)
                    dtype = param_info['dtype']
                    shape = tuple(param_info['shape'])
                    if shape == ():
                        shape = (1, )
                    if param_info['dtype'] == torch.bfloat16:
                        dtype = 'int16'
                    # load from disk
                    array: np.memmap = np.memmap(
                        param_path, dtype=dtype, mode='r', shape=shape)
                    if len(param_info['shape']) == 0:
                        array = array[0]
                    weight = torch.as_tensor(array)
                    # convert to the target dtype
                    if param_info['dtype'] == 'bfloat16':
                        weight = weight.to(torch.bfloat16)
                    weight = weight.to(target_device)
                    state_dict[full_name] = weight
                # load
                self._load_state_dict(info['module'], state_dict, module_name,
                                      target_device)
            elif target_device == 'disk':
                # disk offload
                for param_name, param in info['module'].named_paramters():
                    full_name = f'{module_name}.{param_name}'
                    dtype = None
                    if param.dtype == torch.bfloat16:
                        param = param.to(torch.int16)
                        dtype = 'bfloat16'
                    array = param.cpu().numpy()
                    if array.ndim == 0:
                        array = array[None]
                    if not os.path.exists(self.offload_directory):
                        os.makedirs(self.offload_directory, exist_ok=True)
                    # save the param info
                    self.offloaded_weights[full_name] = {
                        'dtype': array.dtype if dtype is None else dtype,
                        'shape': list(array.shape),
                    }
                    offload_path = os.path.join(self.offload_directory,
                                                f'{full_name}.npy')
                    # offload
                    file_array: np.memmap
                    file_array = np.memmap(
                        offload_path,
                        dtype=array.dtype,
                        mode='w+',
                        shape=array.shape)
                    file_array[:] = array[:]
                    file_array.flush()
            else:
                # just move
                module = info['module']
                module.to(target_device)
                info['curr_device'] = target_device

    def forward(self, data: dict) -> List[Any]:
        """The forward function of the model."""
        exec_info = None
        chunked_data = _chunk_data(data, self.num_chunks)
        results: List[List[Any]] = [[] for _ in range(len(chunked_data))]
        # get flops, init device map, offload map, module map and exec order
        # we should get the shape of input data, so we can
        # get the flops of each module and the execution order
        # after that, we can init the device map, offload map
        # load the weights and dispatch them to the corresponding devices
        if not self.is_inited:
            self._prepare_forward(chunked_data[0])
            self.is_inited = True

        num_chunks = min(len(chunked_data), self.num_chunks)
        # record finished chunks
        finished_chunks: Set[int] = set()
        # clear visited times
        self.hook_visited_times = {}
        # main loop
        offloaded_part_id, onloaded_part_id = -1, -1
        for schedules in self._clock(num_chunks):
            if len(finished_chunks) == num_chunks:
                break
            if isinstance(schedules, str):
                if schedules == 'offload':
                    # get the first offloaded part
                    # and the first onloaded part
                    for i in range(len(self.offload_map)):
                        if self.offload_map[i] == 0 and \
                                offloaded_part_id == -1:
                            offloaded_part_id = i
                        if self.offload_map[i] == 1 and \
                                onloaded_part_id == -1:
                            onloaded_part_id = i
                    # get the target device
                    if offloaded_part_id != -1:
                        target_device = None
                        for info in self.device_map.values():
                            if info['part_id'] == offloaded_part_id:
                                target_device = info['exec_device']
                                break
                        # offload the first onloaded part
                        task = partial(self._move_part, onloaded_part_id,
                                       target_device)
                        self.in_queues['move-out'].put(task)
                        self.offload_map[onloaded_part_id] = 0
                elif schedules == 'onload':
                    # because 'onload' is after 'offload'
                    # the first offloaded part id does not change
                    target_device = None
                    for info in self.device_map.values():
                        if info['part_id'] == onloaded_part_id:
                            target_device = info['init_device']
                            break
                    # onload the first offloaded part
                    task = partial(self._move_part, offloaded_part_id,
                                   target_device)
                    self.in_queues['move-in'].put(task)
                    self.offload_map[offloaded_part_id] = 1
                    offloaded_part_id, onloaded_part_id = -1, -1
            else:
                # send data
                for chunk_id, part_id in schedules:
                    if chunk_id in finished_chunks:
                        continue
                    else:
                        # get the current part id and the next part id
                        curr_part_id = part_id % self.num_pipelines
                        next_part_id = (part_id + 1) % self.num_pipelines
                        # if there is a language model
                        # the generate function will call the forward function
                        # many times, so we should add the offset
                        if part_id >= self.num_pipelines:
                            curr_part_id += self.lm_offset
                            next_part_id += self.lm_offset
                        # lock the next event
                        self.events[chunk_id][next_part_id].clear()
                        # unlock the current event
                        self.events[chunk_id][curr_part_id].set()
                        # new task
                        if part_id == 0:
                            curr_data = chunked_data[chunk_id]
                            task = partial(
                                self.module.test_step,  # type: ignore
                                curr_data)
                            self.in_queues[f'chunk-{chunk_id}'].put(task)
            if isinstance(schedules, str):
                if schedules == 'offload':
                    # receive the success signal
                    success, result = self.out_queues['move-out'].get()
                    if exec_info is not None:
                        continue
                    elif not success:
                        exec_info = result
                        continue
                elif schedules == 'onload':
                    # receive the success signal
                    success, result = self.out_queues['move-in'].get()
                    if exec_info is not None:
                        continue
                    elif not success:
                        exec_info = result
                        continue
            else:
                # recv data
                for chunk_id, part_id in schedules:
                    if chunk_id in finished_chunks:
                        continue
                    else:
                        success, result = self.out_queues[
                            f'chunk-{chunk_id}'].get()
                        if exec_info is not None:
                            continue
                        elif not success:
                            exec_info = result
                            continue
                        if not isinstance(result, _MMPipelineParallelFlag):
                            # it is not a flag, it is a result
                            # this chunk is finished
                            results[chunk_id] = result
                            finished_chunks.add(chunk_id)
            if exec_info is not None:
                raise exec_info[0].with_traceback(exec_info[1], exec_info[2])
        # merge results
        merged_results: List[Any] = []
        for result_item in results:
            merged_results.extend(result_item)
        return merged_results

    def train_step(self, data: dict) -> Any:
        raise NotImplementedError(
            'MMPipelineParallel wrapper cannot be used to train a model')

    def val_step(self, data: dict) -> Any:
        results = self(data)
        return results

    def test_step(self, data: dict) -> Any:
        results = self(data)
        return results


def _init_memory_map(memory_map: Optional[Dict[str, str]],
                     memory_threshold: float) -> Dict[str, float]:
    """Check or get the memory map of the cpu and gpus."""
    new_memory_map: Dict[str, float] = {}
    # cpu
    cpu_memory = psutil.virtual_memory().available
    new_memory_map['cpu'] = cpu_memory
    # gpu
    for i in range(torch.cuda.device_count()):
        gpu_memory = torch.cuda.get_device_properties(i).total_memory
        new_memory_map[f'cuda:{i}'] = gpu_memory
    # check memory map
    if memory_map is not None:
        converted_memory_map = _convert_memory_map(memory_map)
        for device, memory in converted_memory_map.items():
            if device not in new_memory_map:
                raise ValueError(f'{device} is not a valid device.' +
                                 'Please use cpu or cuda:i')
            if memory > new_memory_map[device]:
                raise ValueError(
                    f'The memory of {device} ({memory}) ' +
                    'is larger than the available memory ' +
                    f'({new_memory_map[device]}). Please use a smaller one')
            new_memory_map[device] = memory
    # apply threshold
    for k in new_memory_map.keys():
        new_memory_map[k] *= memory_threshold
    return new_memory_map


def _convert_memory_map(memory_map: Dict[str, str]) -> Dict[str, int]:
    """Convert the memory map from string to int."""
    converted_memory_map = {}
    for device, memory in memory_map.items():
        if memory.upper().endswith('GIB'):
            size = int(memory[:-3]) * (1024**3)
        elif memory.upper().endswith('MIB'):
            size = int(memory[:-3]) * (1024**2)
        elif memory.upper().endswith('KIB'):
            size = int(memory[:-3]) * 1024
        elif memory.upper().endswith('GB'):
            size = int(memory[:-2]) * (10**9)
            if memory.endswith('b'):
                size = size // 8
        elif memory.upper().endswith('MB'):
            size = int(memory[:-2]) * (10**6)
            if memory.endswith('b'):
                size = size // 8
        elif memory.upper().endswith('KB'):
            size = int(memory[:-2]) * (10**3)
            if memory.endswith('b'):
                size = size // 8
        else:
            raise ValueError(
                f'{memory} is not in a valid format.' +
                'Please use GiB, MiB, KiB, GB, MB or KB, e.g. 6GB')
        converted_memory_map[device] = size
    return converted_memory_map


def _parameter_size(module: nn.Module) -> int:
    """Get the parameter size of a module."""
    size = 0
    for _, param in module.named_parameters():
        size += param.nelement() * param.element_size()
    return size


def _chunk_data(data: dict, num_chunks: int) -> List[dict]:
    """Chunk the data into mini-batches."""
    chunked_data = {}
    min_length = num_chunks
    for k in data:
        if isinstance(data[k], torch.Tensor):
            chunked_data[k] = torch.chunk(data[k], num_chunks)
            min_length = min(num_chunks, len(chunked_data[k]))
        elif isinstance(data[k], list):
            chunked_list = []
            steps = len(data[k]) // num_chunks
            if len(data[k]) - steps * num_chunks > 0:
                steps += 1
            for j in range(0, len(data[k]), steps):
                if j + steps <= len(data[k]):
                    chunked_list.append(data[k][j:j + steps])
                else:
                    chunked_list.append(data[k][j:])
            chunked_data[k] = chunked_list
            min_length = min(num_chunks, len(chunked_data[k]))
        else:
            chunked_data[k] = [data[k]] * num_chunks
    # merged
    merged_data = []
    for i in range(min_length):
        merged_data.append({k: v[i] for k, v in chunked_data.items()})
    return merged_data


class _MMPipelineParallelFlag:
    """The flag for communication."""

    def __init__(self, part_id: int):
        self.part_id = part_id

    def __str__(self) -> str:
        return f'Part {self.part_id}'

    def __repr__(self) -> str:
        return self.__str__()


class _MMPipelineParallelHook:
    """The hook to check the execution device, and send exit signal."""

    def __init__(self,
                 part_id: int,
                 num_parts: int,
                 exec_device: str,
                 is_part_begin: bool = False,
                 out_queues: Dict[str, Queue] = {},
                 events: List[List[Event]] = [],
                 hook_visited_times: Dict[str, int] = {}):
        self.part_id = part_id
        self.num_parts = num_parts
        self.exec_device = exec_device
        self.is_part_begin = is_part_begin
        self.out_queues = out_queues
        self.events = events
        self.hook_visited_times = hook_visited_times

    def __call__(self, module: nn.Module, args: tuple,
                 kwargs: dict) -> Tuple[tuple, dict]:
        if self.is_part_begin:
            # send exit signal
            self._send_signal()
        # check device
        args, kwargs = self._check_device(args, kwargs)
        return args, kwargs

    def _send_signal(self):
        """Send exit signal."""
        chunk_id = int(current_thread().name.split('-')[1])
        visit_times = self.hook_visited_times.get(f'chunk-{chunk_id}', 0)
        if self.part_id != 0 or visit_times != 0:
            prev_part_id = (self.part_id - 1) % self.num_parts
            # send exit signal
            self.out_queues[f'chunk-{chunk_id}'].put(
                (True, _MMPipelineParallelFlag(prev_part_id)))
        # update visited times
        self.hook_visited_times[f'chunk-{chunk_id}'] = visit_times + 1
        # wait
        self.events[chunk_id][self.part_id].wait()

    def _check_device(self, args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
        """Check the device and move the data to the execution device."""
        args = apply_to(args, lambda x: hasattr(x, 'to'),
                        lambda x: x.to(self.exec_device))
        kwargs = apply_to(kwargs, lambda x: hasattr(x, 'to'),
                          lambda x: x.to(self.exec_device))
        return args, kwargs
