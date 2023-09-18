# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import tempfile
from contextlib import contextmanager
from functools import partial
from queue import Queue
from threading import Event, Thread, current_thread
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import psutil
import torch
import torch.nn as nn

from mmengine.analysis import FlopAnalyzer
from mmengine.registry import MODEL_WRAPPERS, MODELS


@MODEL_WRAPPERS.register_module()
class MMPipelineParallel(nn.Module):
    """
    TODO
    """
    in_queues: Dict[str, Queue] = {}
    out_queues: Dict[str, Queue] = {}
    events: List[List[Event]] = []
    stream_contexts: List[torch.cuda.StreamContext] = []
    hook_visited_times: Dict[str, int] = {}

    def __init__(self,
                 model: Union[dict, nn.Module],
                 weights: Optional[str] = None,
                 num_pipelines: Optional[int] = None,
                 num_chunks: Optional[int] = None,
                 memory_threshold: float = 0.7,
                 memory_map: Optional[Dict[str, str]] = None,
                 no_split_module_classes: Optional[List[str]] = None,
                 language_module_classes: Optional[str] = None,
                 device_map: Union[str, Dict[str, dict]] = 'auto',
                 offload_directory: Optional[str] = None,
                 exec_entry: str = '__call__'):

        super().__init__()

        # init model
        self.weights = weights
        self.model_state_dict: Dict[str, torch.Tensor] = {}
        self.model = self._init_model(model)
        self.entry = getattr(self.model, exec_entry)
        # init pipeline parallelism
        if num_pipelines is not None:
            self.num_pipelines = num_pipelines
        else:
            self.num_pipelines = torch.cuda.device_count()
        if num_chunks is not None:
            self.num_chunks = num_chunks
        else:
            self.num_chunks = self.num_pipelines * 32

        # init memory map
        self.memory_threshold = memory_threshold
        self.memory_map = self._init_memory_map(memory_map)

        # init device map
        self.no_split_module_classes = [] if no_split_module_classes is None \
            else no_split_module_classes
        self.language_module_classes = language_module_classes
        self.lm_offset = 0
        self.model_tree = self._get_model_tree()
        if isinstance(device_map, dict):
            self.device_map = device_map
            self.offload_map = self._init_offload_map()
            self.module_map = self._init_module_map()
            self._load_and_dispatch(self.weights)
            self._register_hooks()
            MMPipelineParallel.stream_contexts = self._init_stream_contexts()
            self._inited = True
        else:
            # after we get the input shape, we can init the device map
            self.device_map_policy = device_map
            self._inited = False

        # init offload directory
        if offload_directory is not None:
            self.offload_directory = offload_directory
        else:
            self.offload_directory = tempfile.mkdtemp()
        self.offloaded_weights: Dict[str, Dict[str, Any]] = {}

        # init queues
        MMPipelineParallel.in_queues, MMPipelineParallel.out_queues = \
            self._init_queues()

        # init events
        MMPipelineParallel.events = self._init_events()

    def _init_model(self, model: Union[dict, nn.Module]) -> nn.Module:
        """
        TODO
        """
        if isinstance(model, nn.Module):
            if self.weights is not None:
                return model.to('meta')
            else:
                for n, p in model.named_parameters():
                    self.model_state_dict[n] = p
                for n, b in model.named_buffers():
                    self.model_state_dict[n] = b
                return model.to('meta')
        elif isinstance(model, dict):
            builded_model: nn.Module
            with self._init_empty():
                builded_model = MODELS.build(model)
            return builded_model
        else:
            raise TypeError(f'Unsupported model type {type(model)}')

    @contextmanager
    def _init_empty(self):
        """
        TODO
        """

        def _parameter_hook(module, name, param):
            if self.weights is None:
                self.model_state_dict[name] = param
            param_class = type(param)
            output = param_class(param.to('meta'), **param.__dict__)
            return output

        def _buffer_hook(module, name, buffer):
            if self.weights is None:
                self.model_state_dict[name] = buffer
            output = buffer.to('meta')
            return output

        nn.modules.module.register_module_parameter_registration_hook(
            _parameter_hook)
        nn.modules.module.register_module_buffer_registration_hook(
            _buffer_hook)
        yield

    def _init_memory_map(self, memory_map: Optional[Dict[str, str]]
                         ) -> Dict[str, int]:
        """
        TODO
        """
        new_memory_map = {}
        # cpu
        cpu_memory = psutil.virtual_memory().available
        new_memory_map['cpu'] = cpu_memory
        # gpu
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory
            new_memory_map[f'cuda:{i}'] = gpu_memory
        # check memory map
        if memory_map is not None:
            converted_memory_map = self._convert_memory_map(memory_map)
            for device, memory in converted_memory_map.items():
                if device not in new_memory_map:
                    raise ValueError(f'{device} is not a valid device.' +
                                     'Please use cpu or cuda:i')
                if memory > new_memory_map[device]:
                    raise ValueError(
                        f'The memory of {device} ({memory}) ' +
                        'is larger than the available memory ' +
                        f'({new_memory_map[device]}). Please use a smaller one'
                    )
                new_memory_map[device] = int(memory * self.memory_threshold)
        return new_memory_map

    def _convert_memory_map(self,
                            memory_map: Dict[str, str]) -> Dict[str, int]:
        """
        TODO
        """
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

    def _get_model_tree(self) -> Dict[str, Any]:
        """
        TODO
        """

        def bfs(module: Optional[nn.Module], prefix: str,
                info: Dict[str, Any]):
            # None
            if module is None:
                return
            # self
            info['self'] = module
            info['parameter_size'] = self._parameter_size(module)
            info['flops'] = None
            info['exec_order'] = None
            info['checked'] = False
            # buffer
            if len(module._buffers) != 0:
                info['buffers'] = {}
                for name, buffer in module._buffers.items():
                    curr_name = name if prefix == '' else f'{prefix}.{name}'
                    info['buffers'][curr_name] = buffer
            # submodule
            module_class_name = module.__class__.__name__
            if not (len(module._modules) == 0
                    or module_class_name in self.no_split_module_classes):
                info['submodules'] = {}
                for name, submodule in module._modules.items():
                    curr_name = name if prefix == '' else f'{prefix}.{name}'
                    info['submodules'][curr_name] = {}
                    bfs(submodule, curr_name, info['submodules'][curr_name])

        tree: Dict[str, Any] = {}
        bfs(self.model, '', tree)
        return tree

    def _parameter_size(self, module: nn.Module) -> int:
        """
        TODO
        """
        size = 0
        for _, param in module.named_parameters():
            size += param.nelement() * param.element_size()
        return size

    def _iter_tree(self, module_name: str) -> Optional[Dict[str, Any]]:
        """
        TODO
        """
        tree = self.model_tree
        if module_name == '':
            return tree
        else:
            name_split = module_name.split('.')
            for i in range(len(name_split)):
                curr_name = '.'.join(name_split[:i + 1])
                if i == len(name_split) - 1:
                    # leaf node
                    if 'buffers' in tree:
                        if curr_name in tree['buffers']:
                            # it is buffer
                            return tree['buffers'][curr_name]
                    # it is submodule
                    if 'submodules' not in tree:
                        break
                    else:
                        return tree['submodules'][curr_name]
                else:
                    # due to no_split_module_classes
                    if 'submodules' not in tree:
                        break
                    else:
                        tree = tree['submodules'][curr_name]
        # if not found
        return None

    def _find_tied_weights(self):
        """
        TODO
        """

        def _find_tied_parameters(module: nn.Module,
                                  named_parameters: Dict[str, torch.Tensor],
                                  prefix: str = '',
                                  result: Dict[str, List[str]] = {}):
            if named_parameters is None:
                named_parameters = {n: p for n, p in module.named_parameters()}
            else:
                for name, param in module.named_parameters():
                    full_name = name if prefix == '' else f'{prefix}.{name}'
                    if full_name not in named_parameters:
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
        _find_tied_parameters(self.model, None, '', result)
        # merge into model tree
        self.model_tree['tied_parameters'] = result

    def _get_meta_data(self, data_sample: Tuple[tuple, dict]):
        """
        TODO
        """
        args, kwargs = data_sample
        args_meta = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                args_meta.append(arg.to('meta'))
            else:
                args_meta.append(arg)
        kwargs_meta = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs_meta[k] = v.to('meta')
            else:
                kwargs_meta[k] = v
        data_meta = (tuple(args_meta), kwargs_meta)
        return data_meta

    def _get_flops(self, data_sample: Tuple[tuple, dict]):
        """
        TODO
        """
        data_meta = self._get_meta_data(data_sample)
        if data_meta[1] == {}:
            inputs = data_meta[0]
        else:
            inputs = data_meta
        flop_analyzer = FlopAnalyzer(self.model, inputs=inputs)
        flops = flop_analyzer.by_module()
        # merge into model tree
        for name, num_flops in flops.items():
            tree = self._iter_tree(name)
            if tree is None:
                continue
            else:
                tree['flops'] = num_flops

    def _get_exec_order(self, data_sample: Tuple[tuple, dict]):
        """
        TODO
        """
        exec_order = []
        module_name_map = {m: n for n, m in self.model.named_modules()}

        def return_module_name_hook(module: nn.Module, args: tuple):
            module_name = module_name_map[module]
            if module_name not in exec_order:
                exec_order.append(module_name)

        handle = nn.modules.module.register_module_forward_pre_hook(
            return_module_name_hook)

        data_meta = self._get_meta_data(data_sample)
        with torch.no_grad():
            self.entry(*data_meta[0], **data_meta[1])

        handle.remove()
        # merge into model tree
        for order, name in enumerate(exec_order):
            tree = self._iter_tree(name)
            if tree is None:
                continue
            else:
                tree['exec_order'] = order

    def _init_device_map(self, device_map_policy: str) -> Dict[str, dict]:
        """
        TODO
        """
        if device_map_policy == 'auto':
            device_map_policy = 'balanced'
        if device_map_policy == 'balanced':
            return self._init_device_map_balanced()
        else:
            raise ValueError(
                f'Unsupported device map policy {device_map_policy}')

    def _init_device_map_balanced(self) -> Dict[str, dict]:
        """
        TODO
        """
        avg_flops = self.model_tree['flops'] / self.num_pipelines
        modules: List[Dict[str, Any]] = []
        devices = list(self.memory_map.keys())
        cuda_devices = [d for d in devices if d.startswith('cuda')]
        meta_info = {
            'module_pointer': 0,
            'cuda_pointer': 0,
        }

        for _ in range(self.num_pipelines):
            modules.append({
                'modules': [],
                'flops': 0,
                'parameter_size': 0,
                'init': None,
                'exec': None
            })

        def dfs(tree: Dict[str, Any], name: str, meta_info: Dict[str, int]):
            # handle language model
            if tree['self']._get_name() == self.language_module_classes:
                if meta_info['module_pointer'] != 0:
                    meta_info['module_pointer'] += 1
                meta_info['cuda_pointer'] += 1
                meta_info['cuda_pointer'] %= len(cuda_devices)
                self.lm_offset = meta_info['module_pointer']
            is_handled = False
            module_pointer = meta_info['module_pointer']
            cuda_pointer = meta_info['cuda_pointer']
            # handle self
            module_flops = tree['flops']
            curr_flops = modules[module_pointer]['flops']
            module_size = tree['parameter_size']
            curr_size = modules[module_pointer]['parameter_size']
            curr_cuda_memory = self.memory_map[cuda_devices[cuda_pointer]]
            # infer exec
            is_memory_enough = False
            if module_size + curr_size < curr_cuda_memory:
                is_memory_enough = True
                if module_flops + curr_flops < 1.2 * avg_flops or \
                        module_pointer == self.num_pipelines - 1:
                    modules[module_pointer]['modules'].append(name)
                    modules[module_pointer]['flops'] += module_flops
                    modules[module_pointer]['parameter_size'] += module_size
                    is_handled = True
                    if module_flops + curr_flops > avg_flops:
                        cuda_device = cuda_devices[cuda_pointer]
                        modules[module_pointer]['exec'] = cuda_device
                        if module_pointer != self.num_pipelines - 1:
                            module_pointer += 1
                            cuda_pointer += 1
                            cuda_pointer %= len(cuda_devices)
                            meta_info['module_pointer'] = module_pointer
                            meta_info['cuda_pointer'] = cuda_pointer
            # handle buffers
            if 'buffers' in tree and not is_handled:
                for name, _ in tree['buffers'].items():
                    modules[module_pointer]['modules'].append(name)
            # handle submodules
            if 'submodules' in tree and not is_handled:
                for name, submodule in tree['submodules'].items():
                    dfs(submodule, name, meta_info)
            # if it is not handled, but it has no submodules
            if not is_handled and 'submodules' not in tree:
                if is_memory_enough:
                    if module_pointer == self.num_pipelines - 1:
                        modules[module_pointer]['modules'].append(name)
                        modules[module_pointer]['flops'] += module_flops
                        modules[module_pointer][
                            'parameter_size'] += module_size
                        if module_flops + curr_flops > avg_flops:
                            cuda_device = cuda_devices[cuda_pointer]
                            modules[module_pointer]['exec'] = cuda_device
                            if module_pointer != self.num_pipelines - 1:
                                module_pointer += 1
                                cuda_pointer += 1
                                cuda_pointer %= len(cuda_devices)
                                meta_info['module_pointer'] = module_pointer
                                meta_info['cuda_pointer'] = cuda_pointer
                else:
                    if module_pointer == self.num_pipelines - 1:
                        raise RuntimeError(
                            'The model is too large to fit into' +
                            f'{cuda_devices[cuda_pointer]}' +
                            'Please use more GPUs.')
                if module_pointer != self.num_pipelines - 1:
                    cuda_device = cuda_devices[cuda_pointer]
                    modules[module_pointer]['exec'] = cuda_device
                    module_pointer += 1
                    cuda_pointer += 1
                    cuda_pointer %= len(cuda_devices)
                    meta_info['module_pointer'] = module_pointer
                    meta_info['cuda_pointer'] = cuda_pointer
                    modules[module_pointer]['modules'].append(name)
                    modules[module_pointer]['flops'] += module_flops
                    modules[module_pointer]['parameter_size'] += module_size

        dfs(self.model_tree, name='', meta_info=meta_info)
        # handle last
        if modules[-1]['exec'] is None:
            modules[-1]['exec'] = cuda_devices[meta_info['cuda_pointer']]
        # infer init
        cpu_used = 0
        for i in range(self.num_pipelines):
            if i < len(cuda_devices):
                modules[i]['init'] = cuda_devices[i]
            else:
                cpu_used += modules[i]['parameter_size']
                if cpu_used > self.memory_map['cpu']:
                    modules[i]['init'] = 'disk'
                else:
                    modules[i]['init'] = 'cpu'

        # format modules
        device_map = {}
        for i in range(self.num_pipelines):
            modules_i = modules[i]
            for module in modules_i['modules']:
                device_map[module] = {
                    'part_id': i,
                    'init_device': modules_i['init'],
                    'exec_device': modules_i['exec']
                }
        # handle tied weights
        tied_weights = self.model_tree['tied_parameters']
        for source, targets in tied_weights.items():
            source_info = device_map[source]
            for target in targets:
                target_info = device_map[target]
                target_info['init_device'] = source_info['init_device']
                target_info['exec_device'] = source_info['exec_device']
        return device_map

    def _init_offload_map(self) -> Dict[int, int]:
        """
        TODO
        """
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
        """
        TODO
        """
        module_map = {}
        for name, info in self.device_map.items():
            tree = self._iter_tree(name)
            if isinstance(tree, dict):
                # it is a submodule
                module = tree['self']
            else:
                # it is a buffer
                module = tree
            module_map[name] = {
                'module': module,
                'curr_device': info['init_device'],
                'part_id': info['part_id'],
            }
        return module_map

    def _init_queues(self) -> Tuple[Dict[str, Queue], Dict[str, Queue]]:
        """
        TODO
        """
        in_queues, out_queues = {}, {}
        # init move queues
        for move in ['out', 'in']:
            in_queue: Queue = Queue()
            out_queue: Queue = Queue()
            thread = Thread(
                target=MMPipelineParallel._worker,
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
            thread = Thread(
                target=MMPipelineParallel._worker,
                args=(in_queue, out_queue),
                name=f'chunk-{i}',
                daemon=True,
            )
            thread.start()

            in_queues[f'chunk-{i}'] = in_queue
            out_queues[f'chunk-{i}'] = out_queue
        return in_queues, out_queues

    def _init_events(self) -> List[List[Event]]:
        """
        TODO
        """
        events = []
        for i in range(self.num_chunks):
            events.append([Event() for _ in range(self.num_pipelines)])
        return events

    def _init_stream_contexts(self) -> List[torch.cuda.StreamContext]:
        """
        TODO
        """
        curr_part_id = -1
        inited_streams = {}
        stream_contexts = []
        for info in self.device_map.values():
            if info['part_id'] != curr_part_id:
                curr_part_id = info['part_id']
                exec_device = info['exec_device']
                if exec_device not in inited_streams:
                    stream = torch.cuda.Stream(exec_device)
                    inited_streams[exec_device] = stream
                else:
                    stream = inited_streams[exec_device]
                stream_context = torch.cuda.stream(stream)
                stream_contexts.append(stream_context)
        return stream_contexts

    def _load_and_dispatch(self, weights: Optional[str] = None):
        """
        TODO
        """
        if weights is not None:
            # load weights
            from mmengine.runner.checkpoint import CheckpointLoader
            ckpt = CheckpointLoader.load_checkpoint(weights)
            if 'state_dict' in ckpt:
                self.model_state_dict = ckpt['state_dict']
            else:
                self.model_state_dict = ckpt
        # dispatch weights
        modules_weights: Dict[str, Dict[str, torch.Tensor]]
        modules_weights = {k: {} for k in self.device_map.keys()}
        for weight_name, param in self.model_state_dict.items():
            name_split = weight_name.split('.')
            is_found = False
            for i in range(len(name_split)):
                curr_name = '.'.join(name_split[:i + 1])
                if curr_name in self.device_map:
                    is_found = True
                    init_device = self.device_map[curr_name]['init_device']
                    if init_device == 'disk':
                        dtype = None
                        if param.dtype == torch.bfloat16:
                            param = param.to(torch.int16)
                            dtype = 'bfloat16'
                        array = param.cpu().numpy()
                        if array.ndim == 0:
                            array = array[None]
                        if not os.path.exists(self.offload_directory):
                            os.makedirs(self.offload_directory, exist_ok=True)
                        self.offloaded_weights[weight_name] = {
                            'dtype': array.dtype if dtype is None else dtype,
                            'shape': list(array.shape),
                        }
                        offload_path = os.path.join(self.offload_directory,
                                                    f'{weight_name}.npy')
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
                        # remove prefix
                        weight_name = weight_name.replace(f'{curr_name}.', '')
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
            self.module_map[module_name]['curr_device'] = init_device
            self._load_state_dict(module, module_weights)
        del self.model_state_dict

    def _register_hooks(self):
        """
        TODO
        """
        curr_part_id = -1
        for name, info in self.device_map.items():
            module = self.module_map[name]['module']
            if info['part_id'] != curr_part_id:
                curr_part_id = info['part_id']
                hook = MMPipelineParallel.Hook(
                    part_id=curr_part_id,
                    num_parts=self.num_pipelines,
                    exec_device=info['exec_device'],
                    is_part_begin=True)
            else:
                hook = MMPipelineParallel.Hook(
                    part_id=curr_part_id,
                    num_parts=self.num_pipelines,
                    exec_device=info['exec_device'],
                    is_part_begin=False)
            module.register_forward_pre_hook(hook, with_kwargs=True)

    @staticmethod
    @torch.no_grad()
    def _worker(in_queue: Queue, out_queue: Queue):
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
        """
        TODO
        """
        chunk_id = 0
        schedules: List[List[int]] = []
        waiting_offload_chunks: Set[int] = set()
        schedule = 'exec'
        first_part_id = -1
        blocked_part_id = -1
        while True:
            if schedule == 'offload':
                schedule = 'onload'
            else:
                if schedule == 'onload':
                    schedule = 'exec'
                    schedules = []
                    chunk_id = 0
                    first_part_id = blocked_part_id
                if chunk_id < num_chunks:
                    schedules.insert(0, [chunk_id, first_part_id])
                for i in range(len(schedules)):
                    curr_part_id = schedules[i][1]
                    next_part_id = (curr_part_id + 1) % self.num_pipelines
                    curr_chunk_id = schedules[i][0]
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
                         state_dict: Dict[str, torch.Tensor]):
        """
        TODO
        """
        for name, param in state_dict.items():
            new_module = module
            name_split = name.split('.')
            for i in name_split[:-1]:
                new_module = getattr(new_module, i)
            param_name = name_split[-1]
            is_buffer = param_name in new_module._buffers
            if is_buffer:
                new_module._buffers[param_name] = param
            else:
                old_tensor = getattr(new_module, param_name)
                param_class = type(old_tensor)
                is_requires_grad = old_tensor.requires_grad
                new_tensor = param_class(param, requires_grad=is_requires_grad)
                new_module._parameters[param_name] = new_tensor

    def _move_part(self, part_id: int, target_device: str):
        """
        TODO
        """
        for module_name, info in self.module_map.items():
            if info['part_id'] != part_id:
                continue
            if info['curr_device'] == target_device:
                continue
            if info['curr_device'] == 'disk':
                param_names = [n for n, _ in info['module'].named_parameters()]
                state_dict = {}
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
                    array: np.memmap = np.memmap(
                        param_path, dtype=dtype, mode='r', shape=shape)
                    if len(param_info['shape']) == 0:
                        array = array[0]
                    weight = torch.as_tensor(array)
                    if param_info['dtype'] == 'bfloat16':
                        weight = weight.to(torch.bfloat16)
                    weight = weight.to(target_device)
                    state_dict[param_name] = weight
                self._load_state_dict(info['module'], state_dict)
            elif target_device == 'disk':
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
                    self.offloaded_weights[full_name] = {
                        'dtype': array.dtype if dtype is None else dtype,
                        'shape': list(array.shape),
                    }
                    offload_path = os.path.join(self.offload_directory,
                                                f'{full_name}.npy')
                    file_array: np.memmap
                    file_array = np.memmap(
                        offload_path,
                        dtype=array.dtype,
                        mode='w+',
                        shape=array.shape)
                    file_array[:] = array[:]
                    file_array.flush()
            else:
                module = info['module']
                module.to(target_device)
                info['curr_device'] = target_device

    class Flag:
        """
        TODO
        """

        def __init__(self, part_id: int):
            self.part_id = part_id

        def __str__(self) -> str:
            return f'Part {self.part_id}'

        def __repr__(self) -> str:
            return self.__str__()

    class Hook:
        """
        TODO
        """

        def __init__(self,
                     part_id: int,
                     num_parts: int,
                     exec_device: str,
                     is_part_begin: bool = False):
            self.part_id = part_id
            self.num_parts = num_parts
            self.exec_device = exec_device
            self.is_part_begin = is_part_begin

        def __call__(self, module: nn.Module, args: tuple,
                     kwargs: dict) -> Tuple[tuple, dict]:
            """
            TODO
            """
            if not self.is_part_begin:
                return args, kwargs
            # exit previous part
            self._exit_prev_part()
            # check device
            args, kwargs = self._check_device(args, kwargs)
            # enter current part
            self._enter_curr_part()
            return args, kwargs

        def _exit_prev_part(self):
            """
            TODO
            """
            chunk_id = int(current_thread().name.split('-')[1])
            visit_times = MMPipelineParallel.hook_visited_times.get(
                f'chunk-{chunk_id}', 0)
            if self.part_id != 0 or visit_times != 0:
                prev_part_id = (self.part_id - 1) % self.num_parts
                # send exit signal
                MMPipelineParallel.out_queues[f'chunk-{chunk_id}'].put(
                    (True, MMPipelineParallel.Flag(prev_part_id)))
                # exit stream context
                MMPipelineParallel.stream_contexts[prev_part_id].__exit__(
                    None, None, None)
            # update visited times
            MMPipelineParallel.hook_visited_times[
                f'chunk-{chunk_id}'] = visit_times + 1
            # wait
            MMPipelineParallel.events[chunk_id][self.part_id].wait()

        def _check_device(self, args: tuple,
                          kwargs: dict) -> Tuple[tuple, dict]:
            """
            TODO
            """
            # args
            list_args = list(args)
            for i in range(len(list_args)):
                if isinstance(list_args[i], torch.Tensor):
                    list_args[i] = list_args[i].to(self.exec_device)
            args = tuple(list_args)
            # kwargs
            for k in kwargs:
                if isinstance(kwargs[k], torch.Tensor):
                    kwargs[k] = kwargs[k].to(self.exec_device)
            return args, kwargs

        def _enter_curr_part(self):
            """
            TODO
            """
            MMPipelineParallel.stream_contexts[self.part_id].__enter__()

    def _chunk_data(self, args: tuple,
                    kwargs: dict) -> List[Tuple[tuple, dict]]:
        """
        TODO
        """
        # args
        chunked_args = []
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor):
                chunked_args.append(torch.chunk(args[i], self.num_chunks))
            else:
                chunked_args.append([args[i]] * self.num_chunks)
        # kwargs
        chunked_kwargs = {}
        for k in kwargs:
            if isinstance(kwargs[k], torch.Tensor):
                chunked_kwargs[k] = torch.chunk(kwargs[k], self.num_chunks)
            else:
                chunked_kwargs[k] = [kwargs[k]] * self.num_chunks
        # merge
        lengths = [len(arg) for arg in chunked_args] + \
            [len(v) for v in chunked_kwargs.values()]
        real_num_chunks = min(lengths)
        chunked_data = []

        for i in range(real_num_chunks):
            chunked_data.append((tuple([arg[i] for arg in chunked_args]),
                                 {k: v[i]
                                  for k, v in chunked_kwargs.items()}))
        return chunked_data

    def _merge_results(self, results: List[Any]) -> Any:
        """
        TODO
        """
        # we suppose that the items in results are the same type
        item = results[0]
        if isinstance(item, torch.Tensor):
            result_tensor = torch.cat(results, dim=0)
            return result_tensor
        elif isinstance(item, dict):
            result_dict: Dict[Any, List[Any]] = {k: [] for k in item}
            for item in results:
                for k in item:
                    result_dict[k].append(item[k])
            for k in result_dict:
                result_dict[k] = self._merge_results(result_dict[k])
            return result_dict
        elif isinstance(item, list):
            result_list: List[List[Any]] = [[] for _ in range(len(item))]
            for item in results:
                for i in range(len(item)):
                    result_list[i].append(item[i])
            for i in range(len(result_list)):
                result_list[i] = self._merge_results(result_list[i])
            result_list
        else:
            raise TypeError(f'Unsupported type {type(item)}')

    def forward(self, *args, **kwargs) -> Any:
        """
        TODO
        """
        exec_info = None
        chunked_data = self._chunk_data(args, kwargs)
        results = [None for _ in range(len(chunked_data))]
        # get flops, init device map, offload map, module map and exec order
        if not self._inited:
            self._get_flops(chunked_data[0])
            self._get_exec_order(chunked_data[0])
            self._find_tied_weights()
            self.device_map = self._init_device_map(self.device_map_policy)
            self.offload_map = self._init_offload_map()
            self.module_map = self._init_module_map()
            self._load_and_dispatch(self.weights)
            self._register_hooks()
            MMPipelineParallel.stream_contexts = self._init_stream_contexts()
            self._inited = True

        num_chunks = min(len(chunked_data), self.num_chunks)
        # record finished chunks
        finished_chunks: Set[int] = set()
        # clear visited times
        MMPipelineParallel.hook_visited_times = {}
        # main loop
        offloaded_part_id, onloaded_part_id = -1, -1
        for schedules in self._clock(num_chunks):
            if len(finished_chunks) == num_chunks:
                break
            if isinstance(schedules, str):
                if schedules == 'offload':
                    for i in range(len(self.offload_map)):
                        if self.offload_map[i] == 0 and \
                                offloaded_part_id == -1:
                            offloaded_part_id = i
                        if self.offload_map[i] == 1 and \
                                onloaded_part_id == -1:
                            onloaded_part_id = i
                    if offloaded_part_id != -1:
                        target_device = None
                        for info in self.device_map.values():
                            if info['part_id'] == offloaded_part_id:
                                target_device = info['exec_device']
                                break
                        # offload the first onloaded part
                        task = partial(self._move_part, onloaded_part_id,
                                       target_device)
                        MMPipelineParallel.in_queues['move-out'].put(task)
                        self.offload_map[onloaded_part_id] = 0
                elif schedules == 'onload':
                    target_device = None
                    for info in self.device_map.values():
                        if info['part_id'] == onloaded_part_id:
                            target_device = info['init_device']
                            break
                    # onload the first offloaded part
                    task = partial(self._move_part, offloaded_part_id,
                                   target_device)
                    MMPipelineParallel.in_queues['move-in'].put(task)
                    self.offload_map[offloaded_part_id] = 1
                    offloaded_part_id, onloaded_part_id = -1, -1
            else:
                # send data
                for chunk_id, part_id in schedules:
                    if chunk_id in finished_chunks:
                        continue
                    else:
                        curr_part_id = part_id % self.num_pipelines
                        next_part_id = (part_id + 1) % self.num_pipelines
                        if part_id >= self.num_pipelines:
                            curr_part_id += self.lm_offset
                            next_part_id += self.lm_offset
                        # lock the next event
                        MMPipelineParallel.events[chunk_id][
                            next_part_id].clear()
                        # unlock the current event
                        MMPipelineParallel.events[chunk_id][curr_part_id].set()
                        # new task
                        if part_id == 0:
                            curr_data = chunked_data[chunk_id]
                            task = partial(self.entry, *curr_data[0],
                                           **curr_data[1])
                            MMPipelineParallel.in_queues[
                                f'chunk-{chunk_id}'].put(task)
            if isinstance(schedules, str):
                if schedules == 'offload':
                    success, result = MMPipelineParallel.out_queues[
                        'move-out'].get()
                    if exec_info is not None:
                        continue
                    elif not success:
                        exec_info = result
                        continue
                elif schedules == 'onload':
                    success, result = MMPipelineParallel.out_queues[
                        'move-in'].get()
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
                        success, result = MMPipelineParallel.out_queues[
                            f'chunk-{chunk_id}'].get()
                        if exec_info is not None:
                            continue
                        elif not success:
                            exec_info = result
                            continue
                        if not isinstance(result, MMPipelineParallel.Flag):
                            results[chunk_id] = result
                            finished_chunks.add(chunk_id)
            if exec_info is not None:
                raise exec_info[0].with_traceback(exec_info[1], exec_info[2])
        # merge results
        merged_results = self._merge_results(results)
        return merged_results
