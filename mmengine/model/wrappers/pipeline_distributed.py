# Copyright (c) OpenMMLab. All rights reserved.
import sys
from contextlib import contextmanager
from functools import partial
from queue import Queue
from threading import Event, Thread, current_thread
from typing import Any, Dict, List, Union, Optional, Tuple

import psutil
import torch
import torch.nn as nn

from mmengine.analysis import FlopAnalyzer
from mmengine.config import Config, ConfigDict
from mmengine.registry import MODELS, MODEL_WRAPPERS

ModelType = Union[Config, ConfigDict, dict, nn.Module]


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
                 model: ModelType,
                 weights: Optional[str] = None,
                 num_pipelines: Optional[int] = None,
                 num_chunks: Optional[int] = None,
                 memory_threshold: float = 0.7,
                 memory_map: Optional[Dict[str, int]] = None,
                 no_split_module_classes: Optional[List[str]] = None,
                 language_module_classes: Optional[str] = None,
                 device_map: Union[str, Dict[str, dict]] = 'auto',
                 offload_directory: Optional[str] = None,
                 exec_entry: str = 'forward'):

        super().__init__()

        # init model
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
        self.model_tree = self._get_model_tree()
        if isinstance(device_map, dict):
            self.device_map = device_map
            self.offload_map = self._init_offload_map()
            self.module_map = self._init_module_map()
            self._inited = True
        else:
            # after we get the input shape, we can init the device map
            self.device_map_policy = device_map
            self.offload_map = None
            self.module_map = None
            self._inited = False

        # init offload directory
        self.offload_directory = offload_directory

        # init queues
        MMPipelineParallel.in_queues, MMPipelineParallel.out_queues = \
            self._init_queues()

        # init events
        MMPipelineParallel.events = self._init_events()

        # init stream contexts
        MMPipelineParallel.stream_contexts = self._init_stream_contexts()

    def _init_model(self, model: ModelType) -> nn.Module:
        """
        TODO
        """
        if isinstance(model, nn.Module):
            return model.to('meta')
        elif isinstance(model, dict):
            cfg = ConfigDict(model)
        elif isinstance(model, (Config, ConfigDict)):
            cfg = model
        else:
            raise TypeError(
                f'Unsupported model type {type(model)}'
            )
        with MMPipelineParallel._init_empty():
            model = MODELS.build(cfg)
        return model

    @contextmanager
    @staticmethod
    def _init_empty():
        """
        TODO
        """
        def _parameter_hook(module, name, param):
            param_class = type(param)
            output = param_class(param.to('meta'), **param.__dict__)
            return output

        def _buffer_hook(module, name, buffer):
            output = buffer.to('meta')
            return output

        nn.modules.module.register_module_parameter_registration_hook(
            _parameter_hook
        )
        nn.modules.module.register_module_buffer_registration_hook(
            _buffer_hook
        )
        yield

    def _init_memory_map(self,
                         memory_map: Optional[Dict[str, str]]
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
            memory_map = self._convert_memory_map(memory_map)
            for device, memory in memory_map.items():
                if device not in new_memory_map:
                    raise ValueError(
                        f'{device} is not a valid device.' +
                        'Please use cpu or cuda:i'
                    )
                if memory > new_memory_map[device]:
                    raise ValueError(
                        f'The memory of {device} ({memory}) ' +
                        'is larger than the available memory ' +
                        f'({new_memory_map[device]}). Please use a smaller one'
                    )
                new_memory_map[device] = memory
        return new_memory_map

    def _convert_memory_map(self,
                            memory_map: Dict[str, str]
                            ) -> Dict[str, int]:
        """
        TODO
        """
        for device, memory in memory_map.items():
            if memory.upper().endswith('GIB'):
                memory_map[device] = int(memory[:-3]) * (1024 ** 3)
            elif memory.upper().endswith('MIB'):
                memory_map[device] = int(memory[:-3]) * (1024 ** 2)
            elif memory.upper().endswith('KIB'):
                memory_map[device] = int(memory[:-3]) * 1024
            elif memory.upper().endswith('GB'):
                size = int(memory[:-2]) * (10 ** 9)
                if memory.endswith('b'):
                    size = size // 8
                memory_map[device] = size
            elif memory.upper().endswith('MB'):
                size = int(memory[:-2]) * (10 ** 6)
                if memory.endswith('b'):
                    size = size // 8
                memory_map[device] = size
            elif memory.upper().endswith('KB'):
                size = int(memory[:-2]) * (10 ** 3)
                if memory.endswith('b'):
                    size = size // 8
                memory_map[device] = size
            else:
                raise ValueError(
                    f'{memory} is not in a valid format.' +
                    'Please use GiB, MiB, KiB, GB, MB or KB, e.g. 6GB'
                )
        return memory_map

    def _get_model_tree(self) -> Dict[str, Any]:
        """
        TODO
        """
        def bfs(module: nn.Module,
                prefix: str,
                info: Dict[str, Any]):
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
            if not (len(module._modules) == 0 or
                    module_class_name in self.no_split_module_classes):
                info['submodules'] = {}
                for name, submodule in module._modules.items():
                    curr_name = name if prefix == '' else f'{prefix}.{name}'
                    info['submodules'][curr_name] = {}
                    bfs(submodule, curr_name, info['submodules'][curr_name])
        tree = {}
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

    def _find_tied_weights(self) -> List[List[str]]:
        """
        TODO
        """
        pass

    def _get_meta_data(self, data_sample: Tuple(tuple, dict)):
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

    def _get_flops(self, data_sample: Tuple(tuple, dict)):
        """
        TODO
        """
        data_meta = self._get_meta_data(data_sample)
        flop_analyzer = FlopAnalyzer(self.model, inputs=data_meta)
        flops = flop_analyzer.by_module()
        # merge into model tree
        for name, num_flops in flops.items():
            tree = self.model_tree
            if name == '':
                tree['flops'] = num_flops
            else:
                name_split = name.split('.')
                for i in range(len(name_split)):
                    curr_name = '.'.join(name_split[:i + 1])
                    if i == len(name_split) - 1:
                        tree[curr_name]['flops'] = num_flops
                    else:
                        # due to no_split_module_classes
                        if 'submodules' not in tree[curr_name]:
                            continue
                        else:
                            tree = tree[curr_name]['submodules']

    def _get_exec_order(self, data_sample: Tuple(tuple, dict)):
        """
        TODO
        """
        exec_order = []

        def return_module_name_hook(module: nn.Module,
                                    args: tuple,
                                    kwargs: dict):
            exec_order.append(module.__class__.__name__)

        handle = nn.modules.module.register_module_forward_pre_hook(
            return_module_name_hook
        )

        data_meta = self._get_meta_data(data_sample)
        with torch.no_grad():
            self.entry(*data_meta[0], **data_meta[1])

        handle.remove()
        # merge into model tree
        for name, order in enumerate(exec_order):
            tree = self.model_tree
            if name == '':
                tree['exec_order'] = order
            else:
                name_split = name.split('.')
                for i in range(len(name_split)):
                    curr_name = '.'.join(name_split[:i + 1])
                    if i == len(name_split) - 1:
                        tree[curr_name]['exec_order'] = order
                    else:
                        # due to no_split_module_classes
                        if 'submodules' not in tree[curr_name]:
                            continue
                        else:
                            tree = tree[curr_name]['submodules']            

    def _init_device_map(self, device_map: str) -> Dict[str, dict]:
        """
        TODO
        """
        pass

    def _init_offload_map(self) -> Dict[int, int]:
        """
        TODO
        """
        curr_part_id = -1
        offload_map = {}
        for info in self.device_map.values():
            if info['part_id'] != curr_part_id:
                curr_part_id = info['part_id']
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
            name_split = name.split('.')
            tree = self.model_tree
            for i in range(len(name_split)):
                curr_name = '.'.join(name_split[:i + 1])
                if i == len(name_split) - 1:
                    module = tree[curr_name]['self']
                else:
                    tree = tree[curr_name]['submodules']
            module_map[name] = {
                'module': module,
                'curr_device': info['init_device'],
                'part_id': info['part_id'],
            }
        return module_map

    def _init_queues(self) -> Dict[str, Queue]:
        """
        TODO
        """
        in_queues, out_queues = {}, {}
        # init move queues
        for i in range(self.num_pipelines):
            in_queue = Queue()
            out_queue = Queue()
            thread = Thread(
                target=MMPipelineParallel._worker,
                args=(in_queue, out_queue),
                name=f'part-{i}',
                daemon=True
            )
            thread.start()

            in_queues[f'part-{i}'] = in_queue
            out_queues[f'part-{i}'] = out_queue
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
            events.append(
                [Event() for _ in range(self.num_pipelines)]
            )
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
        if weights is None:
            return
        # load weights
        pass

    def _register_hooks(self):
        """
        TODO
        """
        pass

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

    def _clock(self, num_chunks: int) -> List[Union[Tuple[int, int], str]]:
        """
        TODO
        """
        pass

    def _move_part(self, part_id: int, target_device: str):
        """
        TODO
        """
        pass

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

        def __call__(self,
                     module: nn.Module,
                     args: tuple,
                     kwargs: dict) -> Tuple(tuple, dict):
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
                f'chunk-{chunk_id}', 0
            )
            if self.part_id != 0 or visit_times != 0:
                prev_part_id = (self.part_id - 1) % self.num_parts
                # send exit signal
                MMPipelineParallel.out_queues[f'chunk-{chunk_id}'].put(
                    (True, MMPipelineParallel.Flag(prev_part_id))
                )
                # exit stream context
                MMPipelineParallel.stream_contexts[
                    prev_part_id].__exit__(None, None, None)
            # update visited times
            MMPipelineParallel.hook_visited_times[
                f'chunk-{chunk_id}'] = visit_times + 1
            # wait
            MMPipelineParallel.events[chunk_id][self.part_id].wait()

        def _check_device(self,
                          args: tuple,
                          kwargs: dict) -> Tuple(tuple, dict):
            """
            TODO
            """
            # args
            args = list(args)
            for i in range(len(args)):
                if isinstance(args[i], torch.Tensor):
                    args[i] = args[i].to(self.exec_device)
            args = tuple(args)
            # kwargs
            for k in kwargs:
                if isinstance(kwargs[k], torch.Tensor):
                    kwargs[k] = kwargs[k].to(self.exec_device)
            return args, kwargs

        def _enter_curr_part(self):
            """
            TODO
            """
            MMPipelineParallel.stream_contexts[
                self.part_id].__enter__()

    def _chunk_data(self, args: tuple, kwargs: dict) -> List[Tuple[tuple, dict]]:
        """
        TODO
        """
        # args
        chunked_args = [None for _ in range(len(args))]
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor):
                chunked_args[i] = torch.chunk(args[i], self.num_chunks)
            else:
                chunked_args[i] = [args[i]] * self.num_chunks
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
            chunked_data.append(
                (
                    tuple([arg[i] for arg in chunked_args]),
                    {k: v[i] for k, v in chunked_kwargs.items()}
                )
            )
        return chunked_data

    def forward(self, *args, **kwargs):
        """
        TODO
        """
        exec_info = None
        chunked_data = self._chunk_data(args, kwargs)
        # get flops, init device map, offload map, module map and exec order
        if not self._inited:
            self._get_flops(chunked_data[0])
            self._get_exec_order(chunked_data[0])
            self.device_map = self._init_device_map(self.device_map_policy)
            self.offload_map = self._init_offload_map()
            self.module_map = self._init_module_map()
            self._inited = True

        num_chunks = min(len(chunked_data), self.num_chunks)
        # record finished chunks
        finished_chunks = set()
        # clear visited times
        MMPipelineParallel.hook_visited_times = {}
        # main loop
        for schedule in self._clock(num_chunks):
            pass
