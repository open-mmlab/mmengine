# Copyright (c) OpenMMLab. All rights reserved.
import sys
from contextlib import contextmanager
from functools import partial
from queue import Queue
from threading import Event, Thread, current_thread
from typing import Dict, List, Iterable, Union, Optional, Tuple

import psutil
import torch
import torch.nn as nn

from mmengine.analysis import get_model_complexity_info
from mmengine.config import Config, ConfigDict
from mmengine.registry import MODEL_WRAPPERS

ModelType = Union[Config, ConfigDict, dict, nn.Module, str]


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
                 model: Optional[ModelType] = None,
                 weights: Optional[str] = None,
                 num_pipelines: Optional[int] = None,
                 num_chunks: Optional[int] = None,
                 memory_threshold: float = 0.7,
                 memory_map: Optional[Dict[str, int]] = None,
                 no_split_module_classes: Optional[List[str]] = None,
                 language_module_classes: Optional[str] = None,
                 device_map: Union[str, Dict[str, str]] = 'auto',
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
        if isinstance(device_map, dict):
            self.device_map = self._check_device_map(device_map)
        else:
            self.device_map = self._init_device_map(device_map)
        self.offload_map = self._init_offload_map()
        self.module_map = self._init_module_map()

        # init offload directory
        self.offload_directory = offload_directory

        # init queues
        MMPipelineParallel.in_queues, MMPipelineParallel.out_queues = \
            self._init_queues()

        # init events
        MMPipelineParallel.events = self._init_events()

        # init stream contexts
        MMPipelineParallel.stream_contexts = self._init_stream_contexts()

        # load weights, register hooks and dispatch model
        self._load_and_dispatch(weights)
        self._register_hooks()

    def _init_model(self, model: ModelType) -> nn.Module:
        """
        TODO
        """
        pass

    @contextmanager
    def _init_empty(self):
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

    def _check_device_map(self, device_map: Dict[str, str]) -> Dict[str, str]:
        """
        TODO
        """
        pass

    def _init_device_map(self, device_map: str) -> Dict[str, str]:
        """
        TODO
        """
        pass

    def _init_offload_map(self) -> Dict[str, str]:
        """
        TODO
        """
        pass

    def _init_module_map(self) -> Dict[str, str]:
        """
        TODO
        """
        pass

    def _init_queues(self) -> Dict[str, Queue]:
        """
        TODO
        """
        in_queues, out_queues = {}, {}
        # init move queues
        for move in ['in', 'out']:
            in_queue = Queue()
            out_queue = Queue()
            thread = Thread(
                target=MMPipelineParallel._worker,
                args=(in_queue, out_queue),
                name=f'move-{move}',
                daemon=True
            )
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
            events.append(
                [Event() for _ in range(self.num_pipelines)]
            )
        return events

    def _init_stream_contexts(self) -> List[torch.cuda.StreamContext]:
        """
        TODO
        """
        pass

    def _load_and_dispatch(self, weights: Optional[str] = None):
        """
        TODO
        """
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

    def _clock(self, num_chunks: int):
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
                     kwargs: dict):
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

    def _chunk_data(self, data: Iterable) -> List[Iterable]:
        """
        TODO
        """
        pass

    def forward(self, data):
        """
        TODO
        """
        pass
