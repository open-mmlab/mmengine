# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Callable, Optional, Union

import torch

from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


def check_kineto() -> bool:  # noqa
    kineto_exist = False
    try:
        if torch.autograd.kineto_available():
            kineto_exist = True
    except AttributeError:
        warnings.warn('NO KINETO')
    return kineto_exist


@HOOKS.register_module()
class ProfilerHook(Hook):
    """A hook to analyze performance during training and inference.

    PyTorch Profiler is a tool that allows the collection of the performance
    metrics during the training. More details on Profiler can be found at
    `official docs <https://pytorch.org/docs/stable/profiler.html
    #torch.profiler.profile>`_

    Args:
        by_epoch (bool): Profile performance by epoch or by iteration.
            Defaults to True.
        profile_times (int): The period (epoch/iter) recorded by the profiler.
            Defaults to 1. For example, profile_iters=10 and by_epoch=False,
            indicate that 0-10 iterations are recorded.
        activity_with_cpu (bool): Activities to be used in the analysis (CPU)
        activity_with_cuda (bool): Activities to be used in the analysis (CUDA)
        schedule (dict, optional): Key-word arguments passed to
            `torch.profile.schedule <https://pytorch.org/docs/stable/
            profiler.html#torch.profiler.schedule>`_.
            Defaults to None, which means profiling without a schedule
        on_trace_ready (callable, dict, optional): Either a handler or a dict
            of generating handler. Defaults to None, which means profiling
            without an on_trace_ready.The Callable type needs to construct its
            own function that can handle 'torch.autograd.profiler.profile'.
            Two officially recommended ways are provided, namely terminal
            display or tensorboard display. The terminal display content can be
            adjusted through 'EventList.table()'
            from 'torch.autograd.profiler_util.py'.
            If using tensorboard, save to '{work_dir}/tf_tracing_logs'
            by default.
        record_shapes (bool): Save information about operator's input shapes.
            Defaults to False.
        profile_memory (bool): Track tensor memory allocation/deallocation.
            Defaults to False.
        with_stack (bool): Record source information (file and line number)
            for the ops. Defaults to False.
        with_flops (bool): Use formula to estimate the FLOPS of specific
            operators (matrix multiplication and 2D convolution).
            Defaults to False.
        json_trace_path (str, optional): Exports the collected trace in Chrome
            JSON format. Chrome use 'chrome://tracing' view json file.
            Defaults to None, which means profiling does not store json files.

    Examples:
        >>> # tensorboard trace
        >>> trace_config = dict(type='tb_trace')
        >>> profiler_hook_cfg = dict(on_trace_ready=trace_config)
    """
    priority = 'VERY_LOW'

    def __init__(self,
                 *,
                 by_epoch: bool = True,
                 profile_times: int = 1,
                 activity_with_cpu: bool = True,
                 activity_with_cuda: bool = False,
                 schedule: Optional[dict] = None,
                 on_trace_ready: Union[Callable, dict, None] = None,
                 record_shapes: bool = False,
                 profile_memory: bool = False,
                 with_stack: bool = False,
                 with_flops: bool = False,
                 json_trace_path: Optional[str] = None) -> None:

        try:
            from torch import profiler
        except ImportError:
            raise ImportError('please upgrade torch above 1.8.1')
        if not check_kineto():
            raise ImportError('Due to Kineto support issues, please upgrade '
                              'pytorch above 1.8.1(windows users above 1.9.1)')

        assert isinstance(by_epoch, bool), '``by_epoch`` should be a boolean.'
        self.by_epoch = by_epoch

        if profile_times < 1:
            raise ValueError('profile_iters should be greater than 0, '
                             f'but got {profile_times}')
        if by_epoch and profile_times > 1:
            raise ValueError(
                f'Profiler will profile 0-{profile_times} epochs.\n'
                'Since profiler will slow down the training, it is recommended'
                ' to train 1 epoch with ProfilerHook and adjust your setting '
                'according to the profiler summary.\n'
                'During normal training(epoch > 1), '
                'you may disable the ProfilerHook.')
        self.profile_times = profile_times

        assert isinstance(activity_with_cpu, bool), \
            '``activity_with_cpu`` should be a boolean.'
        assert isinstance(activity_with_cuda, bool), \
            '``activity_with_cuda`` should be a boolean.'
        self.activities = []
        if activity_with_cpu:
            self.activities.append(profiler.ProfilerActivity.CPU)
        if activity_with_cuda:
            self.activities.append(profiler.ProfilerActivity.CUDA)

        if schedule is not None:
            assert isinstance(schedule, dict), '``schedule`` should be a dict.'
            self.schedule = profiler.schedule(**schedule)
        else:
            self.schedule = None

        self.on_trace_ready = on_trace_ready
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops

        self.json_trace_path = json_trace_path

    @master_only
    def before_run(self, runner):
        """Initialize the profiler.

        Through the runner parameter, the validity of the parameter is further
        determined.
        """
        max_times = runner.max_epochs if self.by_epoch else runner.max_iters
        if max_times < self.profile_times:
            raise ValueError(
                f'``profile_times`` should not be greater than {max_times}')

        on_trace_ready = self._parse_trace_config(runner)

        self.profiler = torch.profiler.profile(  # noqa
            activities=self.activities,
            schedule=self.schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops)

        self.profiler.__enter__()
        runner.logger.info('profiler is profiling...')

    def _parse_trace_config(self, runner):
        """Used to parse the parameter 'on_trace_ready'."""
        if self.on_trace_ready is None:
            _on_trace_ready = None
        elif callable(self.on_trace_ready):
            _on_trace_ready = self.on_trace_ready
        elif isinstance(self.on_trace_ready, dict):
            trace_cfg = self.on_trace_ready.copy()
            trace_type = trace_cfg.pop('type')

            # Build a log printing handle
            if trace_type == 'log_trace':

                def _log_handler(_profile):
                    print(_profile.key_averages().table(**trace_cfg))

                _on_trace_ready = _log_handler

            elif trace_type == 'tb_trace':  # tensorboard_trace handler
                try:
                    import torch_tb_profiler  # noqa: F401
                except ImportError:
                    raise ImportError(
                        'please run ``pip install torch-tb-profiler``')

                if 'dir_name' not in trace_cfg:
                    trace_cfg['dir_name'] = osp.join(runner.log_dir,
                                                     'tf_tracing_logs')
                elif not osp.isabs(trace_cfg['dir_name']):
                    trace_cfg['dir_name'] = osp.join(runner.log_dir,
                                                     trace_cfg['dir_name'])
                runner.logger.info('trace_files of ProfilerHook will be '
                                   f'saved to {trace_cfg["dir_name"]}.')

                if self.json_trace_path is not None:
                    runner.logger.warn(
                        'When using tensorboard_trace, it is recommended to '
                        'save json files by setting ``worker_name`` instead of'
                        ' setting ``json_trace_path``')
                _on_trace_ready = torch.profiler.tensorboard_trace_handler(
                    **trace_cfg)
            else:
                raise ValueError('trace_type should be "log_trace" or '
                                 f'"tb_trace", but got {trace_type}')
        else:
            raise ValueError(
                '``on_trace_ready`` should be a handler, or dict, or None, '
                f'but got {self.on_trace_ready}')
        return _on_trace_ready

    @master_only
    def after_train_epoch(self, runner):
        """Determine if the content is exported."""
        if self.by_epoch and runner.epoch == self.profile_times - 1:
            self._export_chrome_trace(runner)

    @master_only
    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """Update the content according to the schedule, and determine if the
        content is exported."""
        if self.schedule is None:
            self.profiler.step()
        if not self.by_epoch and runner.iter == self.profile_times - 1:
            self._export_chrome_trace(runner)

    def _export_chrome_trace(self, runner):
        """Exporting content."""
        runner.logger.info('profiler may take a few minutes...')
        self.profiler.__exit__(None, None, None)
        if self.json_trace_path is not None:
            self.profiler.export_chrome_trace(self.json_trace_path)
