import copy
from typing import Optional, Union, Tuple, Sequence, Dict
from collections import OrderedDict
import datetime
import os.path as osp

import torch
import torch.distributed as dist

from mmengine.data import BaseDataSample
from mmengine.fileio import FileClient
from mmengine.utils import is_tuple_of, scandir
from .hook import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class LoggerHook(Hook):
    def __init__(self,
                 interval: int = 10,
                 custom_keys: Optional[dict] = None,
                 ignore_last: bool = True,
                 interval_exp_name: int = 1000,
                 by_epoch: bool = True,
                 out_dir: Optional[str] = None,
                 out_suffix: Optional[Union[Tuple[str], str]] =
                 ('.log.json', '.log', '.py'),
                 keep_local=True,
                 file_client_args=None,
                 ):
        self.interval = interval
        self.custom_keys = custom_keys
        self.composed_writers = None
        self.ignore_last = ignore_last
        self.by_epoch = by_epoch
        self.time_sec_tot = 0
        self.interval_exp_name = interval_exp_name

        if out_dir is None and file_client_args is not None:
            raise ValueError(
                'file_client_args should be "None" when `out_dir` is not'
                'specified.')
        self.out_dir = out_dir

        if not (out_dir is None or isinstance(out_dir, str)
                or is_tuple_of(out_dir, str)):
            raise TypeError('out_dir should be  "None" or string or tuple of '
                            'string, but got {out_dir}')
        self.out_suffix = out_suffix

        self.keep_local = keep_local
        self.file_client_args = file_client_args
        if self.out_dir is not None:
            self.file_client = FileClient.infer_client(file_client_args,
                                                       self.out_dir)

    def before_run(self, runner: object) -> None:
        if self.out_dir is not None:
            self.file_client = FileClient.infer_client(self.file_client_args,
                                                       self.out_dir)
            # The final `self.out_dir` is the concatenation of `self.out_dir`
            # and the last level directory of `runner.work_dir`
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_client.join_path(self.out_dir, basename)
            runner.logger.info(
                (f'Text logs will be saved to {self.out_dir} by '
                 f'{self.file_client.name} after the training process.'))

        self.start_iter = runner.iter
        if runner.meta is not None:
            self.composed_writers.add_text(runner.meta)

    def after_train_iter(
            self,
            runner: object,
            data_batch: Optional[Sequence[BaseDataSample]] = None,
            outputs: Optional[Sequence[BaseDataSample]] = None) -> None:
        self.log_train(runner)

    def after_val_epoch(self, runner: object) -> None:
        self.log_val(runner)

    def log_train(self, runner: object):
        tag = self._collect_info(runner)
        # `log_tag` will pop some keys and fill `log_str`.
        log_tag = copy.deepcopy(tag)
        if runner.meta is not None and 'exp_name' in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                    self.by_epoch and self.end_of_epoch(runner)):
                exp_info = f'Exp name: {runner.meta["exp_name"]}'
                runner.logger.info(exp_info)

        lr_str = []
        for key, value in tag.items():
            if key.startswith('lr'):
                log_tag.pop(key)
                lr_str.append(f'{key}: {value:.3e}')
        lr_str = ' '.join(lr_str)

        # by epoch: Epoch [4][100/1000]
        # by iter:  Iter [100/100000]
        if self.by_epoch:
            log_str = f'Epoch [{runner.epoch + 1}]' \
                      f'[{runner.inner_iter + 1}/{len(runner.data_loader)}]\t'
        else:
            log_str = f'Iter [{runner.global_iter + 1}/{runner.max_iters}]\t'
        log_str += f'{lr_str}, '
        # Calculate eta time
        self.time_sec_tot += (tag['time'] * self.interval)
        time_sec_avg = self.time_sec_tot / (runner.iter - self.start_iter + 1)
        eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        log_str += f'eta: {eta_str}, '
        log_str += f'time: {tag["time"]:.3f}, ' \
                   f'data_time: {tag["data_time"]:.3f}, '
        # pop recorded keys
        log_tag.pop('time')
        log_tag.pop('data_time')
        # statistic memory
        if torch.cuda.is_available():
            log_str += f'memory: {self._get_max_memory(runner)}, '

        log_items = []
        for name, val in log_tag.items():
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        log_str += ', '.join(log_items)
        runner.logger.info(log_str)
        # Write tag.
        for name, val in tag.items():
            runner.composed_writers.add_scalar(name, val, runner.global_iter+1)

    def log_val(self, runner: object):
        tag = self._collect_info(runner)
        # val/test time
        # here 1000 is the length of the val dataloader
        # by epoch: Epoch[val] [4][1000]
        # by iter: Iter[val] [1000]
        if self.by_epoch:
            log_str = f'Epoch(val) [{runner.epoch}][{runner.inner_iter+1}]\t'
        else:
            log_str = f'Iter(val) [{runner.inner_iter+1}]\t'
        runner.logger.info(log_str)
        # Write tag.
        for name, val in tag.items():
            runner.composed_writers.add_scalar(name, val, runner.global_iter+1)

    def _get_window_size(self, runner, window_size):
        if isinstance(window_size, int):
            assert window_size == self.interval
            return window_size
        else:
            if window_size == 'epoch':
                return runner.inner_iter + 1
            elif window_size == 'global':
                return runner.iter
            elif window_size == 'current':
                return 1

    def _collect_info(self, runner: object):
        tag = OrderedDict()
        log_buffers = runner.message_hub.log_buffers
        # Ensure all metric and lr values are latest.
        for key in log_buffers:
            tag[key] = log_buffers[key].current()
        # Update the latest learning rate and smoothed time logs.
        assert 'time' in log_buffers, 'Runner must contain IterTimerHook.'
        tag['time'] = log_buffers['time'].mean(self.interval)
        tag['data_time'] = log_buffers['data_time'].mean(self.interval)
        # tag loss based on interval smoothing.
        for key in log_buffers:
            if key.startswith('loss'):
                tag[key] = log_buffers[key].mean(self.interval)
        # Update custom keys.
        self._parse_custom_keys(runner, log_buffers, tag,
                                self.custom_keys)
        return tag

    def _parse_custom_keys(self,
                          runner: object,
                          log_buffers: OrderedDict,
                          tag: OrderedDict,
                          cfg_dicts: Optional[OrderedDict]):
        if not isinstance(cfg_dicts, dict):
            return
        for key, value in cfg_dicts.items():
            if isinstance(value, list):
                for cfg_dict in value:
                    self._statistics_single_key(runner, key, cfg_dict,
                                                log_buffers, tag)
            if isinstance(value, dict):
                self._statistics_single_key(runner, key, value, log_buffers,
                                        tag)

    def _statistics_single_key(self, runner, key, cfg_dict, log_buffers, tag):
        if key in ['data_time']:
            assert 'log_name' in cfg_dict, 'time and data_time cannot be ' \
                                           'overwritten by custom keys!'
        if 'window_size' in cfg_dict:
            cfg_dict['window_size'] = self.get_smooth_window_size(
                                                    runner,
                                                    cfg_dict['window_size'])
        if 'log_name' in cfg_dict:
            name = cfg_dict.pop('log_name')
        else:
            name = key
        tag[name] = log_buffers[key].statistics(**cfg_dict)

    def _get_max_memory(self, runner: object):
        device = getattr(runner.model, 'output_device', None)
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                              dtype=torch.int,
                              device=device)
        if runner.world_size > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

