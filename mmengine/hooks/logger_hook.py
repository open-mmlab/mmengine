# Copyright (c) OpenMMLab. All rights reserved.
import copy
import datetime
import os
import os.path as osp
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence, Union

import torch

from mmengine.dist import master_only
from mmengine.fileio import FileClient
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.utils import is_tuple_of, scandir

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class LoggerHook(Hook):
    """In this logger hook, the information will be printed on the terminal and
    saved in JSON file, tensorboard, wandb .etc.

    Args:
        by_epoch (bool): Whether ``EpochBasedLoop`` is used.
            Defaults to True.
        interval (int): Logging interval (every k iterations).
            Defaults to 10.
        custom_keys (dict, optional): Defines the keys in the log and which
            kinds of statistic methods should be used to log them.

            - ``custom_keys`` contains multiple string-dict pairs. In each
            string-dict pair, the string defines a key name in the log and the
            dict is a config defines the statistic methods and corresponding
            arguments used to log the value. For example,
            ``dict(loss=dict(method_name='mean', log_name='global_loss',
            window_size='global'))`` which means the log key ``loss`` will be
            counted as global mean and additionally logged as ``global_loss``.
            If ``log_name`` is not defined in config dict, the original logged
            key will be overwritten.
            - The key in ``LoggerHook.fixed_smooth_keys`` cannot be overwritten
            because ``time`` and ``iter_time`` will be used to calculate
            estimated time of arrival. If you want to recount the time, you
            should set ``log_name`` in corresponding values.
            - For those statistic methods with the ``window_size`` argument,
            if ``by_epoch`` is set to False, ``windows_size`` should not be
            `epoch` to statistics log value by epoch.
        ignore_last (bool): Ignore the log of last iterations in each epoch if
            the number of remaining iterations is less than :attr:`interval`.
            Defaults to True.
        interval_exp_name (int): Logging interval for experiment name. This
            feature is to help users conveniently get the experiment
            information from screen or log file. Defaults to 1000.
        out_dir (str or Path, optional): The root directory to save
            checkpoints. If not specified, ``runner.work_dir`` will be used
            by default. If specified, the ``out_dir`` will be the concatenation
             of ``out_dir`` and the last level directory of
            ``runner.work_dir``. For example, if the input ``our_dir`` is
            ``./tmp`` and ``runner.work_dir`` is ``./work_dir/cur_exp``,
            then the log will be saved in ``./tmp/cur_exp``. Deafule to None.
        out_suffix (Tuple[str] or str): Those filenames ending with
            ``out_suffix`` will be copied to ``out_dir``. Defaults to
            ('.log.json', '.log', '.py').
        keep_local (bool): Whether to keep local logs in the local machine
            when :attr:`out_dir` is specified. If False, the local log will be
            removed. Defaults to True.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None.

    Examples:
        >>> # `log_name` is defined, `loss_mean_window` will be an additional
        >>> # record.
        >>> logger_hook_cfg = dict(by_epoch=True,
        >>>                        custom_keys=dict(
        >>>                            loss=dict(
        >>>                                log_name='loss_mean_window',
        >>>                                method_name='mean',
        >>>                                window_size=10)))
        >>> # `log_name` is not defined. `loss` will be overwritten by
        >>> # `global_mean` statistics.
        >>> logger_hook_cfg = dict(by_epoch=True,
        >>>                        custom_keys=dict(
        >>>                            loss=dict(
        >>>                                method_name='mean',
        >>>                                window_size='global')))
        >>> # `time` cannot be overwritten, `global_time` will be an additional
        >>> # record.
        >>> logger_hook_cfg = dict(by_epoch=True,
        >>>                        custom_keys=dict(
        >>>                            time=dict(
        >>>                                 log_name='global_time',
        >>>                                 method='mean',
        >>>                                 window_size='global')))
        >>> # Record loss with different statistics methods.
        >>> logger_hook_cfg = dict(by_epoch=True,
        >>>                        custom_keys=dict(loss=[
        >>>                            dict(log_name='loss_mean_window',
        >>>                                 method_name='mean',
        >>>                                 window_size=10),
        >>>                            dict(method_name='mean',
        >>>                                 window_size='global')]))
    """
    # eta will be calculated by time. `time` and `data_time` should not be
    # overwritten.
    fixed_smooth_keys = ('time', 'data_time')
    priority = 'BELOW_NORMAL'

    def __init__(
        self,
        by_epoch: bool = True,
        interval: int = 10,
        custom_keys: Optional[dict] = None,
        ignore_last: bool = True,
        interval_exp_name: int = 1000,
        out_dir: Optional[Union[str, Path]] = None,
        out_suffix: Union[Sequence[str], str] = ('.log.json', '.log', '.py'),
        keep_local=True,
        file_client_args=None,
    ):
        self._inner_iter = 0
        self.by_epoch = by_epoch
        self.interval = interval
        self.custom_keys = custom_keys if custom_keys is not None else dict()
        self.ignore_last = ignore_last

        self.time_sec_tot = 0
        self.interval_exp_name = interval_exp_name
        self._check_custom_keys()

        if out_dir is None and file_client_args is not None:
            raise ValueError(
                'file_client_args should be "None" when `out_dir` is not'
                'specified.')
        self.out_dir = out_dir

        if not (out_dir is None or isinstance(out_dir, str)
                or is_tuple_of(out_dir, str)):
            raise TypeError('out_dir should be None or string or tuple of '
                            f'string, but got {type(out_dir)}')
        self.out_suffix = out_suffix

        self.keep_local = keep_local
        self.file_client_args = file_client_args
        if self.out_dir is not None:
            self.file_client = FileClient.infer_client(file_client_args,
                                                       self.out_dir)

    def before_run(self, runner) -> None:
        """Infer ``self.file_client`` from ``self.out_dir``. Initialize the
        ``self.start_iter`` and record the meta information.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.out_dir is not None:
            # The final `self.out_dir` is the concatenation of `self.out_dir`
            # and the last level directory of `runner.work_dir`
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_client.join_path(self.out_dir, basename)
            runner.logger.info(
                (f'Text logs will be saved to {self.out_dir} by '
                 f'{self.file_client.name} after the training process.'))

        self.json_log_path = osp.join(runner.work_dir,
                                      f'{runner.timestamp}.log.json')
        self.yaml_log_path = osp.join(runner.work_dir,
                                      f'{runner.timestamp}.log.json')
        self.start_iter = runner.iter
        if runner.meta is not None:
            runner.writer.add_params(runner.meta, file_path=self.yaml_log_path)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Record training logs.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model.
                Defaults to None.
        """
        self._inner_iter = batch_idx
        if runner.meta is not None and 'exp_name' in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                    self.by_epoch and self.end_of_epoch(
                        runner.train_loop.dataloader, batch_idx)):
                exp_info = f'Exp name: {runner.meta["exp_name"]}'
                runner.logger.info(exp_info)
        if self.by_epoch and self.every_n_inner_iters(batch_idx,
                                                      self.interval):
            self._log_train(runner)
        elif not self.by_epoch and self.every_n_iters(runner, self.interval):
            self._log_train(runner)
        elif self.end_of_epoch(runner.train_loop.dataloader,
                               batch_idx) and not self.ignore_last:
            # `runner.max_iters` may not be divisible by `self.interval`. if
            # `self.ignore_last==True`, the log of remaining iterations will
            # be recorded (Epoch [4][1000/1007], the logs of 998-1007
            # iterations will be recorded).
            self._log_train(runner)

    def after_val_epoch(self, runner) -> None:
        """Record validation logs.

        Args:
            runner (Runner): The runner of the training process.
        """
        self._log_val(runner)

    def after_run(self, runner) -> None:
        """Copy logs to ``self.out_dir`` if ``self.out_dir is not None``

        Args:
            runner (Runner): The runner of the training process.
        """
        # copy or upload logs to self.out_dir
        if self.out_dir is None:
            return
        for filename in scandir(runner.work_dir, self.out_suffix, True):
            local_filepath = osp.join(runner.work_dir, filename)
            out_filepath = self.file_client.join_path(self.out_dir, filename)
            with open(local_filepath, 'r') as f:
                self.file_client.put_text(f.read(), out_filepath)

            runner.logger.info(
                (f'The file {local_filepath} has been uploaded to '
                 f'{out_filepath}.'))

            if not self.keep_local:
                os.remove(local_filepath)
                runner.logger.info((f'{local_filepath} was removed due to the '
                                    '`self.keep_local=False`'))

    @master_only
    def _log_train(self, runner) -> None:
        """Collect and record training logs which start named with "train/*".

        Args:
            runner (Runner): The runner of the training process.
        """
        tag = self._collect_info(runner, 'train')
        # The training log default defines `lr`, `momentum`, `time` and
        # `data_time`. `log_tag` will pop these keys and loop other keys to
        # `log_str`.
        log_tag = copy.deepcopy(tag)
        cur_iter = self._get_iter(runner, inner_iter=True)
        cur_epoch = self._get_epoch(runner, 'train')

        # Record learning rate and momentum.
        lr_str_list = []
        momentum_str_list = []
        for key, value in tag.items():
            if key.startswith('lr'):
                log_tag.pop(key)
                lr_str_list.append(f'{key}: {value:.3e}')
        lr_str = ' '.join(lr_str_list)
        for key, value in tag.items():
            if key.startswith('momentum'):
                log_tag.pop(key)
                momentum_str_list.append(f'{key}: {value:.3e}')
        momentum_str = ' '.join(momentum_str_list)
        lr_momentum_str = f'{lr_str} {momentum_str}'
        # by epoch: Epoch [4][100/1000]
        # by iter:  Iter [100/100000]
        if self.by_epoch:
            log_str = f'Epoch [{cur_epoch}]' \
                      f'[{cur_iter}/{len(runner.train_loop.dataloader)}]\t'
        else:
            log_str = f'Iter [{cur_iter}/{runner.train_loop.max_iters}]\t'
        log_str += f'{lr_momentum_str}, '
        # Calculate eta time.
        self.time_sec_tot += (tag['time'] * self.interval)
        time_sec_avg = self.time_sec_tot / (runner.iter - self.start_iter + 1)
        eta_sec = time_sec_avg * (
            runner.train_loop.max_iters - runner.iter - 1)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        log_str += f'eta: {eta_str}, '
        log_str += f'time: {tag["time"]:.3f}, ' \
                   f'data_time: {tag["data_time"]:.3f}, '
        # Pop recorded keys
        log_tag.pop('time')
        log_tag.pop('data_time')
        # statistic memory
        if torch.cuda.is_available():
            log_str += f'memory: {self._get_max_memory(runner)}, '
        # Loop left keys to fill `log_str`.
        log_items = []
        for name, val in log_tag.items():
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        log_str += ', '.join(log_items)
        runner.logger.info(log_str)
        # Write logs to local, tensorboad, and wandb.
        runner.writer.add_scalars(
            tag, step=runner.iter + 1, file_path=self.json_log_path)

    @master_only
    def _log_val(self, runner) -> None:
        """Collect and record training logs which start named with "val/*".

        Args:
            runner (Runner): The runner of the training process.
        """
        tag = self._collect_info(runner, 'val')
        # Compatible with function `log` https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/logger/text.py # noqa E501
        eval_iter = len(runner.val_loop.dataloader)
        cur_iter = self._get_iter(runner)
        cur_epoch = self._get_epoch(runner, 'val')
        # val/test time
        # here 1000 is the length of the val dataloader
        # by epoch: Epoch[val] [4][1000]
        # by iter: Iter[val] [1000]
        if self.by_epoch:
            # runner.epoch += 1 has been done before val workflow
            log_str = f'Epoch(val) [{cur_epoch}][{eval_iter}]\t'
        else:
            log_str = f'Iter(val) [{eval_iter}]\t'

        log_items = []
        for name, val in tag.items():
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        log_str += ', '.join(log_items)
        runner.logger.info(log_str)
        # Write tag.
        runner.writer.add_scalars(
            tag, step=cur_iter, file_path=self.json_log_path)

    def _get_window_size(self, runner, window_size: Union[int, str]) \
            -> int:
        """Parse window_size specified in ``self.custom_keys`` to int value.

        Args:
            runner (Runner): The runner of the training process.
            window_size (int or str): Smoothing scale of logs.

        Returns:
            int: Smoothing window for statistical methods.
        """
        if isinstance(window_size, int):
            assert window_size == self.interval, \
                'The value of windows size must equal to LoggerHook.interval'
            return window_size
        elif window_size == 'epoch':
            return self._inner_iter + 1
        elif window_size == 'global':
            return runner.iter + 1
        else:
            raise ValueError('window_size should be int, epoch or global, but '
                             f'got invalid {window_size}')

    def _collect_info(self, runner, mode: str) -> dict:
        """Collect log information to a dict according to mode.

        Args:
            runner (Runner): The runner of the training process.
            mode (str): 'train' or 'val', which means the prefix attached by
                runner.

        Returns:
            dict: Statistical values of logs.
        """
        tag = OrderedDict()
        log_buffers = runner.message_hub.log_buffers
        mode_log_buffers = OrderedDict()
        # Filter log_buffers which starts with `mode`.
        for prefix_key, log_buffer in log_buffers.items():
            if prefix_key.startswith(mode):
                key = prefix_key.split('/')[-1]
                mode_log_buffers[key] = log_buffer
        # Ensure all metric and lr values are latest.
        for key in mode_log_buffers:
            # Update the latest learning rate and smoothed time logs.
            if key in self.fixed_smooth_keys or key.startswith('loss'):
                tag[key] = mode_log_buffers[key].mean(self.interval)
            else:
                tag[key] = mode_log_buffers[key].current()
        # Update custom keys.
        if mode == 'train':
            for log_key, log_cfg in self.custom_keys.items():
                self._parse_custom_keys(runner, log_key,
                                        copy.deepcopy(log_cfg),
                                        mode_log_buffers, tag)
        return tag

    def _parse_custom_keys(self, runner, log_key: str, log_cfg: dict,
                           log_buffers: OrderedDict, tag: OrderedDict) -> None:
        """Statistics logs in log_buffers according to custom_keys.

        Args:
            runner (Runner): The runner of the training process.
            log_key (str): log key specified in ``self.custom_keys``
            log_cfg (dict): A config dict for describing the logging
                statistics method.
            log_buffers (OrderedDict): All logs for the corresponding phase.
            tag (OrderedDict): A dict which defines all statistic values of
                logs.
        """
        if isinstance(log_cfg, list):
            log_names = set()
            for cfg in log_cfg:
                log_name = cfg.get('log_name', None)
                if log_name in log_names:
                    raise KeyError(f'{cfg["log_name"]} cannot be redefined in '
                                   'log_key')
                if log_name is not None:
                    log_names.add(log_name)
                self._parse_custom_keys(runner, log_key, cfg, log_buffers, tag)
            assert len(log_names) == len(log_cfg) - 1, \
                f'{log_key} cannot be overwritten multiple times, please ' \
                f'check only one key does not contain `log_name` in {log_cfg}.'
        elif isinstance(log_cfg, dict):
            if 'window_size' in log_cfg:
                log_cfg['window_size'] = \
                    self._get_window_size(runner, log_cfg['window_size'])
            if 'log_name' in log_cfg:
                name = log_cfg.pop('log_name')
            else:
                name = log_key
            tag[name] = log_buffers[log_key].statistics(**log_cfg).item()
        else:
            raise ValueError('The structure of `LoggerHook.custom key` is '
                             'wrong, please make sure the type of each key is '
                             'dict or list.')

    def _get_max_memory(self, runner) -> int:
        """Returns the maximum GPU memory occupied by tensors in megabytes (MB)
        for a given device.

        Args:
            runner (Runner): The runner of the training process.

        Returns:
            The maximum GPU memory occupied by tensors in megabytes for a given
            device.
        """
        device = getattr(runner.model, 'output_device', None)
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                              dtype=torch.int,
                              device=device)
        torch.cuda.reset_peak_memory_stats()
        return int(mem_mb.item())

    def _check_custom_keys(self) -> None:
        """Check the legality of ``self.custom_keys``.

        If ``self.by_epoch==False``, ``window_size`` should not be "epoch". The
        key of ``self.fixed_smooth_keys`` cannot be overwritten.
        """

        def _check_window_size(item):
            if not self.by_epoch:
                assert item['window_size'] != 'epoch', \
                    'window_size cannot be epoch if LoggerHook.by_epoch is ' \
                    'False.'

        def _check_fixed_keys(key, item):
            if key in self.fixed_smooth_keys:
                assert 'log_name' in item, f'{key} cannot be overwritten by ' \
                                           'custom keys!'

        for key, value in self.custom_keys.items():
            if isinstance(value, Sequence):
                [(_check_window_size(item), _check_fixed_keys(key, item))
                 for item in value]

            else:
                _check_window_size(value)
                _check_fixed_keys(key, value)

    def _get_epoch(self, runner, mode: str) -> int:
        """Get epoch according to mode.

        Args:
            runner (Runner): The runner of the training process.
            mode (str): Train or val.

        Returns:
            int: The current epoch.
        """
        if mode == 'train':
            epoch = runner.epoch + 1
        elif mode == 'val':
            # normal val mode
            # runner.epoch += 1 has been done before val workflow
            epoch = runner.epoch
        else:
            raise ValueError(f"runner mode should be 'train' or 'val', "
                             f'but got {runner.mode}')
        return epoch

    def _get_iter(self, runner, inner_iter=False) -> int:
        """Get the current training iteration step.
        Args:
            runner (Runner): The runner of the training process.
            inner_iter (bool): Whether to return the inner iter of an epoch.
                Defaults to False.

        Returns:
            int: The current global iter or inner iter.
        """
        if self.by_epoch and inner_iter:
            current_iter = self._inner_iter + 1
        else:
            current_iter = runner.iter + 1
        return current_iter
