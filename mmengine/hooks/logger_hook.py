import copy
from typing import Optional, Union, Tuple, Sequence
from collections import OrderedDict
import datetime
import os.path as osp
import os

import torch
import torch.distributed as dist

from mmengine.data import BaseDataSample
from mmengine.fileio import FileClient
from mmengine.utils import is_tuple_of, scandir
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class LoggerHook(Hook):
    """Logger hook in text.

    In this logger hook, the information will be printed on terminal and
    saved in json file.

    Args:
        by_epoch (bool): Whether EpochBasedRunner is used.
            Defaults to True.
        interval (int): Logging interval (every k iterations).
            Defaults to 10.
        custom_keys (dict, optional): Using the specified method to statistics
            the logs referred in custom_keys. Defaults to None.

            - The key of customs_keys represent the name of log, such as loss,
            lr. The value of customs_keys is a dict which contains the
            statistics method and corresponding arguments. If ``log_name`` is
            not defined in value, the old key will be overwritten by the
            sepecified statistics method, otherwise a new log named with
            ``log_name`` will be recorded.
            - The key in ``LoggerHook.fixed_smooth_keys`` cannot be overwritten
            because ``time`` and ``iter_time`` will be used to calulate
            estimated time of arrival. If you want to recount the time, you
            should set ``log_name`` in corresponding values.
        ignore_last (bool): Ignore the log of last iterations in each epoch if
            less than :attr:`interval`. Defaults to True.
        interval_exp_name (int): Logging interval for experiment name. This
            feature is to help users conveniently get the experiment
            information from screen or log file. Defaults to 1000.
        out_dir (str, optional): Logs are saved in ``runner.work_dir`` default.
            If ``out_dir`` is specified, logs will be copied to a new directory
            which is the concatenation of ``out_dir`` and the last level
            directory of ``runner.work_dir``. Defaults to None.
        out_suffix (Union[Tuple[str], str]): Those filenames ending with
            ``out_suffix`` will be copied to ``out_dir``. Defaults to
            ('.log.json', '.log', '.py').
        keep_local (bool): Whether to keep local log when :attr:`out_dir` is
            specified. If False, the local log will be removed. Defaults to
            True.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None.

    Examples:
        >>> from mmengine import HOOKS
        >>> logger_hook_cfg = \
        >>>        dict(by_epoch=True,
        >>>                custom_keys=dict(
        >>>                        loss=[dict(log_name='loss_mean_window',
        >>>                                   method_name='mean',
        >>>                                   window_size=10),
        >>> # loss_mean_window will be additional records.
        >>>                              dict(method_name='mean',
        >>>                                   window_size='global')],
        >>> # loss will be overwritten by global mean statistics.
        >>>                        time=dict(log_name='global_time',
        >>>                                  method='mean',
        >>>                                  window_size='global'))
        >>> # time cannot be overwritten, global time will be additional
        >>> # records.
        >>> logger_hook = HOOKS.build(logger_hook_cfg)
    """
    # eta will be calculated by time. `time` and `data_time` should not be
    # overwritten.
    fixed_smooth_keys = ('time', 'data_time')

    def __init__(self,
                 by_epoch: bool = True,
                 interval: int = 10,
                 custom_keys: Optional[dict] = None,
                 ignore_last: bool = True,
                 interval_exp_name: int = 1000,
                 out_dir: Optional[str] = None,
                 out_suffix: Optional[Union[Tuple[str], str]] =
                 ('.log.json', '.log', '.py'),
                 keep_local=True,
                 file_client_args=None,
                 ):
        self.by_epoch = by_epoch
        self.interval = interval
        self.custom_keys = custom_keys if custom_keys is not None else dict()
        self.composed_writer = None
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

    def before_run(self, runner: object) -> None:
        """ Infer ``self.file_client`` from ``self.out_dir``. Initialize
        the ``self.start_iter`` and record the meta information.
        Args:
            runner: The runner of the training process.
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
        self.start_iter = runner.iter
        if runner.meta is not None:
            runner.composed_writer.add_scalars(runner.meta)

    def after_train_iter(
            self,
            runner: object,
            data_batch: Optional[Sequence[BaseDataSample]] = None,
            outputs: Optional[Sequence[BaseDataSample]] = None) -> None:
        """Record training logs.

        Args:
            runner (object): The runner of the training process.
            data_batch (Sequence[BaseDataSample], optional): Data from
                dataloader. Defaults to None.
            outputs (Sequence[BaseDataSample], optional): Outputs from model.
                Defaults to None.
        """
        if self.by_epoch and self.every_n_inner_iters(runner, self.interval):
            self.log_train(runner)
        elif not self.by_epoch and self.every_n_iters(runner, self.interval):
            self.log_train(runner)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            # not precise but more stable
            self.log_train(runner)

    def after_val_epoch(self, runner: object) -> None:
        """Record validation logs.
        Args:
            runner (object): The runner of the training process.
        """
        self.log_val(runner)

    def after_run(self, runner: object) -> None:
        """Copy logs to ``self.out_dir`` if ``self.out_dir is not None``

        Args:
            runner (object): The runner of the training process.
        """
        # copy or upload logs to self.out_dir
        if self.out_dir is not None:
            for filename in scandir(runner.work_dir, self.out_suffix, True):
                local_filepath = osp.join(runner.work_dir, filename)
                out_filepath = self.file_client.join_path(
                    self.out_dir, filename)
                with open(local_filepath, 'r') as f:
                    self.file_client.put_text(f.read(), out_filepath)

                runner.logger.info(
                    (f'The file {local_filepath} has been uploaded to '
                     f'{out_filepath}.'))

                if not self.keep_local:
                    os.remove(local_filepath)
                    runner.logger.info(
                        (f'{local_filepath} was removed due to the '
                         '`self.keep_local=False`'))

    def log_train(self, runner: object) -> None:
        """Collect and record training logs which start named with "train/*".

        Args:
            runner (object): The runner of the training process.
        """
        tag = self._collect_info(runner, 'train')
        # `log_tag` will pop some keys and fill `log_str`.
        log_tag = copy.deepcopy(tag)
        cur_iter = self.get_iter(runner, inner_iter=True)
        cur_epoch = self.get_epoch(runner, 'train')

        if runner.meta is not None and 'exp_name' in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                    self.by_epoch and self.end_of_epoch(runner)):
                exp_info = f'Exp name: {runner.meta["exp_name"]}'
                runner.logger.info(exp_info)
        # Record learning rate and momentum.
        lr_str = []
        momentum_str = []
        for key, value in tag.items():
            if key.startswith('lr'):
                log_tag.pop(key)
                lr_str.append(f'{key}: {value:.3e}')
        lr_str = ' '.join(lr_str)
        for key, value in tag.items():
            if key.startswith('momentum'):
                log_tag.pop(key)
                momentum_str.append(f'{key}: {value:.3e}')
        momentum_str = ' '.join(momentum_str)
        lr_momentum_str = f'{lr_str} {momentum_str}'
        # by epoch: Epoch [4][100/1000]
        # by iter:  Iter [100/100000]
        if self.by_epoch:
            log_str = f'Epoch [{cur_epoch}]' \
                      f'[{cur_iter}/{len(runner.data_loader)}]\t'
        else:
            log_str = f'Iter [{cur_iter}/{runner.max_iters}]\t'
        log_str += f'{lr_momentum_str}, '
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
        runner.composed_writer.add_scalars(self.json_log_path,
                                            tag,
                                            runner.iter+1)

    def log_val(self, runner: object) -> None:
        """Collect and record training logs which start named with "val/*".

        Args:
            runner (object): The runner of the training process.
        """
        tag = self._collect_info(runner, 'val')
        # Compatible with function `log` https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/logger/text.py # noqa E501
        eval_iter = len(runner.data_loader)
        cur_iter = self.get_iter(runner)
        cur_epoch = self.get_epoch(runner, 'val')
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
        runner.composed_writer.add_scalars(self.json_log_path,
                                            tag,
                                            cur_iter)

    def _get_window_size(self, runner: object, window_size: Union[int, str]) \
            -> int:
        """Parse window_size specified in ``self.custom_keys`` to int value.

        Args:
            runner (object): The runner of the training process.
            window_size (Union[int, str]): Smoothing scale of logs.

        Returns:
            int: Smoothing window for statistical methods.
        """
        if isinstance(window_size, int):
            assert window_size == self.interval, \
                'The value of windows size must equal to LoggerHook.interval'
            return window_size
        else:
            if window_size == 'epoch':
                return runner.inner_iter + 1
            elif window_size == 'global':
                return runner.iter + 1

    def _collect_info(self, runner: object, mode: str) -> dict:
        """Collect log information to a dict according to mode.

        Args:
            runner (object): The runner of the training process.
            mode (str): "train" or "val", which means the prefix attached by runner.

        Returns:
            dict: Statistical values of logs.
        """
        tag = OrderedDict()
        log_buffers = runner.message_hub.log_buffers
        mode_log_buffers = OrderedDict()
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

    def _parse_custom_keys(self,
                           runner: object,
                           log_key: str,
                           log_cfg: dict,
                           log_buffers: OrderedDict,
                           tag: OrderedDict) -> None:
        """Statistics logs in log_buffers according to custom_keys.

        Args:
            runner (object): The runner of the training process.
            log_key (str): log key specified in ``self.custom_keys``
            log_cfg (dict): A config dict for describing the logging
                statistics method.
            log_buffers (OrderedDict): All logs for the corresponding phase.
            tag (OrderedDict): A dict which contains all statistic values of
                logs.
        """
        if isinstance(log_cfg, list):
            for cfg in log_cfg:
                self._parse_custom_keys(runner, log_key, cfg, log_buffers, tag)
        if isinstance(log_cfg, dict):
            if 'window_size' in log_cfg:
                log_cfg['window_size'] = \
                    self._get_window_size(runner, log_cfg['window_size'])
            if 'log_name' in log_cfg:
                name = log_cfg.pop('log_name')
            else:
                name = log_key
            tag[name] = log_buffers[log_key].statistics(**log_cfg)

    def _get_max_memory(self, runner: object) -> int:
        """Returns the maximum GPU memory occupied by tensors in bytes for a
        given device.

        Args:
            runner (object): The runner of the training process.

        Returns:
            The maximum GPU memory occupied by tensors in bytes for a given
            device.
        """
        device = getattr(runner.model, 'output_device', None)
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                              dtype=torch.int,
                              device=device)
        if runner.world_size > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

    def _check_custom_keys(self) -> None:
        """Check the legality of ``self.custom_keys``. If
        ``self.by_epoch==False``, ``window_size`` should not be "epoch". The
        key of ``self.fixed_smooth_keys`` cannot be overwritten.
        """
        if self.custom_keys is None:
            return

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

    def get_epoch(self, runner, mode):
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

    def get_iter(self, runner, inner_iter=False):
        """Get the current training iteration step."""
        if self.by_epoch and inner_iter:
            current_iter = runner.inner_iter + 1
        else:
            current_iter = runner.iter + 1
        return current_iter