# Copyright (c) OpenMMLab. All rights reserved.
import copy
import datetime
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch

from mmengine.registry import LOG_PROCESSOR


@LOG_PROCESSOR.register_module()
class LogProcessor:
    """A log processor used to format log information collected from
    ``runner.message_hub.log_scalars``.

    ``LogProcessor`` instance is built by runner and will format
    ``runner.message_hub.log_scalars`` to ``tag`` and ``log_str``, which can
    directly used by ``LoggerHook`` and ``MMLogger``. Besides, the argument
    ``custom_cfg`` of constructor can control the statistics method of logs.

    Args:
        window_size (int): default smooth interval Defaults to 10.
        by_epoch (bool): Whether to format logs with epoch stype. Defaults to
            True.
        custom_cfg (list[dict], optional): Contains multiple log config dict,
            in which key means the data source name of log and value means the
            statistic method and corresponding arguments used to count the
            data source. Defaults to None
            - If custom_cfg is None, all logs will be formatted via default
              methods, such as smoothing loss by default window_size. If
              custom_cfg is defined as a list of config dict, for example:
              [dict(data_src=loss, method='mean', log_name='global_loss',
              window_size='global')]. It means the log item ``loss`` will be
              counted as global mean and additionally logged as ``global_loss``
              (defined by ``log_name``). If ``log_name`` is not defined in
               config dict, the original logged key will be overwritten.

            - The original log item cannot be overwritten twice. Here is
              an error example:
              [dict(data_src=loss, method='mean', window_size='global'),
               dict(data_src=loss, method='mean', window_size='epoch')].
              Both log config dict in custom_cfg do not have ``log_name`` key,
              which means the loss item will be overwritten twice.

            - For those statistic methods with the ``window_size`` argument,
              if ``by_epoch`` is set to False, ``windows_size`` should not be
              `epoch` to statistics log value by epoch.

    Examples:
        >>> # `log_name` is defined, `loss_large_window` will be an additional
        >>> # record.
        >>> log_processor = dict(
        >>>     window_size=10,
        >>>     by_epoch=True,
        >>>     custom_cfg=[dict(data_src='loss',
        >>>                       log_name='loss_large_window',
        >>>                       method_name='mean',
        >>>                       window_size=100)])
        >>> # `log_name` is not defined. `loss` will be overwritten.
        >>> log_processor = dict(
        >>>     window_size=10,
        >>>     by_epoch=True,
        >>>     custom_cfg=[dict(data_src='loss',
        >>>                       method_name='mean',
        >>>                       window_size=100)])
        >>> # Record loss with different statistics methods.
        >>> log_processor = dict(
        >>>     window_size=10,
        >>>     by_epoch=True,
        >>>     custom_cfg=[dict(data_src='loss',
        >>>                       log_name='loss_large_window',
        >>>                       method_name='mean',
        >>>                       window_size=100),
        >>>                  dict(data_src='loss',
        >>>                       method_name='mean',
        >>>                       window_size=100)])
        >>> # Overwrite loss item twice will raise an error.
        >>> log_processor = dict(
        >>>     window_size=10,
        >>>     by_epoch=True,
        >>>     custom_cfg=[dict(data_src='loss',
        >>>                       method_name='mean',
        >>>                       window_size=100),
        >>>                  dict(data_src='loss',
        >>>                       method_name='max',
        >>>                       window_size=100)])
        AssertionError
    """

    def __init__(self,
                 window_size=10,
                 by_epoch=True,
                 custom_cfg: Optional[List[dict]] = None):
        self.window_size = window_size
        self.by_epoch = by_epoch
        self.custom_cfg = custom_cfg if custom_cfg else []
        self._check_custom_cfg()

    def get_log_after_iter(self, runner, batch_idx: int,
                           mode: str) -> Tuple[dict, str]:
        """Format log string after training, validation or testing epoch.

        Args:
            runner (Runner): The runner of training phase.
            batch_idx (int): The index of the current batch in the current
                loop.
            mode (str): Current mode of runner, train, test or val.

        Return:
            Tuple(dict, str): Formatted log dict/string which will be
            recorded by :obj:`runner.message_hub` and :obj:`runner.visualizer`.
        """
        assert mode in ['train', 'test', 'val']
        current_loop = self._get_cur_loop(runner, mode)
        cur_iter = self._get_iter(runner, batch_idx=batch_idx)
        # Overwrite ``window_size`` defined in ``custom_cfg`` to int value.
        custom_cfg_copy = self._parse_windows_size(runner, batch_idx)
        # tag is used to write log information to different backends.
        tag = self._collect_scalars(custom_cfg_copy, runner, mode)
        # `log_tag` will pop 'lr' and loop other keys to `log_str`.
        log_tag = copy.deepcopy(tag)
        # Record learning rate.
        lr_str_list = []
        for key, value in tag.items():
            if key.startswith('lr'):
                log_tag.pop(key)
                lr_str_list.append(f'{key}: {value:.3e}')
        lr_str = ' '.join(lr_str_list)
        # Format log header.
        # by_epoch == True
        #   train/val: Epoch [5][5/10]  ...
        #   test: Epoch [5/10]
        # by_epoch == False
        #  train: Epoch [5/10000] ... (divided by `max_iter`)
        #  val/test: Epoch [5/2000] ... (divided by length of dataloader)
        if self.by_epoch:
            if mode in ['train', 'val']:
                cur_epoch = self._get_epoch(runner, mode)
                log_str = (f'Epoch({mode}) [{cur_epoch}]'
                           f'[{cur_iter}/{len(current_loop.dataloader)}]  ')
            else:
                log_str = (f'Epoch({mode}) '
                           f'[{cur_iter}/{len(current_loop.dataloader)}]  ')
        else:
            if mode == 'train':
                log_str = (f'Iter({mode}) '
                           f'[{cur_iter}/{runner.max_iters}]  ')
            else:
                log_str = (f'Iter({mode}) [{batch_idx+1}'
                           f'/{len(current_loop.dataloader)}]  ')
        # Concatenate lr, momentum string with log header.
        log_str += f'{lr_str}  '
        # If IterTimerHook used in runner, eta, time, and data_time should be
        # recorded.
        if (all(item in tag for item in ['time', 'data_time'])
                and 'eta' in runner.message_hub.runtime_info):
            eta = runner.message_hub.get_info('eta')
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            log_str += f'eta: {eta_str}  '
            log_str += (f'time: {tag["time"]:.3f}  '
                        f'data_time: {tag["data_time"]:.3f}  ')
            # Pop recorded keys
            log_tag.pop('time')
            log_tag.pop('data_time')

        # If cuda is available, the max memory occupied should be calculated.
        if torch.cuda.is_available():
            log_str += f'memory: {self._get_max_memory(runner)}  '
        # Loop left keys to fill `log_str`.
        if mode in ('train', 'val'):
            log_items = []
            for name, val in log_tag.items():
                if mode == 'val' and not name.startswith('val/loss'):
                    continue
                if isinstance(val, float):
                    val = f'{val:.4f}'
                log_items.append(f'{name}: {val}')
            log_str += '  '.join(log_items)
        return tag, log_str

    def get_log_after_epoch(self, runner, batch_idx: int,
                            mode: str) -> Tuple[dict, str]:
        """Format log string after validation or testing epoch.

        Args:
            runner (Runner): The runner of validation/testing phase.
            batch_idx (int): The index of the current batch in the current
                loop.
            mode (str): Current mode of runner.

        Return:
            Tuple(dict, str): Formatted log dict/string which will be
            recorded by :obj:`runner.message_hub` and :obj:`runner.visualizer`.
        """
        assert mode in [
            'test', 'val'
        ], ('`_get_metric_log_str` only accept val or test mode, but got '
            f'{mode}')
        cur_loop = self._get_cur_loop(runner, mode)
        dataloader_len = len(cur_loop.dataloader)

        custom_cfg_copy = self._parse_windows_size(runner, batch_idx)
        # tag is used to write log information to different backends.
        tag = self._collect_scalars(custom_cfg_copy, runner, mode)
        # By epoch:
        #     Epoch(val) [10][1000/1000]  ...
        #     Epoch(test) [1000/1000] ...
        # By iteration:
        #     Iteration(val) [1000/1000]  ...
        #     Iteration(test) [1000/1000]  ...
        if self.by_epoch:
            if mode == 'val':
                cur_epoch = self._get_epoch(runner, mode)
                log_str = (f'Epoch({mode}) [{cur_epoch}][{dataloader_len}/'
                           f'{dataloader_len}]  ')
            else:
                log_str = (
                    f'Epoch({mode}) [{dataloader_len}/{dataloader_len}]  ')

        else:
            log_str = (f'Iter({mode}) [{dataloader_len}/{dataloader_len}]  ')
        # `time` and `data_time` will not be recorded in after epoch log
        # message.
        log_items = []
        for name, val in tag.items():
            if name in ('time', 'data_time'):
                continue
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        log_str += '  '.join(log_items)
        return tag, log_str

    def _collect_scalars(self, custom_cfg: List[dict], runner,
                         mode: str) -> dict:
        """Collect log information to compose a dict according to mode.

        Args:
            custom_cfg (List[dict]): A copy of ``self.custom_cfg`` with int
                ``window_size``.
            runner (Runner): The runner of the training/testing/validation
                process.
            mode (str): Current mode of runner.

        Returns:
            dict: Statistical values of logs.
        """
        tag = OrderedDict()
        # history_scalars of train/val/test phase.
        history_scalars = runner.message_hub.log_scalars
        # corresponding mode history_scalars
        mode_history_scalars = OrderedDict()
        # extract log scalars and remove prefix to `mode_history_scalars`
        # according to mode.
        for prefix_key, log_buffer in history_scalars.items():
            if prefix_key.startswith(mode):
                key = prefix_key.partition('/')[-1]
                mode_history_scalars[key] = log_buffer
        for key in mode_history_scalars:
            # Update the latest learning rate and smoothed time logs.
            if key.startswith('loss'):
                tag[key] = mode_history_scalars[key].mean(self.window_size)
            else:
                # Default statistic method is current.
                tag[key] = mode_history_scalars[key].current()
        # Update custom keys.
        for log_cfg in custom_cfg:
            data_src = log_cfg.pop('data_src')
            if 'log_name' in log_cfg:
                log_name = log_cfg.pop('log_name')
            else:
                log_name = data_src
            # log item in custom_cfg could only exist in train or val
            # mode.
            if data_src in mode_history_scalars:
                tag[log_name] = mode_history_scalars[data_src].statistics(
                    **log_cfg)
        return tag

    def _check_custom_cfg(self) -> None:
        """Check the legality of ``self.custom_cfg``."""

        def _check_window_size():
            for log_cfg in self.custom_cfg:
                if not self.by_epoch:
                    assert log_cfg['window_size'] != 'epoch', \
                        'window_size cannot be epoch if LoggerHook.by_epoch' \
                        ' is False.'

        def _check_repeated_log_name():
            # The `log_name` of the same data_src should not be repeated.
            # If `log_name` is not specified, `data_src` will be overwritten.
            # But only allowed to be overwritten once.
            check_set = set()
            for log_cfg in self.custom_cfg:
                assert 'data_src' in log_cfg
                data_src = log_cfg['data_src']
                log_name = log_cfg.get('log_name', data_src)
                assert log_name not in check_set, (
                    f'Found duplicate {log_name} for {data_src}. Please check'
                    'your `custom_cfg` for `log_processor`. You should '
                    f'neither define duplicate `{log_name}` for {data_src} '
                    f'nor do not define any {log_name} for multiple '
                    f'{data_src}, See more information in the docstring of '
                    'LogProcessor')

                check_set.add(log_name)

        _check_repeated_log_name()
        _check_window_size()

    def _parse_windows_size(self, runner, batch_idx: int) -> list:
        """Parse window_size defined in custom_cfg to int value.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            batch_idx (int): The iteration index of current dataloader.
        """
        custom_cfg_copy = copy.deepcopy(self.custom_cfg)
        for log_cfg in custom_cfg_copy:
            window_size = log_cfg.get('window_size', None)
            if window_size is None or isinstance(window_size, int):
                continue
            elif window_size == 'epoch':
                log_cfg['window_size'] = batch_idx + 1
            elif window_size == 'global':
                log_cfg['window_size'] = runner.iter + 1
            else:
                raise TypeError(
                    'window_size should be int, epoch or global, but got '
                    f'invalid {window_size}')
        return custom_cfg_copy

    def _get_max_memory(self, runner) -> int:
        """Returns the maximum GPU memory occupied by tensors in megabytes (MB)
        for a given device.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.

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

    def _get_iter(self, runner, batch_idx: int = None) -> int:
        """Get current iteration index.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            batch_idx (int, optional): The iteration index of current
                dataloader. Defaults to None.

        Returns:
            int: The current global iter or inner iter.
        """
        if self.by_epoch and batch_idx:
            current_iter = batch_idx + 1
        else:
            current_iter = runner.iter + 1
        return current_iter

    def _get_epoch(self, runner, mode: str) -> int:
        """Get current epoch according to mode.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            mode (str): Current mode of runner.

        Returns:
            int: The current epoch.
        """
        if mode == 'train':
            epoch = runner.epoch + 1
        elif mode == 'val':
            # normal val mode
            # runner.epoch += 1 has been done before validation
            epoch = runner.epoch
        else:
            raise ValueError(
                f"runner mode should be 'train' or 'val', but got {mode}")
        return epoch

    def _get_cur_loop(self, runner, mode: str):
        """Get current loop according to mode.

        Args:
            runner (Runner): The runner of the training/validation/testing
                process.
            mode (str): Current mode of runner.

        Returns:
            BaseLoop: Current loop of runner.
        """
        # returns type hint will occur circular import
        if mode == 'train':
            return runner.train_loop
        elif mode == 'val':
            return runner.val_loop
        else:
            return runner.test_loop
