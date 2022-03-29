import copy
from collections import OrderedDict
import torch
from mmengine.runner import runner
from typing import List, Optional


class LogProcessor:
    """A log prosessor used to format ``runner.message_hub.log_scalars``.

    ``LogProcessor`` instance is built by runner and will format
    ``runner.message_hub.log_scalars`` to ``tag`` and ``log_str``, which can
    direcly used by ``Visualizer`` and ``MMLogger``. Besides, the argument
    custom_cfg of constructor can control the statitics method of logs.

    Args:
        window_size (int): default smooth interval Defaults to 10.
        by_epoch (bool): Whether to format logs with epoch stype. Defaults to
            True.
        custom_cfg (list[dict], optional): Contains multiple log config dict,
            in which key means the data source name of log and value means the
            statistic method and corresponding arguments used to count the
            data souce. Defaults to None
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
        >>>     custom_keys=[dict(data_src='loss',
        >>>                       log_name='loss_large_window',
        >>>                       method_name='mean',
        >>>                       window_size=100)])
        >>> # `log_name` is not defined. `loss` will be overwritten.
        >>> log_processor = dict(
        >>>     window_size=10,
        >>>     by_epoch=True,
        >>>     custom_keys=[dict(data_src='loss',
        >>>                       method_name='mean',
        >>>                       window_size=100)])
        >>> # Record loss with different statistics methods.
        >>>     custom_keys=[dict(data_src='loss',
        >>>                       log_name='loss_large_window',
        >>>                       method_name='mean',
        >>>                       window_size=100)],
        >>>                  dict(data_src='loss',
        >>>                       method_name='mean',
        >>>                       window_size=100))
        >>> # Overwrite loss item twice will raise an error.
        >>>     custom_keys=[dict(data_src='loss',
        >>>                       method_name='mean',
        >>>                       window_size=100)],
        >>>                  dict(data_src='loss',
        >>>                       method_name='max',
        >>>                       window_size=100))
        >>>
    """
    def __init__(self,
                 window_size=10,
                 by_epoch=True,
                 custom_cfg: Optional[List[dict]] = None):
        self.window_size = window_size
        self.by_epoch = by_epoch
        self.custom_cfg = custom_cfg if custom_cfg else OrderedDict()
        self._check_custom_keys()

    def get_log(self, runner: 'runner.Runner',
                batch_idx: int, mode: str = 'train'):
        # Consider the `window_size` such as "epoch" and "global" will
        # change with `runner.iter` Therefore, we should use a copy of
        # `self.custom_cfg` to calculate specific window_size and keep
        # `self.custom_cfg` unchanged.
        custom_cfg = copy.deepcopy(self.custom_cfg)
        # Overwrite ``window_size`` defined in  ``custom_cfg`` to int value.
        self._parse_windows_size(custom_cfg, runner, batch_idx)
        # tag is used to write log information to different backends.
        tag = self._collect_scalars(custom_cfg, runner, mode)
        # Get iter according to mode.
        cur_iter = self._get_iter(runner, batch_idx=batch_idx)

        if mode == 'train':
            # Called by after_train_iter
            log_str = self._get_train_log_str(runner, tag, cur_iter)
        elif mode == 'val' or mode == 'test':
            # Called by after_val_epoch or after_test_epoch
            log_str = self._get_val_log_str(runner, tag, cur_iter)
        else:
            raise ValueError('mode must be train, val or test!')
        return tag, log_str

    def _get_train_log_str(self, runner: 'runner.Runner',
                           tag: dict, cur_iter: int) \
            -> str:
        """Format log string during training phases.

        Args:
            runner (Runner): The runner of training phase.
            tag (dict): Statistical values of logs which will be written to the
                diverse backends, such as local, tensorboard, wandb .etc.
            cur_iter (int): Equals to batch_index if ``self.by_epoch==True``,
                otherwise equals to ``runner.iter``.

        Return:
            str: Formatted log string which will be recorded by ``MMLogger``.
        """
        # The training log default defines `lr`, `momentum`, `time` and
        # `data_time`. `log_tag` will pop these keys and loop other keys to
        # `log_str`.
        log_tag = copy.deepcopy(tag)
        # Record learning rate.
        lr_str_list = []
        for key, value in tag.items():
            if key.startswith('lr'):
                log_tag.pop(key)
                lr_str_list.append(f'{key}: {value:.3e}')
        lr_str = ' '.join(lr_str_list)
        # Format log header.
        # by epoch: Epoch [4][100/1000]
        # by iter:  Iter [100/100000]
        if self.by_epoch:
            cur_epoch = self._get_epoch(runner, 'train')
            log_str = f'Epoch [{cur_epoch}]' \
                      f'[{cur_iter}/{len(runner.cur_dataloader)}]\t'
        else:
            log_str = f'Iter [{cur_iter}/{runner.train_loop.max_iters}]\t'
        # Concatenate lr, momentum string with log header.
        log_str += f'{lr_str}, '
        # If IterTimerHook used in runner, eta, time, and data_time should be
        # recorded.
        if all(item in tag for item in ['time', 'data_time']) and \
                'eta' in runner.message_hub.runtime_info:
            eta = runner.message_hub.get_info('eta')
            log_str += f'eta: {eta}, '
            log_str += f'time: {tag["time"]:.3f}, ' \
                       f'data_time: {tag["data_time"]:.3f}, '
            # Pop recorded keys
            log_tag.pop('time')
            log_tag.pop('data_time')

        # If cuda is available, the max memory occupied should be calculated.
        if torch.cuda.is_available():
            log_str += f'memory: {self._get_max_memory(runner)}, '
        # Loop left keys to fill `log_str`.
        log_items = []
        for name, val in log_tag.items():
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        log_str += ', '.join(log_items)
        return log_str

    def _get_val_log_str(self, runner: 'runner.Runner',
                         tag: dict, cur_iter: int) -> str:
        """Format log string during validation phases.

        Args:
            runner (Runner): The runner of training phase.
            tag (dict): Statistical values of logs which will be written to the
                diverse backends, such as local, tensorboard, wandb .etc.
            cur_iter (int): Equals to batch_index if ``self.by_epoch==True``,
                otherwise equals to ``runner.iter``.

        Return:
            str: Formatted log string which will be recorded by ``MMLogger``.
        """
        eval_iter = len(runner.cur_dataloader)
        # val/test time
        # here 1000 is the length of the val dataloader
        # by epoch: Epoch[val] [4][1000]
        # by iter: Iter[val] [4000][1000]
        if self.by_epoch:
            cur_epoch = self._get_epoch(runner, 'val')
            # runner.epoch += 1 has been done before val workflow
            log_str = f'Epoch(val) [{cur_epoch}][{eval_iter}]\t'
        else:
            log_str = f'Iter(val) [{cur_iter}][{eval_iter}]\t'

        log_items = []
        for name, val in tag.items():
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        log_str += ', '.join(log_items)
        return log_str

    def _collect_scalars(self,
                         custom_cfg: List[dict],
                         runner: 'runner.Runner',
                         mode: str) -> dict:
        """Collect log information to compose a dict according to mode.

        Args:
            List[dict] (dict): A copy of ``self.custom_cfg`` with int
                ``window_size``.
            runner (Runner): The runner of the training process.
            mode (str): 'train' or 'val', which means the prefix attached by
                runner.

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
                key = prefix_key.split('/')[-1]
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

    def _check_custom_keys(self) -> None:
        """Check the legality of ``self.custom_cfg``."""

        def _check_window_size():
            for log_cfg in self.custom_cfg:
                if not self.by_epoch:
                    assert log_cfg['window_size'] != 'epoch', \
                        'window_size cannot be epoch if LoggerHook.by_epoch' \
                        ' is False.'

        def _check_repeated_log_name():
            check_dict = dict()
            # The `log_name` of the same data_src should not be repeated.
            # If `log_name` is not specified, `data_src` will be overwritten.
            # But only allowed to be overwritten once.
            for log_cfg in self.custom_cfg:
                assert 'data_src' in log_cfg
                data_src = log_cfg['data_src']
                log_name = log_cfg.get('log_name', data_src)
                check_dict.setdefault(
                    data_src, dict(log_names=set(), log_counts=0))
                check_dict[data_src]['log_names'].add(log_name)
                check_dict[data_src]['log_counts'] += 1
                assert len(check_dict[data_src]['log_names']) == \
                       check_dict[data_src]['log_counts'], \
                       f'If you want to statistic {data_src} with multiple ' \
                       'statistics method, please check `log_name` is unique' \
                       f'and {data_src} will not be overwritten twice. See ' \
                       f'more information in the docstring of `LogProcessor`'

        _check_repeated_log_name()
        _check_window_size()

    def _parse_windows_size(self, custom_cfg: List[dict],
                            runner: 'runner.Runner',
                            batch_idx: int) -> None:
        """Parse window_size defined in custom_cfg to int value.

        Args:
            custom_cfg (List[dict]): A copy of ``self.custom_cfg``.
            runner (Runner): The runner of the training process.
            batch_idx (int): The iteration index of current dataloader.
        """
        for log_cfg in custom_cfg:
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

    def _get_max_memory(self, runner: 'runner.Runner') -> int:
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

    def _get_iter(self, runner: 'runner.Runner', batch_idx: int = None) -> int:
        """Get current training iteration step.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int, optional): The interation index of current
                dataloader. Defaults to None.

        Returns:
            int: The current global iter or inner iter.
        """
        if self.by_epoch and batch_idx:
            current_iter = batch_idx + 1
        else:
            current_iter = runner.iter + 1
        return current_iter

    def _get_epoch(self, runner: 'runner.Runner', mode: str) -> int:
        """Get current epoch according to mode.

        Args:
            runner (Runner): The runner of the training process.
            mode (str): Current mode of runner, "train" or "val".

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
            raise ValueError(f"runner mode should be 'train' or 'val', "
                             f'but got {mode}')
        return epoch
