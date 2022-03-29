# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

from mmengine.data import BaseDataSample
from mmengine.fileio import FileClient
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.utils import is_tuple_of, scandir

DATA_BATCH = Optional[Sequence[Tuple[Any, BaseDataSample]]]


@HOOKS.register_module()
class LoggerHook(Hook):
    """Collect logs from different components of ``Runner`` and write them to
    terminal, JSON file, tensorboard and wandb .etc.

    ``LoggerHook`` is used to control following behaviers:

    - The frequency of logs update in terminal, local, tensorboad wandb.etc.
    - The frequency of show experiment information in terminal.
    - The work directory to save logs.

    ``LoggerHook`` is used to record logs formatted by ``LogProcessor`` during
     training/validation/testing phase.

    Args:
        interval (int): Logging interval (every k iterations).
            Defaults to 10.
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
        >>> # A simplest LoggerHook config.
        >>> logger_hook_cfg = dict(interval=20)
    """
    priority = 'BELOW_NORMAL'

    def __init__(
        self,
        interval: int = 10,
        ignore_last: bool = True,
        interval_exp_name: int = 1000,
        out_dir: Optional[Union[str, Path]] = None,
        out_suffix: Union[Sequence[str], str] = ('.log.json', '.log', '.py'),
        keep_local=True,
        file_client_args=None,
    ):
        self.interval = interval
        self.ignore_last = ignore_last
        self.interval_exp_name = interval_exp_name

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

    def before_run(self, runner: 'Runner') -> None:
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
        # TODO Compatible with Visualizer.
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
            data_batch (Sequence[BaseDataSample], optional): Data from
                dataloader. Defaults to None.
            outputs (dict, optional): Outputs from model.
                Defaults to None.
        """
        if (self.every_n_iters(runner, self.interval_exp_name)) or (
                runner.log_processor.by_epoch and self.end_of_epoch(
                runner, batch_idx)):
            exp_info = f'Exp name: {runner.experiment_name}'
            runner.logger.info(exp_info)
        if runner.log_processor.by_epoch and \
                self.every_n_inner_iters(batch_idx, self.interval):
            tag, log_str = runner.log_processor.get_log(
                runner, batch_idx, 'train')
        elif not runner.log_processor.by_epoch and \
                self.every_n_iters(runner, self.interval):
            tag, log_str = runner.log_processor.get_log(
                runner, batch_idx, 'train')
        elif self.end_of_epoch(runner, batch_idx) and not self.ignore_last:
            # `runner.max_iters` may not be divisible by `self.interval`. if
            # `self.ignore_last==True`, the log of remaining iterations will
            # be recorded (Epoch [4][1000/1007], the logs of 998-1007
            # iterations will be recorded).
            tag, log_str = runner.log_processor.get_log(
                runner, batch_idx, 'train')
        else:
            return
        runner.logger.info(log_str)
        # TODO compatible with visualizer.
        runner.writer.add_scalars(tag, step=runner.iter+1)

    def after_val_epoch(self, runner) -> None:
        """Record validation logs.

        Args:
            runner (Runner): The runner of the training process.
        """
        tag, log_str = runner.log_processor.get_log(
            runner, len(runner.cur_dataloader), 'val')
        runner.logger.info(log_str)
        # TODO compatible with visualizer.
        runner.writer.add_scalars(tag, step=runner.iter+1)

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
