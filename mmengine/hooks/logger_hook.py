# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from mmengine.fileio import FileClient, dump
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.utils import is_tuple_of, scandir

DATA_BATCH = Optional[Union[dict, tuple, list]]
SUFFIX_TYPE = Union[Sequence[str], str]


@HOOKS.register_module()
class LoggerHook(Hook):
    """Collect logs from different components of ``Runner`` and write them to
    terminal, JSON file, tensorboard and wandb .etc.

    ``LoggerHook`` is used to record logs formatted by ``LogProcessor`` during
    training/validation/testing phase. It is used to control following
    behaviors:

    - The frequency of logs update in terminal, local, tensorboad wandb.etc.
    - The frequency of show experiment information in terminal.
    - The work directory to save logs.

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
            of ``out_dir`` and the last level directory of ``runner.work_dir``.
            For example, if the input ``our_dir`` is ``./tmp`` and
            ``runner.work_dir`` is ``./work_dir/cur_exp``, then the log will be
            saved in ``./tmp/cur_exp``. Defaults to None.
        out_suffix (Tuple[str] or str): Those files in ``runner._log_dir``
            ending with ``out_suffix`` will be copied to ``out_dir``. Defaults
            to ('json', '.log', '.py').
        keep_local (bool): Whether to keep local logs in the local machine
            when :attr:`out_dir` is specified. If False, the local log will be
            removed. Defaults to True.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None.
        log_metric_by_epoch (bool): Whether to output metric in validation step
            by epoch. It can be true when running in epoch based runner.
            If set to True, `after_val_epoch` will set `step` to self.epoch in
            `runner.visualizer.add_scalars`. Otherwise `step` will be
            self.iter. Default to True.

    Examples:
        >>> # The simplest LoggerHook config.
        >>> logger_hook_cfg = dict(interval=20)
    """
    priority = 'BELOW_NORMAL'

    def __init__(self,
                 interval: int = 10,
                 ignore_last: bool = True,
                 interval_exp_name: int = 1000,
                 out_dir: Optional[Union[str, Path]] = None,
                 out_suffix: SUFFIX_TYPE = ('.json', '.log', '.py', 'yaml'),
                 keep_local: bool = True,
                 file_client_args: Optional[dict] = None,
                 log_metric_by_epoch: bool = True):
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
        self.json_log_path: Optional[str] = None
        if self.out_dir is not None:
            self.file_client = FileClient.infer_client(file_client_args,
                                                       self.out_dir)
        self.log_metric_by_epoch = log_metric_by_epoch

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
                f'Text logs will be saved to {self.out_dir} by '
                f'{self.file_client.name} after the training process.')

        self.json_log_path = f'{runner.timestamp}.json'

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Record logs after training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        # Print experiment name every n iterations.
        if self.every_n_train_iters(
                runner, self.interval_exp_name) or (self.end_of_epoch(
                    runner.train_dataloader, batch_idx)):
            exp_info = f'Exp name: {runner.experiment_name}'
            runner.logger.info(exp_info)
        if self.every_n_inner_iters(batch_idx, self.interval):
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'train')
        elif (self.end_of_epoch(runner.train_dataloader, batch_idx)
              and not self.ignore_last):
            # `runner.max_iters` may not be divisible by `self.interval`. if
            # `self.ignore_last==True`, the log of remaining iterations will
            # be recorded (Epoch [4][1000/1007], the logs of 998-1007
            # iterations will be recorded).
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'train')
        else:
            return
        runner.logger.info(log_str)
        runner.visualizer.add_scalars(
            tag, step=runner.iter + 1, file_path=self.json_log_path)

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence] = None) -> None:
        """Record logs after validation iteration.

        Args:
            runner (Runner): The runner of the validation process.
            batch_idx (int): The index of the current batch in the validation
                loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
                Defaults to None.
            outputs (sequence, optional): Outputs from model.
        """
        if self.every_n_inner_iters(batch_idx, self.interval):
            _, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'val')
            runner.logger.info(log_str)

    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Sequence] = None) -> None:
        """Record logs after testing iteration.

        Args:
            runner (Runner): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (sequence, optional): Outputs from model.
        """
        if self.every_n_inner_iters(batch_idx, self.interval):
            _, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'test')
            runner.logger.info(log_str)

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.val_dataloader), 'val')
        runner.logger.info(log_str)
        if self.log_metric_by_epoch:
            # when `log_metric_by_epoch` is set to True, it's expected
            # that validation metric can be logged by epoch rather than
            # by iter. At the same time, scalars related to time should
            # still be logged by iter to avoid messy visualized result.
            # see details in PR #278.
            metric_tags = {k: v for k, v in tag.items() if 'time' not in k}
            runner.visualizer.add_scalars(
                metric_tags, step=runner.epoch, file_path=self.json_log_path)
        else:
            runner.visualizer.add_scalars(
                tag, step=runner.iter, file_path=self.json_log_path)

    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each test epoch.

        Args:
            runner (Runner): The runner of the testing process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on test dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.test_dataloader), 'test')
        runner.logger.info(log_str)
        dump(tag, osp.join(runner.log_dir, self.json_log_path))  # type: ignore

    def after_run(self, runner) -> None:
        """Copy logs to ``self.out_dir`` if ``self.out_dir is not None``

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
        """
        # copy or upload logs to self.out_dir
        if self.out_dir is None:
            return
        for filename in scandir(runner._log_dir, self.out_suffix, True):
            local_filepath = osp.join(runner._log_dir, filename)
            out_filepath = self.file_client.join_path(self.out_dir, filename)
            with open(local_filepath) as f:
                self.file_client.put_text(f.read(), out_filepath)

            runner.logger.info(
                f'The file {local_filepath} has been uploaded to '
                f'{out_filepath}.')

            if not self.keep_local:
                os.remove(local_filepath)
                runner.logger.info(f'{local_filepath} was removed due to the '
                                   '`self.keep_local=False`')
