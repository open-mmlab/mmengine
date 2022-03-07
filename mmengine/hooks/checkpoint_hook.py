# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

from mmengine.data import BaseDataSample
from mmengine.fileio import FileClient
from mmengine.registry import HOOKS
from .hook import Hook


@HOOKS.register_module()
class CheckpointHook(Hook):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional | Path): The root directory to save checkpoints.
            If not specified, ``runner.work_dir`` will be used by default. If
            specified, the ``out_dir`` will be the concatenation of ``out_dir``
            and the last level directory of ``runner.work_dir``. For example,
            if the input ``our_dir`` is ``./tmp`` and ``runner.work_dir`` is
            ``./work_dir/cur_exp``, then the ckpt will be saved in
            ``./tmp/cur_exp``. Deafule to None.
        max_keep_ckpts (int): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Default: -1, which means unlimited.
        save_last (bool): Whether to force the last checkpoint to be
            saved regardless of interval. Default: True.
        sync_buffer (bool): Whether to synchronize buffers in
            different gpus. Default: False.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.
    """

    priority = 'VERY_LOW'

    def __init__(self,
                 interval: int = -1,
                 by_epoch: bool = True,
                 save_optimizer: bool = True,
                 out_dir: Union[str, Path] = None,
                 max_keep_ckpts: int = -1,
                 save_last: bool = True,
                 sync_buffer: bool = False,
                 file_client_args: Optional[dict] = None,
                 **kwargs) -> None:
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs
        self.sync_buffer = sync_buffer
        self.file_client_args = file_client_args

    def before_run(self, runner: object) -> None:
        """Finish all operations, related to checkpoint.

        This function will get the appropriate file client, and the directory
        to save these checkpoints of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        if not self.out_dir:
            self.out_dir = runner.work_dir  # type: ignore

        self.file_client = FileClient.infer_client(self.file_client_args,
                                                   self.out_dir)

        # if `self.out_dir` is not equal to `runner.work_dir`, it means that
        # `self.out_dir` is set so the final `self.out_dir` is the
        # concatenation of `self.out_dir` and the last level directory of
        # `runner.work_dir`
        if self.out_dir != runner.work_dir:  # type: ignore
            basename = osp.basename(
                runner.work_dir.rstrip(  # type: ignore
                    osp.sep))
            self.out_dir = self.file_client.join_path(
                self.out_dir,  # type: ignore
                basename)

        runner.logger.info((  # type: ignore
            f'Checkpoints will be saved to {self.out_dir} by '
            f'{self.file_client.name}.'))

        # disable the create_symlink option because some file backends do not
        # allow to create a symlink
        if 'create_symlink' in self.args:
            if self.args[
                    'create_symlink'] and not self.file_client.allow_symlink:
                self.args['create_symlink'] = False
                warnings.warn(
                    ('create_symlink is set as True by the user but is changed'
                     'to be False because creating symbolic link is not '
                     f'allowed in {self.file_client.name}'))
        else:
            self.args['create_symlink'] = self.file_client.allow_symlink

    def after_train_epoch(self, runner: object) -> None:
        """Save the checkpoint and synchronize buffers after each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        if not self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` epochs
        # 2. reach the last epoch of training
        if self.every_n_epochs(
                runner, self.interval) or (self.save_last
                                           and self.is_last_epoch(runner)):
            runner.logger.info(  # type: ignore
                f'Saving checkpoint at \
                    {runner.epoch + 1} epochs')  # type: ignore
            if self.sync_buffer:
                pass
                # TODO
            self._save_checkpoint(runner)

    # TODO Add master_only decorator
    def _save_checkpoint(self, runner: object) -> None:
        """Save the current checkpoint and delete outdated checkpoint.

        Args:
            runner (Runner): The runner of the training process.
        """
        runner.save_checkpoint(  # type: ignore
            self.out_dir,
            save_optimizer=self.save_optimizer,
            **self.args)
        if runner.meta is not None:  # type: ignore
            if self.by_epoch:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl',
                    'epoch_{}.pth').format(runner.epoch + 1)  # type: ignore
            else:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl',
                    'iter_{}.pth').format(runner.iter + 1)  # type: ignore
            runner.meta.setdefault('hook_msgs', dict())  # type: ignore
            runner.meta['hook_msgs'][  # type: ignore
                'last_ckpt'] = self.file_client.join_path(
                    self.out_dir, cur_ckpt_filename)  # type: ignore
        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            if self.by_epoch:
                name = 'epoch_{}.pth'
                current_ckpt = runner.epoch + 1  # type: ignore
            else:
                name = 'iter_{}.pth'
                current_ckpt = runner.iter + 1  # type: ignore
            redundant_ckpts = range(
                current_ckpt - self.max_keep_ckpts * self.interval, 0,
                -self.interval)
            filename_tmpl = self.args.get('filename_tmpl', name)
            for _step in redundant_ckpts:
                ckpt_path = self.file_client.join_path(
                    self.out_dir, filename_tmpl.format(_step))  # type: ignore
                if self.file_client.isfile(ckpt_path):
                    self.file_client.remove(ckpt_path)
                else:
                    break

    def after_train_iter(
            self,
            runner: object,
            data_batch: Optional[Sequence[Tuple[Any, BaseDataSample]]] = None,
            outputs: Optional[Sequence[BaseDataSample]] = None) -> None:
        """Save the checkpoint and synchronize buffers after each iteration.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional): Data
                from dataloader. Defaults to None.
            outputs (Sequence[BaseDataSample], optional): Outputs from model.
                Defaults to None.
        """
        if self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training
        if self.every_n_iters(
                runner, self.interval) or (self.save_last
                                           and self.is_last_iter(runner)):
            runner.logger.info(  # type: ignore
                f'Saving checkpoint at \
                    {runner.iter + 1} iterations')  # type: ignore
            if self.sync_buffer:
                pass
                # TODO
            self._save_checkpoint(runner)
