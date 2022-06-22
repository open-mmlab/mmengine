# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from collections import OrderedDict
from math import inf
from pathlib import Path
from typing import Optional, Sequence, Union

from mmengine.dist import master_only
from mmengine.fileio import FileClient
from mmengine.registry import HOOKS
from mmengine.utils import is_seq_of
from .hook import Hook

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class CheckpointHook(Hook):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Defaults to -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Defaults to True.
        save_param_scheduler (bool): Whether to save param_scheduler state_dict
            in the checkpoint. It is usually used for resuming experiments.
            Defaults to True.
        out_dir (str, optional | Path): The root directory to save checkpoints.
            If not specified, ``runner.work_dir`` will be used by default. If
            specified, the ``out_dir`` will be the concatenation of ``out_dir``
            and the last level directory of ``runner.work_dir``. For example,
            if the input ``our_dir`` is ``./tmp`` and ``runner.work_dir`` is
            ``./work_dir/cur_exp``, then the ckpt will be saved in
            ``./tmp/cur_exp``. Defaults to None.
        max_keep_ckpts (int): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Defaults to -1, which means unlimited.
        save_last (bool): Whether to force the last checkpoint to be
            saved regardless of interval. Defaults to True.
        save_best (str, optional): If a metric is specified, it would measure
            the best checkpoint during evaluation. The information about best
            checkpoint would be saved in ``runner.message_hub`` to keep
            best score value and best checkpoint path, which will be also
            loaded when resuming checkpoint. Options are the evaluation metrics
            on the test dataset. e.g., ``bbox_mAP``, ``segm_mAP`` for bbox
            detection and instance segmentation. ``AR@100`` for proposal
            recall. If ``save_best`` is ``auto``, the first key of the returned
            ``OrderedDict`` result will be used. Defaults to None.
        rule (str, optional): Comparison rule for best score. If set to
            None, it will infer a reasonable rule. Keys such as 'acc', 'top'
            .etc will be inferred by 'greater' rule. Keys contain 'loss' will
            be inferred by 'less' rule. Options are 'greater', 'less', None.
            Defaults to None.
        greater_keys (List[str], optional): Metric keys that will be
            inferred by 'greater' comparison rule. If ``None``,
            _default_greater_keys will be used. Defaults to None.
        less_keys (List[str], optional): Metric keys that will be
            inferred by 'less' comparison rule. If ``None``, _default_less_keys
            will be used. Defaults to None.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Defaults to None.
    """
    out_dir: str

    priority = 'VERY_LOW'

    # logic to save best checkpoints
    # Since the key for determining greater or less is related to the
    # downstream tasks, downstream repositories may need to overwrite
    # the following inner variables accordingly.

    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    _default_greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
        'mAcc', 'aAcc'
    ]
    _default_less_keys = ['loss']

    def __init__(self,
                 interval: int = -1,
                 by_epoch: bool = True,
                 save_optimizer: bool = True,
                 save_param_scheduler: bool = True,
                 out_dir: Optional[Union[str, Path]] = None,
                 max_keep_ckpts: int = -1,
                 save_last: bool = True,
                 save_best: Optional[str] = None,
                 rule: Optional[str] = None,
                 greater_keys: Optional[Sequence[str]] = None,
                 less_keys: Optional[Sequence[str]] = None,
                 file_client_args: Optional[dict] = None,
                 **kwargs) -> None:
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.save_param_scheduler = save_param_scheduler
        self.out_dir = out_dir  # type: ignore
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs
        self.file_client_args = file_client_args

        # save best logic
        assert isinstance(save_best, str) or save_best is None, \
            '"save_best" should be a str or None ' \
            f'rather than {type(save_best)}'
        self.save_best = save_best

        if greater_keys is None:
            self.greater_keys = self._default_greater_keys
        else:
            if not isinstance(greater_keys, (list, tuple)):
                greater_keys = (greater_keys, )  # type: ignore
            assert is_seq_of(greater_keys, str)
            self.greater_keys = greater_keys  # type: ignore

        if less_keys is None:
            self.less_keys = self._default_less_keys
        else:
            if not isinstance(less_keys, (list, tuple)):
                less_keys = (less_keys, )  # type: ignore
            assert is_seq_of(less_keys, str)
            self.less_keys = less_keys  # type: ignore

        if self.save_best is not None:
            self.best_ckpt_path = None
            self._init_rule(rule, self.save_best)

    def before_train(self, runner) -> None:
        """Finish all operations, related to checkpoint.

        This function will get the appropriate file client, and the directory
        to save these checkpoints of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.out_dir is None:
            self.out_dir = runner.work_dir

        self.file_client = FileClient.infer_client(self.file_client_args,
                                                   self.out_dir)
        # if `self.out_dir` is not equal to `runner.work_dir`, it means that
        # `self.out_dir` is set so the final `self.out_dir` is the
        # concatenation of `self.out_dir` and the last level directory of
        # `runner.work_dir`
        if self.out_dir != runner.work_dir:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_client.join_path(
                self.out_dir, basename)  # type: ignore  # noqa: E501

        runner.logger.info(f'Checkpoints will be saved to {self.out_dir} by '
                           f'{self.file_client.name}.')

        if self.save_best is not None:
            if 'best_ckpt' not in runner.message_hub.runtime_info:
                self.best_ckpt_path = None
            else:
                self.best_ckpt_path = runner.message_hub.get_info('best_ckpt')

    def after_train_epoch(self, runner) -> None:
        """Save the checkpoint and synchronize buffers after each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        if not self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` epochs
        # 2. reach the last epoch of training
        if self.every_n_epochs(runner, self.interval) or (
                self.save_last and self.is_last_train_epoch(runner)):
            runner.logger.info(
                f'Saving checkpoint at {runner.epoch + 1} epochs')
            self._save_checkpoint(runner)

    def after_val_epoch(self, runner, metrics):
        if not self.by_epoch:
            return
        self._save_best_checkpoint(runner, metrics)

    def _get_metric_score(self, metrics):
        eval_res = OrderedDict()
        if metrics is not None:
            eval_res.update(metrics)

        if len(eval_res) == 0:
            warnings.warn(
                'Since `eval_res` is an empty dict, the behavior to save '
                'the best checkpoint will be skipped in this evaluation.')
            return None

        if self.key_indicator == 'auto':
            self._init_rule(self.rule, list(eval_res.keys())[0])

        return eval_res[self.key_indicator]

    @master_only
    def _save_checkpoint(self, runner) -> None:
        """Save the current checkpoint and delete outdated checkpoint.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.by_epoch:
            ckpt_filename = self.args.get(
                'filename_tmpl', 'epoch_{}.pth').format(runner.epoch + 1)
        else:
            ckpt_filename = self.args.get(
                'filename_tmpl', 'iter_{}.pth').format(runner.iter + 1)

        runner.save_checkpoint(
            self.out_dir,
            ckpt_filename,
            self.file_client_args,
            save_optimizer=self.save_optimizer,
            save_param_scheduler=self.save_param_scheduler,
            by_epoch=self.by_epoch,
            **self.args)

        runner.message_hub.update_info(
            'last_ckpt', self.file_client.join_path(self.out_dir,
                                                    ckpt_filename))

        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            if self.by_epoch:
                name = 'epoch_{}.pth'
                current_ckpt = runner.epoch + 1
            else:
                name = 'iter_{}.pth'
                current_ckpt = runner.iter + 1
            redundant_ckpts = range(
                current_ckpt - self.max_keep_ckpts * self.interval, 0,
                -self.interval)
            filename_tmpl = self.args.get('filename_tmpl', name)
            for _step in redundant_ckpts:
                ckpt_path = self.file_client.join_path(
                    self.out_dir, filename_tmpl.format(_step))
                if self.file_client.isfile(ckpt_path):
                    self.file_client.remove(ckpt_path)
                else:
                    break

    @master_only
    def _save_best_checkpoint(self, runner, metrics) -> None:
        """Save the current checkpoint and delete outdated checkpoint.

        Args:
            runner (Runner): The runner of the training process.
        """
        if not self.save_best:
            return

        if self.by_epoch:
            ckpt_filename = self.args.get(
                'filename_tmpl', 'epoch_{}.pth').format(runner.epoch + 1)
            cur_type, cur_time = 'epoch', runner.epoch + 1
        else:
            ckpt_filename = self.args.get(
                'filename_tmpl', 'iter_{}.pth').format(runner.iter + 1)
            cur_type, cur_time = 'iter', runner.iter + 1

        # save best logic
        # get score from messagehub
        # notice `_get_metirc_score` helps to infer
        # self.rule when self.save_best is `auto`
        key_score = self._get_metric_score(metrics)
        if 'best_score' not in runner.message_hub.runtime_info:
            best_score = self.init_value_map[self.rule]
        else:
            best_score = runner.message_hub.get_info('best_score')

        if not key_score or not self.is_better_than(key_score, best_score):
            return

        best_score = key_score
        runner.message_hub.update_info('best_score', best_score)

        if self.best_ckpt_path and self.file_client.isfile(
                self.best_ckpt_path):
            self.file_client.remove(self.best_ckpt_path)
            runner.logger.info(
                f'The previous best checkpoint {self.best_ckpt_path} '
                'is removed')

        best_ckpt_name = f'best_{self.key_indicator}_{ckpt_filename}'
        self.best_ckpt_path = self.file_client.join_path(  # type: ignore # noqa: E501
            self.out_dir, best_ckpt_name)
        runner.message_hub.update_info('best_ckpt', self.best_ckpt_path)
        runner.save_checkpoint(
            self.out_dir,
            filename=best_ckpt_name,
            file_client_args=self.file_client_args,
            save_optimizer=False,
            save_param_scheduler=False,
            by_epoch=False)
        runner.logger.info(
            f'The best checkpoint with {best_score:0.4f} {self.key_indicator} '
            f'at {cur_time} {cur_type} is saved to {best_ckpt_name}.')

    def _init_rule(self, rule, key_indicator) -> None:
        """Initialize rule, key_indicator, comparison_func, and best score.
        Here is the rule to determine which rule is used for key indicator when
        the rule is not specific (note that the key indicator matching is case-
        insensitive):

        1. If the key indicator is in ``self.greater_keys``, the rule will be
        specified as 'greater'.
        2. Or if the key indicator is in ``self.less_keys``, the rule will be
        specified as 'less'.
        3. Or if any one item in ``self.greater_keys`` is a substring of
            key_indicator , the rule will be specified as 'greater'.
        4. Or if any one item in ``self.less_keys`` is a substring of
            key_indicator , the rule will be specified as 'less'.
        Args:
            rule (str | None): Comparison rule for best score.
            key_indicator (str | None): Key indicator to determine the
                comparison rule.
        """

        if rule not in self.rule_map and rule is not None:
            raise KeyError('rule must be greater, less or None, '
                           f'but got {rule}.')

        if rule is None and key_indicator != 'auto':
            # `_lc` here means we use the lower case of keys for
            # case-insensitive matching
            key_indicator_lc = key_indicator.lower()
            greater_keys = [key.lower() for key in self.greater_keys]
            less_keys = [key.lower() for key in self.less_keys]

            if key_indicator_lc in greater_keys:
                rule = 'greater'
            elif key_indicator_lc in less_keys:
                rule = 'less'
            elif any(key in key_indicator_lc for key in greater_keys):
                rule = 'greater'
            elif any(key in key_indicator_lc for key in less_keys):
                rule = 'less'
            else:
                raise ValueError('Cannot infer the rule for key '
                                 f'{key_indicator}, thus a specific rule '
                                 'must be specified.')
        self.rule = rule
        self.key_indicator = key_indicator
        if self.rule is not None:
            self.is_better_than = self.rule_map[self.rule]

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs=Optional[dict]) -> None:
        """Save the checkpoint and synchronize buffers after each iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model. Defaults to None.
        """
        if self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training
        if self.every_n_train_iters(runner, self.interval) or \
                (self.save_last and
                 self.is_last_train_iter(runner)):
            runner.logger.info(
                f'Saving checkpoint at {runner.iter + 1} iterations')
            self._save_checkpoint(runner)
