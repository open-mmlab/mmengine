# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import os.path as osp
import pickle
from math import inf
from pathlib import Path
from typing import List, Optional, Sequence, Union

from mmengine.dist import is_main_process, master_only
from mmengine.hooks import CheckpointHook
from mmengine.logging import print_log
from mmengine.registry import HOOKS

try:
    import wandb
except ImportError:
    raise ImportError('Please run "pip install wandb" to install wandb')


@HOOKS.register_module()
class WandbCheckpointHook(CheckpointHook):
    """Save checkpoints periodically as [W&B Models
    Artifact](https://docs.wandb.ai/guides/model_registry/log-model-to-
    experiment).

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Defaults to -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Defaults to True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Defaults to True.
        save_param_scheduler (bool): Whether to save param_scheduler state_dict
            in the checkpoint. It is usually used for resuming experiments.
            Defaults to True.
        out_dir (str, Path, Optional): The root directory to save checkpoints.
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
        save_best (str, List[str], optional): If a metric is specified, it
            would measure the best checkpoint during evaluation. If a list of
            metrics is passed, it would measure a group of best checkpoints
            corresponding to the passed metrics. The information about best
            checkpoint(s) would be saved in ``runner.message_hub`` to keep
            best score value and best checkpoint path, which will be also
            loaded when resuming checkpoint. Options are the evaluation metrics
            on the test dataset. e.g., ``bbox_mAP``, ``segm_mAP`` for bbox
            detection and instance segmentation. ``AR@100`` for proposal
            recall. If ``save_best`` is ``auto``, the first key of the returned
            ``OrderedDict`` result will be used. Defaults to None.
        rule (str, List[str], optional): Comparison rule for best score. If
            set to None, it will infer a reasonable rule. Keys such as 'acc',
            'top' .etc will be inferred by 'greater' rule. Keys contain 'loss'
            will be inferred by 'less' rule. If ``save_best`` is a list of
            metrics and ``rule`` is a str, all metrics in ``save_best`` will
            share the comparison rule. If ``save_best`` and ``rule`` are both
            lists, their length must be the same, and metrics in ``save_best``
            will use the corresponding comparison rule in ``rule``. Options
            are 'greater', 'less', None and list which contains 'greater' and
            'less'. Defaults to None.
        greater_keys (List[str], optional): Metric keys that will be
            inferred by 'greater' comparison rule. If ``None``,
            _default_greater_keys will be used. Defaults to None.
        less_keys (List[str], optional): Metric keys that will be
            inferred by 'less' comparison rule. If ``None``, _default_less_keys
            will be used. Defaults to None.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
        filename_tmpl (str, optional): String template to indicate checkpoint
            name. If specified, must contain one and only one "{}", which will
            be replaced with ``epoch + 1`` if ``by_epoch=True`` else
            ``iteration + 1``.
            Defaults to None, which means "epoch_{}.pth" or "iter_{}.pth"
            accordingly.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.
            `New in version 0.2.0.`
        published_keys (str, List[str], optional): If ``save_last`` is ``True``
            or ``save_best`` is not ``None``, it will automatically
            publish model with keys in the list after training.
            Defaults to None.
            `New in version 0.7.1.`
        save_begin (int): Control the epoch number or iteration number
            at which checkpoint saving begins. Defaults to 0, which means
            saving at the beginning.
            `New in version 0.8.3.`
        model_name (str, optional): A name assigned to the model artifact
            that the model checkpoint files will be added to. The string must
            contain only the following alphanumeric characters: dashes,
            underscores, and dots. This will default to
            ``f'model-run-{wandb.run.id}'`` if left unspecified.

    Examples:
        >>> # Save best based on single metric
        >>> WandbCheckpointHook(interval=2, by_epoch=True, save_best='acc',
        >>>                rule='less')
        >>> # Save best based on multi metrics with the same comparison rule
        >>> WandbCheckpointHook(interval=2, by_epoch=True,
        >>>                save_best=['acc', 'mIoU'], rule='greater')
        >>> # Save best based on multi metrics with different comparison rule
        >>> WandbCheckpointHook(interval=2, by_epoch=True,
        >>>                save_best=['FID', 'IS'], rule=['less', 'greater'])
        >>> # Save best based on single metric and publish model after training
        >>> WandbCheckpointHook(interval=2, by_epoch=True, save_best='acc',
        >>>                rule='less', published_keys=['meta', 'state_dict'])
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
                 init_kwargs: Optional[dict] = None,
                 interval: int = -1,
                 by_epoch: bool = True,
                 save_optimizer: bool = True,
                 save_param_scheduler: bool = True,
                 out_dir: Optional[Union[str, Path]] = None,
                 max_keep_ckpts: int = -1,
                 save_last: bool = True,
                 save_best: Union[str, List[str], None] = None,
                 rule: Union[str, List[str], None] = None,
                 greater_keys: Optional[Sequence[str]] = None,
                 less_keys: Optional[Sequence[str]] = None,
                 file_client_args: Optional[dict] = None,
                 filename_tmpl: Optional[str] = None,
                 backend_args: Optional[dict] = None,
                 published_keys: Union[str, List[str], None] = None,
                 save_begin: int = 0,
                 model_name: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(interval, by_epoch, save_optimizer,
                         save_param_scheduler, out_dir, max_keep_ckpts,
                         save_last, save_best, rule, greater_keys, less_keys,
                         file_client_args, filename_tmpl, backend_args,
                         published_keys, save_begin, **kwargs)
        self.init_kwargs = init_kwargs or {}
        self._wandb = wandb
        if self._wandb.run is None:
            self._wandb.init(**self.init_kwargs)
        default_model_name = f'model-run-{wandb.run.id}'
        self.model_name = model_name if model_name else default_model_name

    @master_only
    def _publish_model(self, runner, ckpt_path: str) -> None:
        from mmengine.runner import save_checkpoint
        from mmengine.runner.checkpoint import _load_checkpoint
        checkpoint = _load_checkpoint(ckpt_path)
        assert self.published_keys is not None
        removed_keys = []
        for key in list(checkpoint.keys()):
            if key not in self.published_keys:
                removed_keys.append(key)
                checkpoint.pop(key)
        if removed_keys:
            print_log(
                f'Key {removed_keys} will be removed because they are not '
                'found in published_keys. If you want to keep them, '
                f'please set `{removed_keys}` in published_keys',
                logger='current')
        checkpoint_data = pickle.dumps(checkpoint)
        sha = hashlib.sha256(checkpoint_data).hexdigest()
        final_path = osp.splitext(ckpt_path)[0] + f'-{sha[:8]}.pth'
        save_checkpoint(checkpoint, final_path)
        print_log(
            f'The checkpoint ({ckpt_path}) is published to '
            f'{final_path}.',
            logger='current')
        runner.logger.info('HERE........_publish_model')

        wandb.log_model(
            final_path, name=self.model_name, aliases=['published_model'])

    def _save_checkpoint_with_step(
            self,
            runner,
            step,
            meta,
            addition_aliases: Optional[List[str]] = None):
        super()._save_checkpoint_with_step(runner, step, meta)
        aliases = [f"epoch {meta['epoch']}", f"iteration {meta['iter']}"]
        if addition_aliases:
            aliases += addition_aliases
        wandb.log_model(
            osp.join(self.out_dir, self.filename_tmpl.format(step)),
            name=self.model_name,
            aliases=aliases)

    def _save_best_checkpoint(self, runner, metrics) -> None:
        if not self.save_best:
            return

        if self.by_epoch:
            ckpt_filename = self.filename_tmpl.format(runner.epoch)
            cur_type, cur_time = 'epoch', runner.epoch
        else:
            ckpt_filename = self.filename_tmpl.format(runner.iter)
            cur_type, cur_time = 'iter', runner.iter

        meta = dict(epoch=runner.epoch, iter=runner.iter)

        # handle auto in self.key_indicators and self.rules before the loop
        if 'auto' in self.key_indicators:
            self._init_rule(self.rules, [list(metrics.keys())[0]])

        best_ckpt_updated = False
        # save best logic
        # get score from messagehub
        for key_indicator, rule in zip(self.key_indicators, self.rules):
            key_score = metrics[key_indicator]

            if len(self.key_indicators) == 1:
                best_score_key = 'best_score'
                runtime_best_ckpt_key = 'best_ckpt'
                best_ckpt_path = self.best_ckpt_path
            else:
                best_score_key = f'best_score_{key_indicator}'
                runtime_best_ckpt_key = f'best_ckpt_{key_indicator}'
                best_ckpt_path = self.best_ckpt_path_dict[key_indicator]

            if best_score_key not in runner.message_hub.runtime_info:
                best_score = self.init_value_map[rule]
            else:
                best_score = runner.message_hub.get_info(best_score_key)

            if key_score is None or not self.is_better_than[key_indicator](
                    key_score, best_score):
                continue

            best_ckpt_updated = True

            best_score = key_score
            runner.message_hub.update_info(best_score_key, best_score)

            if best_ckpt_path and is_main_process():
                is_removed = False
                if self.file_backend.isfile(best_ckpt_path):
                    self.file_backend.remove(best_ckpt_path)
                    is_removed = True
                elif self.file_backend.isdir(best_ckpt_path):
                    # checkpoints saved by deepspeed are directories
                    self.file_backend.rmtree(best_ckpt_path)
                    is_removed = True

                if is_removed:
                    runner.logger.info(
                        f'The previous best checkpoint {best_ckpt_path} '
                        'is removed')

            best_ckpt_name = f'best_{key_indicator}_{ckpt_filename}'
            # Replace illegal characters for filename with `_`
            best_ckpt_name = best_ckpt_name.replace('/', '_')
            if len(self.key_indicators) == 1:
                self.best_ckpt_path = self.file_backend.join_path(  # type: ignore # noqa: E501
                    self.out_dir, best_ckpt_name)
                runner.message_hub.update_info(runtime_best_ckpt_key,
                                               self.best_ckpt_path)
            else:
                self.best_ckpt_path_dict[
                    key_indicator] = self.file_backend.join_path(  # type: ignore # noqa: E501
                        self.out_dir, best_ckpt_name)
                runner.message_hub.update_info(
                    runtime_best_ckpt_key,
                    self.best_ckpt_path_dict[key_indicator])
            runner.save_checkpoint(
                self.out_dir,
                filename=best_ckpt_name,
                file_client_args=self.file_client_args,
                save_optimizer=False,
                save_param_scheduler=False,
                meta=meta,
                by_epoch=False,
                backend_args=self.backend_args)
            runner.logger.info(
                f'The best checkpoint with {best_score:0.4f} {key_indicator} '
                f'at {cur_time} {cur_type} is saved to {best_ckpt_name}.')
            wandb.log_model(
                osp.join(self.out_dir, best_ckpt_name),
                name=self.model_name,
                aliases=[f'{key_indicator} best_score'])

        # save checkpoint again to update the best_score and best_ckpt stored
        # in message_hub because the checkpoint saved in `after_train_epoch`
        # or `after_train_iter` stage only keep the previous best checkpoint
        # not the current best checkpoint which causes the current best
        # checkpoint can not be removed when resuming training.
        if best_ckpt_updated and self.last_ckpt is not None:
            self._save_checkpoint_with_step(
                runner, cur_time, meta, addition_aliases=['best_checkpoint'])
