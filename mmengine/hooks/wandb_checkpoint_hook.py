import os
import os.path as osp

import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
from mmengine.dist import is_main_process, master_only
from mmengine.hooks import CheckpointHook
from mmengine.logging import print_log
from mmengine.registry import HOOKS

try:
    import wandb
    from wandb.sdk.lib.paths import StrPath
except ImportError:
    raise ImportError('Please run "pip install wandb" to install wandb')


@master_only
def _log_artifact(filepath: StrPath,
                  aliases: Optional[List] = None,
                  metadata: Optional[Dict] = None):
    aliases = ["latest"] if aliases is None else aliases + ["latest"]
    metadata = wandb.run.config.as_dict() if metadata is None else metadata
    model_checkpoint_artifact = wandb.Artifact(
        f"run_{wandb.run.id}_model", type="model", metadata=metadata)
    if os.path.isfile(filepath):
        model_checkpoint_artifact.add_file(filepath)
    elif os.path.isdir(filepath):
        model_checkpoint_artifact.add_dir(filepath)
    else:
        raise FileNotFoundError(f"No such file or directory {filepath}")
    wandb.log_artifact(model_checkpoint_artifact, aliases=aliases or [])


@HOOKS.register_module()
class WandbCheckpointHook(CheckpointHook):

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
        runner.logger.info("HERE........_publish_model")
        _log_artifact(final_path, aliases=['published_model'])

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
        _log_artifact(
            osp.join(self.out_dir, self.filename_tmpl.format(step)),
            aliases=aliases,
            metadata=meta)

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
            _log_artifact(
                osp.join(self.out_dir, best_ckpt_name),
                aliases=[f"{key_indicator} best_score"],
                metadata=meta)

        # save checkpoint again to update the best_score and best_ckpt stored
        # in message_hub because the checkpoint saved in `after_train_epoch`
        # or `after_train_iter` stage only keep the previous best checkpoint
        # not the current best checkpoint which causes the current best
        # checkpoint can not be removed when resuming training.
        if best_ckpt_updated and self.last_ckpt is not None:
            self._save_checkpoint_with_step(
                runner, cur_time, meta, addition_aliases=["best_checkpoint"])
