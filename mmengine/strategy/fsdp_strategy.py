# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import os.path as osp
from copy import deepcopy
from typing import Dict, Optional, Union

from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (LocalStateDictConfig,
                                    ShardedStateDictConfig, StateDictType)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig, LocalOptimStateDictConfig,
    ShardedOptimStateDictConfig)

from mmengine.dist import get_rank, is_main_process
from mmengine.model import is_model_wrapper
from mmengine.registry import MODEL_WRAPPERS, STRATEGIES, Registry
from mmengine.utils import mkdir_or_exist
from .ddp_strategy import DDPStrategy

FSDP_CONFIGS = Registry('fsdp configs')
FSDP_CONFIGS.register_module(module=FullOptimStateDictConfig)
FSDP_CONFIGS.register_module(module=ShardedOptimStateDictConfig)
FSDP_CONFIGS.register_module(module=LocalOptimStateDictConfig)
FSDP_CONFIGS.register_module(module=FullStateDictConfig)
FSDP_CONFIGS.register_module(module=ShardedStateDictConfig)
FSDP_CONFIGS.register_module(module=LocalStateDictConfig)


@STRATEGIES.register_module()
class FSDPStrategy(DDPStrategy):

    def __init__(self,
                 *,
                 logger: logging.Logger = None,
                 amp: Union[bool, Dict] = False,
                 accumulative_counts: int = 1,
                 clip_grad: Optional[Dict] = None,
                 state_dict_type: str = 'FULL_STATE_DICT',
                 state_dict_config: dict = dict(
                     type='FullStateDictConfig',
                     offload_to_cpu=True,
                     rank0_only=True),
                 optim_state_dict_config: dict = dict(
                     type='FullOptimStateDictConfig',
                     offload_to_cpu=True,
                     rank0_only=True),
                 **fsdp_kwargs):
        self.state_dict_type = StateDictType[state_dict_type]
        self.state_dict_config = FSDP_CONFIGS.build(state_dict_config)
        self.optim_state_dict_config = FSDP_CONFIGS.build(
            optim_state_dict_config)
        self.fsdp_kwargs = fsdp_kwargs
        super(DDPStrategy, self).__init__(
            logger=logger,
            amp=amp,
            accumulative_counts=accumulative_counts,
            clip_grad=clip_grad,
            **fsdp_kwargs)

    def _maybe_wrap_model(self) -> None:
        # model has been wrapped, do not re-wrap
        if is_model_wrapper(self.model):
            return

        assert self.model is not None, (
            'Model should have been built before wrap')

        self.fsdp_kwargs.setdefault('type', 'MMFullyShardedDataParallel')
        model_wrapper_cfg: dict = self.cfg.get('model_wrapper_cfg', dict())
        if self.fsdp_kwargs:
            assert not model_wrapper_cfg, (
                'fsdp_kwargs and model_wrapper_cfg cannot be configured at the same time'
            )
        else:
            self.fsdp_kwargs.update(model_wrapper_cfg)
        self.model = MODEL_WRAPPERS.build(self.fsdp_kwargs)
        self.model.set_state_dict_type(self.model, self.state_dict_type,
                                       self.state_dict_config,
                                       self.optim_state_dict_config)

    def _is_full_state_dict(self):
        return self.state_dict_type == StateDictType.FULL_STATE_DICT

    def save_checkpoint(self, out_dir: str, name: str, *args, **kwargs):
        # In non-FULL_STATE_DICT model, FSDPStrategy will save checkpoint
        # of different ranks in different files.
        if not self._is_full_state_dict():
            rank = get_rank()
            out_dir = osp.join(out_dir, name)
            mkdir_or_exist(out_dir)
            name = f'rank{rank}.pth'
        return super(DDPStrategy,
                     self).save_checkpoint(out_dir, name, *args, **kwargs)

    def _get_state_dict(self):
        # We've set state_dict by `FSDP.set_state_dict_type`, therefore we
        # should get model state dict by `FSDO.state_dict`
        return self.model.state_dict()

    def _get_optim_state_dict(self):
        if self._is_full_state_dict():
            # Same as `_get_state_dict`
            return FSDP.optim_state_dict(self.model, self.optim)
        else:
            return self.optim.state_dict()

    def _save_checkpoint_to_file(self, checkpoint, filepath):
        # In `FULL_STATE_DICT` mode, state dict will be gathered in rank0.
        # So we only need to save checkpoint in rank0.
        # In other modes, state dict is different across all ranks, therefore
        # we need to save checkpoint in all ranks.
        from mmengine.runner.checkpoint import save_checkpoint
        if self._is_full_state_dict():
            if is_main_process():
                save_checkpoint(checkpoint, filepath)
        else:
            save_checkpoint(checkpoint, filepath)

    def load_checkpoint(self, *args, **kwargs) -> Optional[Dict]:
        # Avoid to call DDPStrategy.load_checkpoint
        return super(DDPStrategy, self).load_checkpoint(*args, **kwargs)

    def _load_checkpoint_from_file(self, filepath, map_location, logger):
        from mmengine.runner.checkpoint import _load_checkpoint
        if self._is_full_state_dict():
            return _load_checkpoint(filepath, map_location, logger)
        else:
            rank = get_rank()
            filepath = osp.join(filepath, f'rank{rank}.pth')
            return _load_checkpoint(filepath, map_location, logger)

    def _load_checkpoint_to_model(self, checkpoint, strict):
        # We should load state dict by `FSDP.load_state_dict`
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict, strict=strict)
        return checkpoint

    def _resume_optim_wrapper(self, optim_state_dict):
        if self._is_full_state_dict():
            optim_state_dict = FSDP.shard_full_optim_state_dict(
                optim_state_dict, self.model)
            super()._resume_optim_wrapper(optim_state_dict)
        else:
            super()._resume_optim_wrapper(optim_state_dict)
