# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import warnings
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch.nn as nn
from torch.distributed.fsdp import (FullStateDictConfig,
                                    FullyShardedDataParallel,
                                    LocalStateDictConfig,
                                    ShardedStateDictConfig, StateDictType)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig, LocalOptimStateDictConfig, OptimStateDictConfig,
    ShardedOptimStateDictConfig, StateDictConfig)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from mmengine.config import Config, ConfigDict
from mmengine.dist import get_local_rank, get_rank, is_main_process
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.optim import (AmpOptimWrapper, OptimWrapper, OptimWrapperDict,
                            _ParamScheduler, build_optim_wrapper)
from mmengine.registry import (MODEL_WRAPPERS, MODELS, OPTIM_WRAPPERS,
                               PARAM_SCHEDULERS, STRATEGIES, Registry)
from mmengine.utils import mkdir_or_exist
from .ddp_strategy import DDPStrategy
from .utils import MetaTensorContext, _load_state_dict_meta

FSDP = FullyShardedDataParallel
FSDP_CONFIGS = Registry('fsdp configs')
FSDP_CONFIGS.register_module(module=FullOptimStateDictConfig)
FSDP_CONFIGS.register_module(module=LocalOptimStateDictConfig)
FSDP_CONFIGS.register_module(module=FullStateDictConfig)
FSDP_CONFIGS.register_module(module=LocalStateDictConfig)
# Currently, SharededStateDictConfig and ShardedOptimStateDictConfig
# should not be used in FSDPStrategy
FSDP_CONFIGS.register_module(module=ShardedStateDictConfig)
FSDP_CONFIGS.register_module(module=ShardedOptimStateDictConfig)


class ReconstructOptimizerException(Exception):
    ...


@STRATEGIES.register_module()
class FSDPStrategy(DDPStrategy):

    def __init__(self,
                 *,
                 model_wrapper: Optional[dict] = None,
                 skip_init_weights=False,
                 state_dict_cfg: Union[str, dict, None] = 'local',
                 **kwargs):
        self._init_state_dict_cfg(state_dict_cfg)
        model_wrapper = model_wrapper or dict()
        if model_wrapper.get('use_orig_params', False):
            assert self.state_dict_type == StateDictType.FULL_STATE_DICT, (
                'use_orig_params only works with `FULL_STATE_DICT`, please '
                'check your state_dict_cfg')
        if not isinstance(skip_init_weights, bool):
            raise TypeError('skip_init_weights must be a boolean, but got '
                            f'{type(skip_init_weights)}')
        self.skip_init_weights = skip_init_weights
        self.model_wrapper = model_wrapper
        super().__init__(model_wrapper=model_wrapper, **kwargs)
        self.last_checkpoint = None

    def wrap_model(self, model) -> None:
        # model has been wrapped, do not re-wrap
        if is_model_wrapper(model):
            return

        assert model is not None, ('Model should have been built before wrap')

        self.model_wrapper.setdefault('type', 'MMFullyShardedDataParallel')
        self.model_wrapper.setdefault('module', model)
        self.model_wrapper.setdefault('device_id', get_local_rank())
        model = MODEL_WRAPPERS.build(self.model_wrapper)
        model.set_state_dict_type(model, self.state_dict_type,
                                  self.state_dict_config,
                                  self.optim_state_dict_config)
        return model

    def _is_full_state_dict(self):
        return self.state_dict_type == StateDictType.FULL_STATE_DICT

    def build_model(self, model: Union[nn.Module, dict]) -> nn.Module:
        if self.skip_init_weights:
            # Accelerate initialization by skipping init weights
            with MetaTensorContext():
                model = super().build_model(model)
        else:
            model = super().build_model(model)

        # `id_to_name` is used to convert the `optim_state_dict` of the
        # unsharded optimizer to the `optim_state_dict` of the sharded
        # optimizer which it returned by `FSDP.optim_state_dict` in
        # `StateDictType.FULL_STATE_DICT`
        self.id_to_name = dict()
        for name, param in model.named_parameters():
            self.id_to_name[id(param)] = name
        return model

    def save_checkpoint(self,
                        filename: str,
                        *,
                        save_optimizer: bool = True,
                        save_param_scheduler: bool = True,
                        extra_ckpt: Optional[dict] = None,
                        callback: Optional[Callable] = None) -> None:
        from mmengine.runner.checkpoint import save_checkpoint
        state_dict = self.state_dict(
            save_optimizer=save_optimizer,
            save_param_scheduler=save_param_scheduler)
        if self._is_full_state_dict():
            # save extra checkpoint passed by users
            if extra_ckpt is not None:
                state_dict.update(extra_ckpt)

            # users can do some modification before saving checkpoint
            if callback is not None:
                callback(state_dict)

        # In non-FULL_STATE_DICT model, FSDPStrategy will save checkpoint
        # of different ranks in different files.
        if not self._is_full_state_dict():
            rank = get_rank()
            mkdir_or_exist(filename)
            ckpt_name = f'rank{rank}.pth'
            filename = osp.join(filename, ckpt_name)
            save_checkpoint(state_dict, filename)

        if is_main_process():
            save_checkpoint(state_dict, filename)

    def model_state_dict(self) -> dict:
        # We've set state_dict by `FSDP.set_state_dict_type`, therefore we
        # should get model state dict by `FSDO.state_dict`
        return self.model.state_dict()

    def optim_state_dict(self) -> dict:
        return FSDP.optim_state_dict(self.model, self.optim_wrapper)

    def load_checkpoint(self, filename, *args, **kwargs) -> Optional[Dict]:
        # Avoid to call DDPStrategy.load_checkpoint
        if self._is_full_state_dict():
            return super(DDPStrategy,
                         self).load_checkpoint(filename, *args, **kwargs)
        else:
            rank = get_rank()
            filename = osp.join(filename, f'rank{rank}.pth')
            return super(DDPStrategy,
                         self).load_checkpoint(filename, *args, **kwargs)

    def load_model_state_dict(self,
                              state_dict: dict,
                              *,
                              strict: bool = False,
                              **kwargs) -> None:

        # We should load state dict by `FSDP.load_state_dict`
        assert 'state_dict' in state_dict
        state_dict = state_dict['state_dict']
        if not kwargs:
            self.logger.warn(f'{kwargs} will not be used when loading '
                             '`state_dict` in `FSDPStrategy`')
        if not self.skip_init_weights:
            self.model.load_state_dict(state_dict, strict=strict)
        else:
            # TODO: Use torchdistX to accelerate loading checkpoint
            _load_state_dict_meta(self.model, state_dict)

    def load_optim_state_dict(self, state_dict):
        assert 'optimizer' in state_dict
        opitm_state_dict = state_dict['optimizer']
        optim_state_dict = FSDP.optim_state_dict_to_load(
            opitm_state_dict, self.model, self.optim_wrapper.optimizer)
        self.optim_wrapper.load_state_dict(optim_state_dict)

    def _init_state_dict_cfg(self, state_dict_cfg):
        if isinstance(state_dict_cfg, str):
            if state_dict_cfg == 'full':
                self.state_dict_type = StateDictType.FULL_STATE_DICT
                self.state_dict_config = FullStateDictConfig(
                    rank0_only=True, offload_to_cpu=True)
                self.optim_state_dict_config = FullOptimStateDictConfig(
                    rank0_only=True, offload_to_cpu=True)
            elif state_dict_cfg == 'local':
                self.state_dict_type = StateDictType.LOCAL_STATE_DICT
                self.state_dict_config = LocalStateDictConfig()
                self.optim_state_dict_config = LocalOptimStateDictConfig()
            else:
                raise ValueError('FSDP only supports `full` and `local` '
                                 f'state_dict_type, but got {state_dict_cfg}')
        elif isinstance(state_dict_cfg, dict):
            if 'state_dict_type' not in state_dict_cfg:
                self.state_dict_type = StateDictType.LOCAL_STATE_DICT
            else:
                state_dict_type = state_dict_cfg['state_dict_type']
                if isinstance(state_dict_type, str):
                    self.state_dict_type = StateDictType[
                        state_dict_cfg['state_dict_type']]
                else:
                    self.state_dict_type = state_dict_type
            state_dict_config = state_dict_cfg.get('state_dict_config')
            if state_dict_config is None:
                self.state_dict_config = LocalStateDictConfig()
            elif isinstance(state_dict_config, dict):
                self.state_dict_config = FSDP_CONFIGS.build(
                    state_dict_cfg['state_dict_config'])
            else:
                self.state_dict_config = state_dict_config

            optim_state_dict_config = state_dict_cfg.get(
                'optim_state_dict_config')
            if optim_state_dict_config is None:
                self.optim_state_dict_config = LocalOptimStateDictConfig()
            elif isinstance(optim_state_dict_config, dict):
                self.optim_state_dict_config = FSDP_CONFIGS.build(
                    state_dict_cfg['optim_state_dict_config'])
            else:
                self.optim_state_dict_config = optim_state_dict_config
        else:
            raise TypeError('state_dict_cfg should be a `str` or a `dict`, '
                            f'but got {type(state_dict_cfg)}')

        if not isinstance(self.state_dict_type, StateDictType):
            raise TypeError('state_dict_type must be StateDictType, but got '
                            f'{type(self.state_dict_type)}')
        if not isinstance(self.state_dict_config, StateDictConfig):
            raise TypeError('state_dict_config must be StateDictConfig, but '
                            f'got {type(self.state_dict_config)}')
        if not isinstance(self.optim_state_dict_config, OptimStateDictConfig):
            raise TypeError('optim_state_dict_config must be '
                            'OptimStateDictConfig, but got '
                            f'{type(self.optim_state_dict_config)}')

    def convert_model(self, model: nn.Module) -> nn.Module:
        return model

    def build_optim_wrapper(
        self, optim_wrapper: Union[Optimizer, OptimWrapper, dict]
    ) -> Union[OptimWrapper, OptimWrapperDict]:
        """Shard the optimizer state_dict for the built optim_wrapper."""
        if isinstance(optim_wrapper, OptimWrapper):
            # NOTE: The only difference is that FSDPStrategy will shard
            # the the built OptimWrapper
            optimizer = optim_wrapper.optimizer
            param_groups = optimizer.param_groups
            optim_state_dict = optimizer.state_dict()
            assert not optim_state_dict['state'], (
                'Optimizer state_dict should be empty when giving an built '
                'optim_wrapper to FSDPStrategy')
            # Allign the state_dict with state_dict generated by
            # FSDP.full_optim_state_dict
            new_param_groups = []
            for group in param_groups:
                new_group = {
                    key: value
                    for key, value in group.items() if key != 'param'
                }
                new_group['params'] = [
                    self.id_to_name[id(param)] for param in group['params']
                ]
                new_param_groups.append(new_group)
            optim_state_dict['param_groups'] = new_param_groups
            defaults = {
                k: v
                for k, v in optimizer.defaults.items() if k != 'differentiable'
            }

            params_dict = {}
            for k, v in self.model.named_parameters():
                if '_fsdp_wrapped_module' in k:
                    k = k.replace('_fsdp_wrapped_module.', '')
                params_dict[k] = v

            params = []
            for param_group in new_param_groups:
                _params = []
                for param_name in param_group['params']:
                    if param_name not in params_dict:
                        raise RuntimeError(
                            'Failed to reconstruct the sharded optimizer. '
                            'You can try to set `use_orig_params=True` in '
                            '`model_wrapper`')
                    _params.append(params_dict[param_name])
                param_group = {
                    k: v
                    for k, v in param_group.items() if k != 'param'
                }
                param_group['params'] = _params
                params.append(param_group)

            new_optimizer = optimizer.__class__(params, **defaults)

            # Force to load the converted optim_state_dict in full mode.
            with FSDP.state_dict_type(self.model,
                                      StateDictType.FULL_STATE_DICT):
                optim_state_dict = FSDP.optim_state_dict_to_load(
                    optim_state_dict, self.model, new_optimizer)
                new_optimizer.load_state_dict(optim_state_dict)
            optim_wrapper.optimizer = new_optimizer
            return optim_wrapper
        if isinstance(optim_wrapper, (dict, ConfigDict, Config)):
            # optimizer must be defined for single optimizer training.
            optimizer = optim_wrapper.get('optimizer', None)
            optim_wrapper.setdefault('type', 'OptimWrapper')
            if optim_wrapper.get('type',
                                 'AmpOptimWrapper') in ('AmpOptimWrapper',
                                                        AmpOptimWrapper):
                optim_wrapper.setdefault('use_fsdp', True)

            # If optimizer is a built `Optimizer` instance, the optimizer
            # wrapper should be built by `OPTIM_WRAPPERS` registry.
            if isinstance(optimizer, Optimizer):
                return OPTIM_WRAPPERS.build(optim_wrapper)  # type: ignore

            # If `optimizer` is not None or `constructor` is defined, it means,
            # optimizer wrapper will be built by optimizer wrapper
            # constructor. Therefore, `build_optim_wrapper` should be called.
            if optimizer is not None or 'constructor' in optim_wrapper:
                return build_optim_wrapper(self.model, optim_wrapper)
            else:
                # if `optimizer` is not defined, it should be the case of
                # training with multiple optimizers. If `constructor` is not
                # defined either, each value of `optim_wrapper` must be an
                # `OptimWrapper` instance since `DefaultOptimizerConstructor`
                # will not handle the case of training with multiple
                # optimizers. `build_optim_wrapper` will directly build the
                # `OptimWrapperDict` instance from `optim_wrapper.`
                optim_wrappers = OrderedDict()
                for name, optim in optim_wrapper.items():
                    if not isinstance(optim, OptimWrapper):
                        raise ValueError(
                            'each item mush be an optimizer object when '
                            '"type" and "constructor" are not in '
                            f'optimizer, but got {name}={optim}')
                    optim_wrappers[name] = optim
                return OptimWrapperDict(**optim_wrappers)
        else:
            raise TypeError('optimizer wrapper should be an OptimWrapper '
                            f'object or dict, but got {optim_wrapper}')

    def _build_param_scheduler(self, scheduler: Union[_ParamScheduler, Dict,
                                                      List],
                               optim_wrapper: OptimWrapper,
                               default_args: dict) -> List[_ParamScheduler]:
        """Update the scheduler which is built with a unsharded optimzer."""
        if not isinstance(scheduler, Sequence):
            schedulers = [scheduler]
        else:
            schedulers = scheduler

        max_epochs = default_args.pop('max_epochs', None)
        max_iters = default_args.pop('max_iters', None)

        param_schedulers = []
        for scheduler in schedulers:
            # NOTE: Update the built scheduler with the sharded optimizer
            if isinstance(scheduler, _ParamScheduler):
                if type(scheduler).step is not _ParamScheduler.step:
                    warnings.warn(
                        'FSDPStrategy accept a built scheduler, and try to '
                        'update it with the sharded optimizer. However, '
                        f'{type(scheduler)}.step is overriden by user, and it '
                        'could lead to some unexpected error')
                scheduler._last_value = [
                    group[scheduler.param_name]
                    for group in scheduler.optimizer.param_groups
                ]
                param_schedulers.append(scheduler)
            elif isinstance(scheduler, LRScheduler):
                if type(scheduler).step is not LRScheduler.step:
                    warnings.warn(
                        'FSDPStrategy accept a built scheduler, and try to '
                        'update it with the sharded optimizer. However, '
                        f'{type(scheduler)}.step is overriden by user, and it '
                        'could lead to some unexpected error')
                self._last_lr = [
                    group['lr'] for group in scheduler.optimizer.param_groups
                ]
                param_schedulers.append(scheduler)
            elif isinstance(scheduler, dict):
                _scheduler = copy.deepcopy(scheduler)

                # Set default end
                if _scheduler.get('by_epoch', True):
                    if max_epochs is None:
                        raise ValueError(
                            'max_epochs must be specified in default_args')
                    default_end = max_epochs
                else:
                    if max_iters is None:
                        raise ValueError(
                            'max_iters must be specified in default_args')
                    default_end = max_iters
                _scheduler.setdefault('end', default_end)
                self.logger.debug(
                    f'The `end` of {_scheduler["type"]} is not set. '
                    'Use the max epochs/iters of train loop as default.')

                param_schedulers.append(
                    PARAM_SCHEDULERS.build(
                        _scheduler,
                        default_args=dict(
                            optimizer=optim_wrapper, **default_args)))
            else:
                raise TypeError(
                    'scheduler should be a _ParamScheduler object or dict, '
                    f'but got {scheduler}')
        return param_schedulers
