# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

from torch.optim import Optimizer

from mmengine.device import get_device
from mmengine.fileio import FileClient, join_path
from mmengine.model import revert_sync_batchnorm
from mmengine.optim import (DefaultOptimWrapperConstructor, OptimWrapper,
                            OptimWrapperDict, _ParamScheduler)
from mmengine.registry import (MODELS, OPTIM_WRAPPER_CONSTRUCTORS,
                               OPTIM_WRAPPERS, PARAM_SCHEDULERS, STRATEGIES)
# from mmengine.runner.checkpoint import (_load_checkpoint,
#                                         _load_checkpoint_to_model,
#                                         get_state_dict, save_checkpoint,
#                                         weights_to_cpu)
from .strategy import Mode, Strategy
from .utils import dfs_dict, inconsistent_keys


@STRATEGIES.register_module()
class NativeStrategy(Strategy):

    valid_model_wrappers: tuple = ()
    valid_optim_wrappers: tuple = ('OptimWraper', 'OptimWrapperDict',
                                   'AmpOptimWrapper', 'ApexOptimWrapper')
    valid_optim_constructors: tuple = ('DefaultOptimWrapperConstructor', )

    LAST_CKPT_SAVE_FILE: str = 'last_checkpoint'

    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        amp: Union[bool, Dict] = False,
        accumulative_counts: int = 1,
        clip_grad: Optional[Dict] = None,
    ):
        super().__init__(logger)
        assert isinstance(amp, (bool, dict))
        if isinstance(amp, dict):
            assert 'type' in amp
            assert amp['type'] in ('pytorch', 'apex')
        self.amp = amp
        self.accumulative_counts = accumulative_counts
        self.clip_grad = clip_grad

    def setup(
            self,
            model: Any,
            optim: Any = None,
            scheduler: Any = None,
            *,
            mode: Mode = Mode.TRAIN,
            cfg: Any = None,
            # below are for backward compatibility
            max_epochs: Optional[int] = None,
            epoch_length: Optional[int] = None,
            max_iters: Optional[int] = None,
            auto_scale_lr: Optional[Dict] = None):
        # Only build when necessaray
        if self.mode >= mode:
            self.logger.debug(
                'Trying to setup for {mode.name}, but {self.model.name} has '
                'been setup. Skip setup process.')
            return self.model, self.optim, self.scheduler
        self.mode = mode

        self.max_epochs = max_epochs
        self.epoch_length = epoch_length
        self.max_iters = max_iters
        self.auto_scale_lr = auto_scale_lr

        self._store_config_or_instance(model, optim, scheduler, cfg=cfg)
        assert self.model is not None or self.model_cfg is not None, (
            'Model instance or config must be provided to Strategy, got '
            f'{model}')

        self._maybe_build_model()

        self.model = self.model.to(get_device())
        # NativeStrategy should convert sync_bn to bn
        self.model = revert_sync_batchnorm(self.model)

        # Only build optimizer and param_schedulers in training
        if self.mode == Mode.TRAIN:
            self._maybe_build_optim()
            # Since auto_scale_lr must be called after building optim_wrapper
            # and before building param_schedulers, this must be called here
            self._maybe_scale_lr()
            self._maybe_build_scheduler()

        return self.model, self.optim, self.schedulers

    def setup_distributed(self, *args, **kwargs):
        pass

    def save_checkpoint(self,
                        out_dir: str,
                        name: str,
                        meta: Dict = None,
                        save_optimizer: bool = True,
                        save_param_scheduler: bool = True,
                        *,
                        file_client_args: Optional[Dict] = None,
                        backend_args: Optional[Dict] = None,
                        callback: Optional[Callable] = None) -> None:
        # Avoid circular import
        if meta is None:
            meta = {}
        assert isinstance(meta, dict)

        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set at '
                    'the same time.')

            file_client = FileClient.infer_client(file_client_args, out_dir)
            filepath = file_client.join_path(out_dir, name)
        else:
            filepath = join_path(  # type: ignore
                out_dir, name, backend_args=backend_args)

        checkpoint = {
            'state_dict': self._get_state_dict(),
        }
        # save optimizer state dict to checkpoint
        if save_optimizer:
            if isinstance(self.optim, OptimWrapper):
                checkpoint['optimizer'] = self._get_optim_state_dict()
            else:
                raise TypeError('self.optim should be an `OptimWrapper` '
                                'or `OptimWrapperDict` instance, but got '
                                f'{self.optim}')

        # save param scheduler state dict
        if save_param_scheduler and self.schedulers is None:
            self.logger.warn(
                '`save_param_scheduler` is True but `self.param_schedulers` '
                'is None, so skip saving parameter schedulers')
            save_param_scheduler = False
        if save_param_scheduler:
            if isinstance(self.schedulers, dict):
                checkpoint['param_schedulers'] = dict()
                for name, schedulers in self.param_schedulers.items():
                    checkpoint['param_schedulers'][name] = [
                        scheduler.state_dict() for scheduler in schedulers
                    ]
            else:
                checkpoint['param_schedulers'] = [
                    scheduler.state_dict() for scheduler in self.schedulers
                ]

        # check and save meta info
        common_keys = set(checkpoint.keys()) & set(meta.keys())
        assert len(common_keys) == 0
        checkpoint.update(meta)

        # Support before_save_checkpoint hook
        if callback is not None:
            callback(checkpoint)

        filepath = osp.join(out_dir, name)
        self._save_checkpoint_to_file(checkpoint, filepath)

    def _get_state_dict(self):
        from mmengine.runner.checkpoint import get_state_dict, weights_to_cpu
        return weights_to_cpu(get_state_dict(self.model))

    def _get_optim_state_dict(self):
        return self.optim.state_dict()

    def _save_checkpoint_to_file(self, checkpoint, filepath):
        from mmengine.runner.checkpoint import save_checkpoint
        save_checkpoint(checkpoint, filepath)

    def load_checkpoint(self,
                        load_dir: str,
                        filepath: Optional[str] = None,
                        load_optimizer: bool = False,
                        load_param_scheduler: bool = False,
                        *,
                        strict: bool = False,
                        map_location: Union[str, Callable] = 'cpu',
                        callback: Optional[Callable] = None) -> Optional[Dict]:
        # Avoid circular import
        if filepath is None:
            # should auto detect latest checkpoint
            filepath = self._latest_checkpoint_name(load_dir)
        if filepath is None:
            return None

        checkpoint = self._load_checkpoint_from_file(
            filepath, map_location=map_location, logger=self.logger)

        # Support after_load_checkpoint hook
        if callback is not None:
            callback(checkpoint)

        checkpoint = self._load_checkpoint_to_model(checkpoint, strict)

        if load_optimizer:
            assert self.optim is not None
            self._resume_optim_wrapper(checkpoint.get('optimizer'))

        if load_param_scheduler and self.schedulers is None:
            self.logger.warn(
                '`resume_param_scheduler` is True but `self.param_schedulers` '
                'is None, so skip resuming parameter schedulers')
            load_param_scheduler = False
        if 'param_schedulers' in checkpoint and load_param_scheduler:
            assert self.schedulers is not None
            ckpt_schedulers = checkpoint.get('param_schedulers')
            if isinstance(self.schedulers, dict):
                for name, schedulers in self.schedulers.items():
                    for scheduler, ckpt_scheduler in zip(
                            schedulers, ckpt_schedulers[name]):
                        scheduler.load_state_dict(ckpt_scheduler)
            else:
                for scheduler, ckpt_scheduler in zip(self.schedulers,
                                                     ckpt_schedulers):
                    scheduler.load_state_dict(ckpt_scheduler)

        return checkpoint

    def _load_checkpoint_from_file(self, filepath, map_location, logger):
        from mmengine.runner.checkpoint import _load_checkpoint
        return _load_checkpoint(filepath, map_location, logger)

    def _load_checkpoint_to_model(self, checkpoint, strict):
        from mmengine.runner.checkpoint import _load_checkpoint_to_model
        return _load_checkpoint_to_model(self.model, checkpoint, strict)

    def _resume_optim_wrapper(self, optim_state_dict):
        self.optim.load_state_dict(optim_state_dict)

    def _latest_checkpoint_name(self, ckpt_dir) -> Optional[str]:
        # Modified from mmengine.runner.checkpoint.find_latest_checkpoint
        save_file = osp.join(ckpt_dir, self.LAST_CKPT_SAVE_FILE)
        if not osp.exists(save_file):
            self.logger.info('Did not find latest checkpoint to be resumed.')
            return None
        with open(save_file) as f:
            ckpt_name: str = f.read().strip()
        self.logger.info(f'Detected the latest checkpoint: {ckpt_name}')
        return ckpt_name

    def _maybe_build_model(self) -> None:
        # no model config given, no need to build
        if self.model_cfg is None:
            return

        # optimizer instances may hide in `optim_wrapper_cfg`, find them out
        def _contain_optim(item):
            return isinstance(item, (Optimizer, OptimWrapper))

        contain_optim = dfs_dict(self.optim_cfg, _contain_optim, set())
        assert self.optim is None and not contain_optim, (
            'Optimizer cannot be built before Model')
        self.model = MODELS.build(self.model_cfg)

        # Only init_weight at trainning to avoid unnecessary runtime/download
        # cost. Refer to PR #367
        if self.mode == Mode.TRAIN and hasattr(self.model, 'init_weights'):
            self.model.init_weights()

        # build success, release configs
        self.model_cfg = None

    def _maybe_build_optim(self) -> None:
        # TODO: auto_scale_lr
        # no optim config given, no need to build
        if self.optim_cfg is None:
            return

        assert self.model is not None, (
            'Model must have been built before Optimizer')

        def _contain_scheduler(item):
            if isinstance(item, _ParamScheduler):
                return True
            if isinstance(item, (list, tuple)):
                return any(isinstance(x, _ParamScheduler) for x in item)
            return False

        contain_scheduler = dfs_dict(self.scheduler_cfg, _contain_scheduler,
                                     set())
        assert self.schedulers is None and not contain_scheduler, (
            'ParamScheduler cannot be built before Optimizer')

        wrapper_args = deepcopy(self.optim_cfg)
        constructor_type = wrapper_args.pop('constructor', None)
        is_builtin_constructor = (
            constructor_type is None
            or constructor_type == 'DefaultOptimWrapperConstructor')

        # Try to merge strategy_args into wrapper_args
        if not is_builtin_constructor:
            # We cannot infer what keys are used in optim_cfg in user-defined
            # OptimWrapperConstructor. In this case, we cannot check & update
            # wrapper_args with params in strategy. Just use raw config
            # and give warnings to users.
            default_values = dict(
                accumulative_counts=1, clip_grad=None, amp=False)
            ignored_configs = dict()
            for key, value in default_values.items():
                if getattr(self, key) != value:
                    ignored_configs[key] = getattr(self, key)
            if len(ignored_configs) > 0:
                self.logger.debug(
                    'Non-builtin OptimWrapperConstructor detected: '
                    f'{constructor_type}, the following params in '
                    f'DDPStrategy will be lost: {ignored_configs}')
        else:
            # Merge configs in strategy to optim_wrapper_cfg as long as they
            # are consistent
            strategy_args = dict(
                accumulative_counts=self.accumulative_counts,
                clip_grad=self.clip_grad)
            if isinstance(self.amp, dict):
                amp_configs = deepcopy(self.amp)
                type_mapping = dict(
                    pytorch='AmpOptimWrapper', apex='ApexOptimWrapper')
                amp_configs['type'] = type_mapping[amp_configs['type']]
            elif self.amp:
                # amp = True
                amp_configs = dict(type='AmpOptimWrapper')
            else:
                # amp = False, we cannot assume its `OptimWrapper` here
                # because it could be user-defined wrapper
                amp_configs = dict()
            strategy_args.update(amp_configs)
            keys = inconsistent_keys(strategy_args, wrapper_args)
            if len(keys) > 0:
                a = {k: wrapper_args[k] for k in keys}
                b = {k: strategy_args[k] for k in keys}
                # TODO: more friendly error message for amp related configs
                raise ValueError(
                    f'Inconsistent configurations: optim_wrapper_cfg = {a}, '
                    f'strategy = {b}')
            wrapper_args.update(strategy_args)

        # There are basically 4 cases in building optim_wrapper:
        #   1) `constructor` != None. Deliver everything to `constructor`,
        #      refer to `build_optim_wrapper`.
        #      NOTE: No optimizer instance should exist in this case.
        #   2) `constructor` == None, 'optimizer' not in optim_cfg. This is
        #      the case where all values in optim_cfg are OptimWrapper. All
        #      of them are composed to a single OptimWrapperDict.
        #      NOTE: has been handled in _store_instance_or_config
        #   3) `constructor` == None, optim_cfg.optimizer is Optimizer.
        #      Directly build optim_wrapper with OPTIM_WRAPPERS.build
        #   4) `constructor` == None, optim_cfg.optimizer is Config. Build
        #      optimizer according to DefaultOptimWrapperConstructor, then
        #      build optim_wrapper with OPTIM_WRAPPERS.build. This is part
        #      of the original `build_optim_wrapper` implementation.
        if constructor_type is not None:
            paramwise_cfg = wrapper_args.pop('paramwise_cfg', None)
            optim_wrapper_constructor = OPTIM_WRAPPER_CONSTRUCTORS.build(
                dict(
                    type=constructor_type,
                    optim_wrapper_cfg=wrapper_args,
                    paramwise_cfg=paramwise_cfg))
            self.optim = optim_wrapper_constructor(self.model)
        elif isinstance(wrapper_args['optimizer'], Optimizer):
            self.optim = OPTIM_WRAPPERS.build(wrapper_args)
        else:
            paramwise_cfg = wrapper_args.pop('paramwise_cfg', None)
            optim_wrapper_constructor = DefaultOptimWrapperConstructor(
                optim_wrapper_cfg=wrapper_args, paramwise_cfg=paramwise_cfg)
            self.optim = optim_wrapper_constructor(self.model)

        # build optim done, release configs
        self.optim_cfg = None

    def _maybe_build_scheduler(self) -> None:
        # no scheduler config given, no need to build
        if self.scheduler_cfg is None:
            return

        assert self.optim is not None, (
            'Optimizer must have been built before ParamScheduler')

        if isinstance(self.optim, OptimWrapperDict):
            self.schedulers = dict()
            is_seperate_schedulers = (
                isinstance(self.scheduler_cfg, dict)
                and self.scheduler_cfg.get('type') is None)
            for name, optim in self.optim.items():
                if is_seperate_schedulers:
                    self.schedulers[name] = self._build_param_scheduler(
                        self.scheduler_cfg.get(name), optim)
                else:
                    self.schedulers[name] = self._build_param_scheduler(
                        self.scheduler_cfg, optim)
        else:
            self.schedulers: list = self._build_param_scheduler(
                self.scheduler_cfg, self.optim)

        # build param_schedulers done, release configs
        self.scheduler_cfg = None

    def _build_param_scheduler(
            self, scheduler: Union[_ParamScheduler, Dict, List],
            optim_wrapper: OptimWrapper) -> List[_ParamScheduler]:
        """Build parameter schedulers for a single optimizer.

        Args:
            scheduler (dict or list): A dict or list of dict to build
                parameter schedulers.
            optim_wrapper (OptimWrapper): An optimizer wrapper object is
                passed to construct ParamScheduler object.

        Returns:
            list[_ParamScheduler]: List of parameter schedulers build from
            ``scheduler``.
        """
        if not isinstance(scheduler, (list, tuple)):
            schedulers = [scheduler]
        else:
            schedulers = scheduler

        # TODO: epoch_length and scheduler.end should be handled in Runner
        param_schedulers = []
        for scheduler in schedulers:
            if isinstance(scheduler, _ParamScheduler):
                param_schedulers.append(scheduler)
            elif isinstance(scheduler, dict):
                # Set default end, refer to PR #361
                # TO DEVELOPERS: Do **NOT** attemp to move these pieces of
                # codes outside of the build process, e.g. modify the config
                # files directly, since you can **NEVER** know how to parse the
                # param_scheduler unless OptimWrapper/OptimWrapperDict has been
                # built. I hate OPTIM_WRAPPER_CONSTRUCTORS
                by_epoch = scheduler.get('by_epoch', True)
                default_end = self.max_epochs if by_epoch else self.max_iters
                if default_end is not None:
                    scheduler.setdefault('end', default_end)
                    self.logger.debug(
                        f'The `end` of {scheduler["type"]} is not set. '
                        'Use the max epochs/iters of train loop as default.')

                _scheduler = PARAM_SCHEDULERS.build(
                    deepcopy(scheduler),
                    default_args=dict(
                        optimizer=optim_wrapper,
                        epoch_length=self.epoch_length))
                param_schedulers.append(_scheduler)
            else:
                raise TypeError(
                    'scheduler should be a _ParamScheduler object or dict, '
                    f'but got {scheduler}')
        return param_schedulers

    # Migrated from `runner.scale_lr` before, with slight modifications
    def _maybe_scale_lr(self) -> None:
        """Automatically scaling learning rate in training according to the
        ratio of ``base_batch_size`` in ``autoscalelr_cfg`` and real batch
        size.

        It scales the learning rate linearly according to the
        `paper <https://arxiv.org/abs/1706.02677>`_.

        Note:
            ``scale_lr`` must be called after building optimizer wrappers
            and before building parameter schedulers.
        """
        if (self.auto_scale_lr is None
                or not self.auto_scale_lr.get('enable', False)):
            return None

        assert 'base_batch_size' in self.auto_scale_lr, (
            'Lack of `base_batch_size` in `auto_scale_lr`.')
        assert 'real_bs' in self.auto_scale_lr, (
            'Lack of `real_bs` in `auto_scale_lr`. This should happen only '
            'when you are using Strategy alone. Please complete this dict '
            'in your code after you have built your dataloader to enable '
            'auto_scale_lr.')
        real_bs = self.auto_scale_lr['real_bs']
        base_bs = self.auto_scale_lr['base_batch_size']
        # TODO: should we consider `accumulative_counts` here?
        ratio = float(real_bs) / float(base_bs)
        self.logger.info(f'LR is set based on batch size of {base_bs} '
                         f'and the current batch size is {real_bs}. '
                         f'Scaling the original LR by {ratio}.')

        def _contain_scheduler(item):
            if isinstance(item, _ParamScheduler):
                return True
            if isinstance(item, (list, tuple)):
                return any(isinstance(x, _ParamScheduler) for x in item)
            return False

        contain_scheduler = dfs_dict(self.scheduler_cfg, _contain_scheduler,
                                     set())
        if self.schedulers is not None or contain_scheduler:
            raise RuntimeError('`scale_lr` should be called before building '
                               'ParamScheduler because ParamScheduler will '
                               'store initial lr from optimizer wrappers')

        assert isinstance(self.optim, OptimWrapper), \
            '`scale_lr should be called after building OptimWrapper'
        wrappers = list(self.optim.values()) if isinstance(
            self.optim, OptimWrapperDict) else [self.optim]
        for wrapper in wrappers:
            for group in wrapper.optimizer.param_groups:
                group['lr'] = group['lr'] * ratio
