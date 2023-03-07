# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from copy import deepcopy
from typing import Any, Dict, Optional, Union
import logging
from torch.optim import Optimizer

from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm,
                            is_model_wrapper)
from mmengine.logging import print_log
from mmengine.registry import MODEL_WRAPPERS, MODELS, OPTIM_WRAPPERS
from .strategy import ConfigType, Strategy


class DDPStrategy(Strategy):

    valid_model_wrappers: tuple = (
        'MMDistributedDataParallel', 'MMSeparateDistributedDataParallel'
    )
    valid_optim_wrappers: tuple = (
        'OptimWraper', 'AmpOptimWrapper', 'OptimWrapperDict',
        'ZeroRedundancyOptimizer'
    )

    def __init__(
            self,
            # OptimWrapper related kwargs
            amp: Union[bool, Dict] = False,
            accumulative_counts: int = 1,
            clip_grad: Optional[Dict] = None,
            # ModelWrapper related kwargs
            detect_anomalous_params: bool = False,
            # Passed to torch.DistributedDataParallel
            **ddp_kwargs):
        super().__init__()
        if isinstance(amp, dict):
            self._check_amp_config(amp)
        self.amp = amp
        self.accumulative_counts = accumulative_counts
        self.clip_grad = clip_grad
        self.detect_anomalous_params = detect_anomalous_params
        self.ddp_kwargs = ddp_kwargs

        # Compatible with existing use cases, where `model_wrapper_cfg` is None
        # and `find_unused_parameters` is set in `cfg`
        arg_name = 'find_unused_parameters'  # name too long, use var instead
        arg1 = self.cfg.get(arg_name, None)
        arg2 = self.ddp_kwargs.get(arg_name, None)
        if arg1 is not None or arg2 is not None:
            consistent = arg1 is None or arg2 is None or arg1 == arg2
            if not consistent:
                raise ValueError(
                    f'Inconsistent configuration: cfg.{arg_name} = {arg1}, '
                    f'strategy.{arg_name} = {arg2}')
            if arg2 is None:
                self.ddp_kwargs[arg_name] = arg1

    def setup(self,
              model: Any,
              optim: Any = None,
              scheduler: Any = None,
              cfg: Any = None):
        super().setup(model, optim=optim, scheduler=scheduler, cfg=cfg)
        assert self.model is not None or self.model_cfg is not None, (
            'A model must be provided to Strategy, got None')
        self._maybe_build_model()
        # TODO: Change ApexOptimWrapper logic to build it before wrap
        self._maybe_wrap_model()
        self._maybe_build_optim()
        self._maybe_build_scheduler()
        return self.model, self.optim, self.scheduler

    def setup_distributed(self, *args, **kwargs):
        pass

    def save_checkpoint(self, *args, **kwargs):
        pass

    def load_checkpoint(self, *args, **kwargs):
        pass

    def _check_amp_config(self, amp: dict) -> None:
        assert 'type' in amp
        assert amp['type'] in ('pytorch', 'apex')

    def _maybe_build_model(self) -> None:
        # model has been built, do not rebuild
        if self.model is not None:
            return
        # some built optimizers may hide in optim_wrapper_cfg, find them out
        def dfs_search_config(cfg: Any, memo: set):
            # be aware of loop config
            if cfg in memo:
                return False
            memo.add(cfg)
            if isinstance(cfg, ConfigType):
                return any(dfs_search_config(c, memo) for _, c in cfg.items())
            else:
                return isinstance(cfg, Optimizer)
        has_optimizer = dfs_search_config(self.optim_cfg, set())
        assert self.optim is None and not has_optimizer, (
                'Optimizer cannot be built before Model')
        self.model = MODELS.build(self.model_cfg)
        # DDP should use sync_bn if specified
        sync_bn = self.cfg.get('sync_bn', None)
        if sync_bn is not None:
            try:
                self.model = convert_sync_batchnorm(self.model, sync_bn)
            except ValueError as e:
                print_log('cfg.sync_bn should be "torch" or "mmcv", but got'
                          f'{sync_bn}',
                          logger='current',
                          level=logging.ERROR)
                raise e

    def _maybe_wrap_model(self) -> None:
        assert self.model is not None, 'Model should be built before wrap'
        # model has been wrapped, do not re-wrap
        if is_model_wrapper(self.model):
            return

        model_wrapper_cfg: dict = self.cfg.get('model_wrapper_cfg', dict())
        # use type given by user; otherwise use DDPStrategy's default
        wrapper_type = wrapper_args.pop('type', 'MMDistributedDataParallel')
        wrapper_cls = MODEL_WRAPPERS.get(wrapper_type)
        if wrapper_type in self.builtin_model_wrappers:
            # 1. MMEngine builtin model wrapper, check its validness in this
            # strategy, and update wrapper_args with strategy_args
            assert wrapper_type in self.valid_model_wrappers, (
                f'{self.__class__.__name__} only supports the following '
                f'model_wrappers: {self.valid_model_wrappers}, got invalid '
                f'type: {wrapper_type}')
            strategy_args = deepcopy(self.ddp_kwargs)
            # TODO: should non-builtin wrappers use `detect_anomalous_params`?
            strategy_args.update(
                {'detect_anomalous_params': self.detect_anomalous_params})
            # NOTE: This default value is potentially BC-breaking. In original
            # codes, when `broadcast_buffers` is not set, there are generally
            # 4 cases:
            #   1) model_wrapper_cfg = None: this defaults to False in `Runner`
            #   2) model_wrapper_cfg != None, type = MMSepDDP: this defaults
            #      to False in `MMSepDDP`
            #   3) model_wrapper_cfg != None, type = MMDDP: this defaults to
            #      True in `torch.DDP`
            #   4) model_wrapper_cfg != None, type not in builtin_wrapers:
            #      this defaults to user-defined default value
            # Currently this simple modification covers case (1) and (2).
            #
            # TODO(20230306, C1rN09): Further investigate whether there are
            # (3) or (4) use cases in downstream repos.
            #
            # TODO(20230306, C1rN09): Since this behavior is not consistent
            # with PyTorch, we should illustrate this in documentation.
            strategy_args.setdefault('broadcast_buffers', False)
            # Some arguments can both be set in strategy and model_wraper_cfg,
            # so we should check consistency here
            keys = self._find_inconsistency(wrapper_args, strategy_args)
            if len(keys) > 0:
                a = {k: wrapper_args[k] for k in keys}
                b = {k: strategy_args[k] for k in keys}
                raise ValueError(
                    f'Inconsistent configurations: model_wrapper_cfg = {a}, '
                    f'strategy = {b}')
            wrapper_args.update(strategy_args)
        else:
            # 2. user-defined custom wrappers, no check, args must all
            # be set in `model_wrapper_cfg` instead of `strategy`
            if len(self.ddp_kwargs) > 0:
                warnings.warn(
                    f'Detected non-builtin ModelWrapper: {wrapper_type} '
                    'in MMEngine. Following arguments in `strategy` will '
                    f'be dropped: {self.ddp_kwargs}')
        self.model = wrapper_cls(
            module=self.model,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            **wrapper_args)

    def _maybe_build_optim(self) -> None:
        # TODO: auto_scale_lr
        # optim has been built, do not rebuild
        if self.optim is not None:
            return
        assert self.model is not None, (
            'Model must have been built before Optimizer')
        assert self.scheduler is None, (
            'ParamScheduler cannot be built before Optimizer')
        if isinstance(self.optim, _ConfigType):
            # optimizer must be defined for single optimizer training.
            optimizer = self.optim.get('optimizer', None)

            # If optimizer is a built `Optimizer` instance, the optimizer
            # wrapper should be built by `OPTIM_WRAPPERS` registry.
            if isinstance(optimizer, Optimizer):
                self.optim.setdefault('type', 'OptimWrapper')
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

    def _maybe_build_scheduler(self) -> None:
        # scheduler has been built, do not rebuild
        if self.scheduler is not None:
            return
        assert self.optim is not None, (
            'Optimizer must have been built before ParamScheduler')

    @staticmethod
    def _find_inconsistency(x: ConfigType, y: ConfigType):
        common_keys = set(x.keys()) & set(y.keys())
        return {k for k in common_keys if x[k] != y[k]}
