# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from copy import deepcopy
from typing import Any, Dict, Optional, Union

from mmengine.device import get_device
from mmengine.dist import master_only
from mmengine.dist.utils import init_dist
from mmengine.model import convert_sync_batchnorm, is_model_wrapper
from mmengine.registry import MODEL_WRAPPERS, STRATEGIES
from .native_strategy import NativeStrategy
from .strategy import Mode
from .utils import inconsistent_keys


@STRATEGIES.register_module()
class DDPStrategy(NativeStrategy):

    valid_model_wrappers: tuple = ('MMDistributedDataParallel',
                                   'MMSeparateDistributedDataParallel')
    valid_optim_wrappers: tuple = ('OptimWraper', 'OptimWrapperDict',
                                   'AmpOptimWrapper', 'ApexOptimWrapper')
    valid_optim_constructors: tuple = ('DefaultOptimWrapperConstructor', )

    def __init__(self,
                 *,
                 logger: logging.Logger = None,
                 amp: Union[bool, Dict] = False,
                 accumulative_counts: int = 1,
                 clip_grad: Optional[Dict] = None,
                 detect_anomalous_params: bool = False,
                 **ddp_kwargs):
        super().__init__(
            logger=logger,
            amp=amp,
            accumulative_counts=accumulative_counts,
            clip_grad=clip_grad)
        self.detect_anomalous_params = detect_anomalous_params
        self.ddp_kwargs = ddp_kwargs

        # Compat with existing use cases, where `model_wrapper_cfg` is None
        # and `find_unused_parameters` is set in `cfg`
        arg1 = self.cfg.get('find_unused_parameters', None)
        arg2 = self.ddp_kwargs.get('find_unused_parameters', None)
        if arg1 is not None or arg2 is not None:
            consistent = arg1 is None or arg2 is None or arg1 == arg2
            if not consistent:
                raise ValueError(f'Inconsistent configuration: '
                                 f'cfg.find_unused_parameters = {arg1}, '
                                 f'strategy.find_unused_parameters = {arg2}')
            if arg2 is None:
                self.ddp_kwargs['find_unused_parameters'] = arg1

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
        # DDP should use sync_bn if specified. Currently read from `cfg`
        # for backward compatibility
        sync_bn = self.cfg.get('sync_bn', None)
        if sync_bn is not None:
            try:
                self.model = convert_sync_batchnorm(self.model, sync_bn)
            except ValueError as e:
                self.logger.error(
                    'cfg.sync_bn should be "torch" or "mmcv", but got'
                    f'{sync_bn}')
                raise e

        # TODO: Change ApexOptimWrapper logic to build it before wrap
        self._maybe_wrap_model()

        # Only build optimizer and param_schedulers in training
        if self.mode == Mode.TRAIN:
            self._maybe_build_optim()
            # Since auto_scale_lr must be called after building optim_wrapper
            # and before building param_schedulers, this must be called here
            self._maybe_scale_lr()
            self._maybe_build_scheduler()

        return self.model, self.optim, self.schedulers

    def setup_distributed(self, launcher, backend='nccl', **kwargs):
        init_dist(launcher, backend, **kwargs)

    @master_only
    def save_checkpoint(self, *args, **kwargs):
        ddp_model, self.model = self.model, self.model.module
        super().save_checkpoint(*args, **kwargs)
        self.model = ddp_model

    def load_checkpoint(self, *args, **kwargs) -> Optional[Dict]:
        ddp_model, self.model = self.model, self.model.module
        checkpoint = super().load_checkpoint(*args, **kwargs)
        self.model = ddp_model
        return checkpoint

    def _maybe_wrap_model(self) -> None:
        # model has been wrapped, do not re-wrap
        if is_model_wrapper(self.model):
            return

        assert self.model is not None, (
            'Model should have been built before wrap')

        model_wrapper_cfg: dict = self.cfg.get('model_wrapper_cfg', dict())
        wrapper_args = deepcopy(model_wrapper_cfg)
        # use `type` declared by user; otherwise use DDPStrategy's default
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
            keys = inconsistent_keys(wrapper_args, strategy_args)
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
                self.logger.debug(
                    f'Detected non-builtin ModelWrapper: {wrapper_type} '
                    'in MMEngine. Following arguments in `strategy` will '
                    f'be dropped: {self.ddp_kwargs}')

        self.model = wrapper_cls(
            module=self.model,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            **wrapper_args)
