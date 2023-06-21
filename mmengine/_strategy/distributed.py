# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Callable, Dict, List, Optional, Union

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from mmengine.device import get_device
from mmengine.dist import init_dist, is_distributed, master_only
from mmengine.model import convert_sync_batchnorm, is_model_wrapper
from mmengine.optim import BaseOptimWrapper, OptimWrapperDict, _ParamScheduler
from mmengine.registry import MODEL_WRAPPERS, STRATEGIES
from .single_device import SingleDeviceStrategy


@STRATEGIES.register_module()
class DDPStrategy(SingleDeviceStrategy):
    """Distribution strategy for distributed data parallel training.

    Args:
        model_wrapper (dict): Dict for model wrapper. Defaults to None.
        auto_scale_lr (dict, Optional): Config to scale the learning rate
            automatically. It includes ``base_batch_size`` and ``enable``.
            ``base_batch_size`` is the batch size that the optimizer lr is
            based on. ``enable`` is the switch to turn on and off the feature.
        sync_bn (str): Type of sync batch norm. Defaults to None.
            Options are 'torch' and 'mmcv'.
        **kwargs: Other arguments for :class:`BaseStrategy`.
    """

    def __init__(
        self,
        *,
        model_wrapper: Optional[dict] = None,
        auto_scale_lr: Optional[dict] = None,
        sync_bn: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if model_wrapper is None:
            # set broadcast_buffers as False to keep compatibility with
            # OpenMMLab repos
            model_wrapper = dict(
                type='MMDistributedDataParallel', broadcast_buffers=False)
        self.model_wrapper = model_wrapper
        self.auto_scale_lr = auto_scale_lr
        self.sync_bn = sync_bn

    def prepare(
        self,
        model: Union[nn.Module, dict],
        *,
        optim_wrapper: Union[BaseOptimWrapper, dict, None] = None,
        param_scheduler: Union[_ParamScheduler, Dict, List, None] = None,
        compile: Union[dict, bool] = False,
        dispatch_kwargs: Optional[dict] = None,
    ):
        """Prepare model and some components.

        Args:
            model (:obj:`torch.nn.Module` or dict): The model to be run. It
                can be a dict used for building a model.

        Kwargs:
            optim_wrapper (BaseOptimWrapper or dict, optional): Computing the
                gradient of model parameters and updating them.
                Defaults to None.
                See :meth:`build_optim_wrapper` for examples.
            param_scheduler (_ParamScheduler or dict or list, optional):
                Parameter scheduler for updating optimizer parameters. If
                specified, :attr:`optim_wrapper` should also be specified.
                Defaults to None.
                See :meth:`build_param_scheduler` for examples.
            compile (dict, optional): Config to compile model.
                Defaults to False. Requires PyTorch>=2.0.
            dispatch_kwargs (dict, optional): Kwargs to be passed to other
                methods of Strategy. Defaults to None.
        """
        if dispatch_kwargs is not None:
            self.dispatch_kwargs.update(dispatch_kwargs)

        return_items = []

        model = self.build_model(model)
        model = self._init_model_weights(model)
        model = self._wrap_model(model)
        model = self.compile_model(model, compile=compile)
        return_items.append(model)

        self.model = model

        if optim_wrapper is not None:
            self.optim_wrapper = self.build_optim_wrapper(optim_wrapper, model)
            return_items.append(self.optim_wrapper)

        if param_scheduler is not None:
            self.param_schedulers = self.build_param_scheduler(
                param_scheduler, self.optim_wrapper)
            return_items.append(self.param_schedulers)

        if optim_wrapper is not None:
            self._scale_lr()

        return return_items[0] if len(return_items) == 1 else return_items

    def setup_distributed(  # type: ignore
        self,
        launcher: str = 'pytorch',
        backend: str = 'nccl',
        **kwargs,
    ):
        """Setup distributed environment.

        Args:
            launcher (str): Way to launcher multi processes. Supported
                launchers are 'pytorch', 'mpi' and 'slurm'.
            backend (str): Communication Backends. Supported backends are
                'nccl', 'gloo' and 'mpi'. Defaults to 'nccl'.
            **kwargs: Other arguments for :func:`init_dist`.
        """
        if not is_distributed():
            init_dist(launcher, backend, **kwargs)

    def _scale_lr(self) -> None:
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

        assert 'base_batch_size' in self.auto_scale_lr, \
            'Lack of `base_batch_size` in `auto_scale_lr`.'

        real_bs = self.world_size * self.dispatch_kwargs[
            'train_micro_batch_size_per_gpu']
        base_bs = self.auto_scale_lr['base_batch_size']
        ratio = float(real_bs) / float(base_bs)
        self.logger.info(f'LR is set based on batch size of {base_bs} '
                         f'and the current batch size is {real_bs}. '
                         f'Scaling the original LR by {ratio}.')

        def _is_built(schedulers):
            if isinstance(schedulers, dict):
                return False if 'type' in schedulers else any(
                    _is_built(s) for s in schedulers.values())
            if isinstance(schedulers, list):
                return any(_is_built(s) for s in schedulers)
            return isinstance(schedulers, _ParamScheduler)

        if _is_built(self.param_schedulers):
            raise RuntimeError('`scale_lr` should be called before building '
                               'ParamScheduler because ParamScheduler will '
                               'store initial lr from optimizer wrappers')

        assert isinstance(self.optim_wrapper, BaseOptimWrapper), \
            '`scale_lr should be called after building OptimWrapper'

        if isinstance(self.optim_wrapper, OptimWrapperDict):
            wrappers = list(self.optim_wrapper.values())
        else:
            wrappers = [self.optim_wrapper]  # type: ignore

        for wrapper in wrappers:
            for group in wrapper.optimizer.param_groups:
                group['lr'] = group['lr'] * ratio

    def convert_model(self, model: nn.Module) -> nn.Module:
        """convert all ``BatchNorm`` layers in the model to ``SyncBatchNorm``
        (SyncBN) or ``mmcv.ops.sync_bn.SyncBatchNorm`` (MMSyncBN) layers.

        Args:
            model (nn.Module): Model to be converted.

        Returns:
            nn.Module: Converted model.
        """
        if self.sync_bn is not None:
            try:
                model = convert_sync_batchnorm(model, self.sync_bn)
            except ValueError as e:
                self.logger.error('cfg.sync_bn should be "torch" or '
                                  f'"mmcv", but got {self.sync_bn}')
                raise e

        return model

    def _wrap_model(self, model: nn.Module) -> DistributedDataParallel:
        """Wrap the model to :obj:``MMDistributedDataParallel`` or other custom
        distributed data-parallel module wrappers.

        Args:
            model (nn.Module): Model to be wrapped.

        Returns:
            nn.Module or DistributedDataParallel: nn.Module or subclass of
            ``DistributedDataParallel``.
        """
        if is_model_wrapper(model):
            return model

        model = model.to(get_device())

        model = self.convert_model(model)

        default_args = dict(module=model)
        default_args.setdefault('device_ids', [int(os.environ['LOCAL_RANK'])])
        model = MODEL_WRAPPERS.build(
            self.model_wrapper, default_args=default_args)
        return model

    @master_only
    def save_checkpoint(
        self,
        filename: str,
        *,
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
        extra_ckpt: Optional[dict] = None,
        callback: Optional[Callable] = None,
    ) -> None:
        super().save_checkpoint(
            filename=filename,
            save_optimizer=save_optimizer,
            save_param_scheduler=save_param_scheduler,
            extra_ckpt=extra_ckpt,
            callback=callback)
