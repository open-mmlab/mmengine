# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp
import time
from typing import Callable, Dict, List, Optional, Union

import deepspeed
import torch
import torch.nn as nn

import mmengine
from mmengine.model.wrappers._deepspeed import MMDeepSpeedEngineWrapper
from mmengine.optim import OptimWrapper, _ParamScheduler
from mmengine.registry import STRATEGIES
from mmengine.utils import get_git_hash
from .base import BaseStrategy


@STRATEGIES.register_module()
class DeepSpeedStrategy(BaseStrategy):
    """

    Args:
        zero_optimization (dict, optional):
        fp16 (dict, optional):
    """
    dispatch_keys = [
        'train_batch_size', 'num_batches_per_epoch', 'max_epochs', 'max_iters'
    ]

    def __init__(
        self,
        *,
        # the following args are for deepspeed
        config: Union[str, dict, None] = None,
        zero_optimization: Optional[dict] = None,
        fp16: Optional[dict] = None,
        inputs_to_half: Optional[List[Union[int, str]]] = None,
        bf16: Optional[dict] = None,
        amp: Optional[dict] = None,
        activation_checkpointing: Optional[dict] = None,
        aio: Optional[dict] = None,
        # disable the log printed by deepseed
        steps_per_print: int = 10000000000000,
        # the following args are for BaseStrategy
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.config = self._parse_config(config)
        if zero_optimization is not None:
            self.config['zero_optimization'] = zero_optimization
        if fp16 is not None:
            self.config['fp16'] = fp16
        if bf16 is not None:
            self.config['bf16'] = bf16
        if amp is not None:
            self.config['amp'] = amp
        if activation_checkpointing is not None:
            self.config['activation_checkpointing'] = activation_checkpointing
        if aio is not None:
            self.config['aio'] = aio

        self.config['steps_per_print'] = steps_per_print

        self._inputs_to_half = inputs_to_half

    def _parse_config(self, config):
        if config is None:
            config = dict()
        elif isinstance(config, str):
            with open(config) as f:
                config = json.load(f)
        return config

    def setup_distributed(self, launcher=None, backend='nccl', **kwargs):
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        deepspeed.init_distributed(dist_backend=backend)

    def prepare(
        self,
        model: Union[nn.Module, dict],
        *,
        optim_wrapper: Optional[Union[OptimWrapper, dict]] = None,
        param_scheduler: Optional[Union[_ParamScheduler, Dict, List]] = None,
        dispatch_kwargs: Optional[dict] = None,
    ):
        """Prepare model and some components.

        Args:
            model (:obj:`torch.nn.Module` or dict): The model to be run. It
                can be a dict used for build a model.

        Kwargs:
            optim_wrapper (OptimWrapper or dict, optional):
                Computing gradient of model parameters. If specified,
                :attr:`train_dataloader` should also be specified. If automatic
                mixed precision or gradient accmulation
                training is required. The type of ``optim_wrapper`` should be
                AmpOptimizerWrapper. See :meth:`build_optim_wrapper` for
                examples. Defaults to None.
            param_scheduler (_ParamScheduler or dict or list, optional):
                Parameter scheduler for updating optimizer parameters. If
                specified, :attr:`optimizer` should also be specified.
                Defaults to None.
                See :meth:`build_param_scheduler` for examples.
        """
        assert dispatch_kwargs is not None
        self.dispatch_kwargs.update(dispatch_kwargs)

        return_items = []

        model = self.build_model(model)
        model = self._init_model_weights(model)

        if optim_wrapper is not None:
            self.model = model
            self.optim_wrapper = self.build_optim_wrapper(optim_wrapper)
            self.model = self.wrap_model(self.model)
            return_items.append(self.model)
            return_items.append(self.optim_wrapper)
        else:
            self.model = self.wrap_model(model)
            return_items.append(self.model)

        if param_scheduler is not None:
            self.param_schedulers = self.build_param_scheduler(param_scheduler)
            return_items.append(self.param_schedulers)

        self.load_or_resume()

        return return_items[0] if len(return_items) == 1 else return_items

    def wrap_model(self, model: nn.Module) -> nn.Module:
        self.config['train_batch_size'] = self.dispatch_kwargs[
            'train_batch_size']

        if hasattr(self, 'optim_wrapper'):
            engine, self.optim_wrapper.optimizer, *_ = deepspeed.initialize(
                model=model,
                optimizer=self.optim_wrapper.optimizer,
                config=self.config)
        else:
            engine, *_ = deepspeed.initialize(model=model, config=self.config)

        wrapper = MMDeepSpeedEngineWrapper(
            model=engine, inputs_to_half=self._inputs_to_half)
        return wrapper

    def load_checkpoint(
        self,
        filename: str,
        *,
        map_location: Union[str, Callable] = 'cpu',
        callback: Optional[Callable] = None,
    ) -> dict:
        """Load checkpoint from given ``filename``.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
            map_location (str or callable): A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'cpu'.
        """
        if hasattr(self, 'extra_ckpt'):
            return self.extra_ckpt

        self.logger.info(f'Load checkpoint from {filename}')

        dirname, basename = osp.split(filename)
        self.extra_ckpt: dict
        _, self.extra_ckpt = self.model.load_checkpoint(
            dirname, tag=basename, load_optimizer_states=False)

        return self.extra_ckpt

    def resume(
        self,
        filename: str,
        *,
        resume_optimizer: bool = True,
        resume_param_scheduler: bool = True,
        map_location: Union[str, Callable] = 'default',
        callback: Optional[Callable] = None,
    ) -> dict:
        if hasattr(self, 'extra_ckpt'):
            return self.extra_ckpt

        self.logger.info(f'Resume checkpoint from {filename}')

        dirname, basename = osp.split(filename)
        _, self.extra_ckpt = self.model.load_checkpoint(dirname, tag=basename)

        if 'param_schedulers' in self.extra_ckpt:
            param_schedulers = self.extra_ckpt.pop('param_schedulers')
            self.load_scheduler_state_dict(param_schedulers)

        # resume random seed
        resumed_seed = self.extra_ckpt['meta'].get('seed', None)
        current_seed = self._randomness.get('seed')
        if resumed_seed is not None and resumed_seed != current_seed:
            if current_seed is not None:
                self.logger.warning(f'The value of random seed in the '
                                    f'checkpoint "{resumed_seed}" is '
                                    f'different from the value in '
                                    f'`randomness` config "{current_seed}"')
            self._randomness.update(seed=resumed_seed)
            self._set_randomness(**self._randomness)

        return self.extra_ckpt

    def save_checkpoint(
        self,
        filename: str,
        *,
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
        extra_ckpt: Optional[dict] = None,
        callback: Optional[Callable] = None,
    ) -> None:
        if extra_ckpt is None:
            extra_ckpt = dict()
        if 'meta' not in extra_ckpt:
            extra_ckpt['meta'] = dict()
        extra_ckpt['meta'].update(
            seed=self.seed,
            time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            mmengine=mmengine.__version__ + get_git_hash(),
        )

        extra_ckpt['param_schedulers'] = self.scheduler_state_dict()

        dirname, basename = osp.split(filename)
        self.model.save_checkpoint(
            dirname, tag=basename, client_state=extra_ckpt, save_latest=False)
