# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import time
from typing import Callable, Dict, List, Optional, Union

import deepspeed
import torch.nn as nn

import mmengine
from mmengine.dist import init_dist
from mmengine.model.wrappers._deepspeed import MMDeepSpeedEngineWrapper
from mmengine.optim import BaseOptimWrapper, _ParamScheduler
from mmengine.registry import STRATEGIES
from mmengine.utils import get_git_hash
from .base import BaseStrategy


@STRATEGIES.register_module()
class DeepSpeedStrategy(BaseStrategy):
    """Support training models with DeepSpeed.

    Note:
        The detailed usage of parameters can be found at
        https://www.deepspeed.ai/docs/config-json/.

    Args:
        config (str or dict, optional): If it is a string, it is a path to load
            config for deepspeed. Defaults to None.
        zero_optimization (dict, optional): Enabling and configuring ZeRO
            memory optimizations. Defaults to None.
        gradient_clipping (float): Enable gradient clipping with value.
            Defaults to 1.0.
        fp16 (dict, optional): Configuration for using mixed precision/FP16
            training that leverages NVIDIA's Apex package.
        inputs_to_half (list[int or str], optional): Which inputs are to
            converted to half precision. Defaults to None.
            If ``fp16`` is enabled, it also should be set.
        bf16 (dict, optional): Configuration for using bfloat16 floating-point
            format as an alternative to FP16. Defaults to None.
        amp (dict, optional): Configuration for using automatic mixed
            precision (AMP) training that leverages NVIDIA's Apex AMP package.
            Defaults to None.
        activation_checkpointing (dict, optional): Reduce memory usage by
            clearing activations of certain layers and recomputing them
            during a backward pass.
            Defaults to None.
        aio (dict, optional): Configuring the asynchronous I/O module for
            offloading parameter and optimizer states to persistent (NVMe)
            storage. This module uses Linux native asynchronous I/O (libaio).
            Defaults to None.
    """

    def __init__(
        self,
        *,
        # the following args are for deepspeed
        config: Union[str, dict, None] = None,
        zero_optimization: Optional[dict] = None,
        gradient_clipping: float = 1.0,
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
        self.config['gradient_clipping'] = gradient_clipping
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

    def _setup_distributed(  # type: ignore
        self,
        launcher: Optional[str] = None,
        backend: str = 'nccl',
        **kwargs,
    ):
        """Setup distributed environment.

        Args:
            launcher (str, optional): Way to launch multi processes.
                DeepSpeedStrategy does not support the launcher argument.
            backend (str): Communication Backends. Supported backends are
                'nccl', 'gloo' and 'mpi'. Defaults to 'nccl'.
            **kwargs: Other arguments for :func:`deepspeed.init_distributed`.
        """
        init_dist(launcher, backend, init_backend='deepspeed', **kwargs)

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
                can be a dict used for build a model.

        Keyword Args:
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
        assert dispatch_kwargs is not None
        self.dispatch_kwargs.update(dispatch_kwargs)

        return_items = []

        model = self.build_model(model)
        model = self._init_model_weights(model)

        if optim_wrapper is not None:
            self.optim_wrapper = self.build_optim_wrapper(optim_wrapper, model)
            self.model = self._wrap_model(model)

            self.optim_wrapper.model = self.model  # type: ignore

            return_items.append(self.model)
            return_items.append(self.optim_wrapper)
        else:
            self.model = self._wrap_model(model)
            return_items.append(self.model)

        if param_scheduler is not None:
            self.param_schedulers = self.build_param_scheduler(
                param_scheduler, self.optim_wrapper)
            return_items.append(self.param_schedulers)

        return return_items[0] if len(return_items) == 1 else return_items

    def _wrap_model(self, model: nn.Module) -> nn.Module:
        self.config['train_micro_batch_size_per_gpu'] = self.dispatch_kwargs[
            'train_micro_batch_size_per_gpu']

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
        strict: bool = False,
        revise_keys: list = [(r'^module.', '')],
        callback: Optional[Callable] = None,
    ) -> dict:
        """Load checkpoint from given ``filename``.

        Warning:
            `map_localtion` and `callback` parameters are not supported yet.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
        """
        self.logger.info(f'Load checkpoint from {filename}')

        dirname, basename = osp.split(filename)
        _, extra_ckpt = self.model.load_checkpoint(
            dirname, tag=basename, load_optimizer_states=False)

        return extra_ckpt

    def resume(
        self,
        filename: str,
        *,
        resume_optimizer: bool = True,
        resume_param_scheduler: bool = True,
        map_location: Union[str, Callable] = 'default',
        callback: Optional[Callable] = None,
    ) -> dict:
        """Resume training from given ``filename``.

        Warning:
            `map_location` and `callback` parameters are not supported yet.

        Args:
            filename (str): Accept local filepath.

        Keyword Args:
            resume_optimizer (bool): Whether to resume optimizer state.
                Defaults to True.
            resume_param_scheduler (bool): Whether to resume param scheduler
                state. Defaults to True.
        """
        self.logger.info(f'Resume checkpoint from {filename}')

        dirname, basename = osp.split(filename)
        _, extra_ckpt = self.model.load_checkpoint(
            dirname, tag=basename, load_optimizer_states=resume_optimizer)

        if resume_optimizer:
            self.load_optim_state_dict(extra_ckpt.pop('optim_wrapper'))

        if resume_param_scheduler:
            param_schedulers = extra_ckpt.pop('param_schedulers')
            self.load_scheduler_state_dict(param_schedulers)

        # resume random seed
        resumed_seed = extra_ckpt['meta'].get('seed', None)
        current_seed = self._randomness.get('seed')
        if resumed_seed is not None and resumed_seed != current_seed:
            if current_seed is not None:
                self.logger.warning(f'The value of random seed in the '
                                    f'checkpoint "{resumed_seed}" is '
                                    f'different from the value in '
                                    f'`randomness` config "{current_seed}"')
            self._randomness.update(seed=resumed_seed)
            self._set_randomness(**self._randomness)

        return extra_ckpt

    def save_checkpoint(
        self,
        filename: str,
        *,
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
        extra_ckpt: Optional[dict] = None,
        callback: Optional[Callable] = None,
    ) -> None:
        """Save checkpoint to given ``filename``.

        Warning:
            `save_optimizer` and `callback` parameters are not supported yet.

        Args:
            filename (str): Filename to save checkpoint.

        Keyword Args:
            save_param_scheduler (bool): Whether to save the param_scheduler
                to the checkpoint. Defaults to True.
            extra_ckpt (dict, optional): Extra checkpoint to save.
                Defaults to None.
        """
        if extra_ckpt is None:
            extra_ckpt = dict()
        if 'meta' not in extra_ckpt:
            extra_ckpt['meta'] = dict()
        extra_ckpt['meta'].update(
            seed=self.seed,
            time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            mmengine=mmengine.__version__ + get_git_hash(),
        )

        if save_optimizer:
            # The key can not be 'optimizer', otherwise error will be thrown
            # when loading or resuming checkpoint.
            extra_ckpt['optim_wrapper'] = self.optim_state_dict()

        if save_param_scheduler:
            extra_ckpt['param_schedulers'] = self.scheduler_state_dict()

        dirname, basename = osp.split(filename)
        self.model.save_checkpoint(
            dirname, tag=basename, client_state=extra_ckpt, save_latest=False)
