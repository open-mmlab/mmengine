# Copyright (c) OpenMMLab. All rights reserved.
import time
from typing import Callable, Dict, List, Optional, Union

import torch.nn as nn

import mmengine
from mmengine.device import get_device
from mmengine.optim import OptimWrapper, _ParamScheduler
from mmengine.registry import STRATEGIES
from mmengine.utils import get_git_hash
from .base import BaseStrategy


@STRATEGIES.register_module()
class SingleDeviceStrategy(BaseStrategy):
    """Strategy for single device training."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        Keyword Args:
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
        if dispatch_kwargs is not None:
            self.dispatch_kwargs.update(dispatch_kwargs)

        return_items = []
        model = self.build_model(model)
        model = self._init_model_weights(model)
        model = self.wrap_model(model)
        self.model = self.compile_model(model)
        return_items.append(self.model)

        if optim_wrapper is not None:
            self.optim_wrapper = self.build_optim_wrapper(optim_wrapper)
            return_items.append(self.optim_wrapper)

        if param_scheduler is not None:
            self.param_schedulers = self.build_param_scheduler(param_scheduler)
            return_items.append(self.param_schedulers)

        self.load_or_resume()

        if optim_wrapper is not None:
            # Initiate inner count of `optim_wrapper`.
            self.optim_wrapper.initialize_count_status(
                self.model, self.dispatch_kwargs.get('cur_iter', 0),
                self.dispatch_kwargs['max_iters'])

        return return_items[0] if len(return_items) == 1 else return_items

    def wrap_model(self, model: nn.Module) -> nn.Module:
        model = self.convert_model(model)
        current_device = get_device()
        return model.to(current_device)

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

        Keyword Args:
            map_location (str or callable): A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'cpu'.
            callback (callable, callable): Callback function to modify the
                checkpoint after loading the checkpoint.
                Defaults to None.
        """
        from mmengine.runner.checkpoint import _load_checkpoint

        if hasattr(self, 'extra_ckpt'):
            return self.extra_ckpt

        self.logger.info(f'Load checkpoint from {filename}')

        self.extra_ckpt: dict
        if map_location == 'default':
            device = get_device()
            self.extra_ckpt = _load_checkpoint(filename, map_location=device)
        else:
            self.extra_ckpt = _load_checkpoint(
                filename, map_location=map_location)

        # users can do some modification after loading checkpoint
        if callback is not None:
            callback(self.extra_ckpt)

        state_dict = self.extra_ckpt.pop('state_dict')
        self.load_model_state_dict(state_dict)

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
        """Resume training from given ``filename``.

        Four types of states will be resumed.

        - model state
        - optimizer state
        - scheduler state
        - randomness state

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.

        Keyword Args:
            resume_optimizer (bool): Whether to resume optimizer state.
                Defaults to True.
            resume_param_scheduler (bool): Whether to resume param scheduler
                state. Defaults to True.
            map_location (str or callable):A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'default'.
            callback (callable, callable): Callback function to modify the
                checkpoint before saving the checkpoint.
                Defaults to None.
        """
        if hasattr(self, 'extra_ckpt'):
            return self.extra_ckpt

        self.logger.info(f'Resume checkpoint from {filename}')

        self.extra_ckpt = self.load_checkpoint(
            filename, map_location=map_location, callback=callback)

        if not resume_optimizer:
            self.extra_ckpt.pop('optimizer', None)
        else:
            self.load_optim_state_dict(self.extra_ckpt.pop('optimizer'))

        if not resume_param_scheduler:
            self.extra_ckpt.pop('param_schedulers', None)
        else:
            self.load_scheduler_state_dict(
                self.extra_ckpt.pop('param_schedulers'))

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

        # resume iter
        self.dispatch_kwargs['cur_iter'] = self.extra_ckpt['meta']['iter']

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
        """Save checkpoint to given ``filename``.

        Args:
            filename (str): Filename to save checkpoint.

        Keyword Args:
            save_optimizer (bool): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            save_param_scheduler (bool): Whether to save the param_scheduler
                to the checkpoint. Defaults to True.
            extra_ckpt (dict, optional): Extra checkpoint to save.
                Defaults to None.
            callback (callable, callable): Callback function to modify the
                checkpoint before saving the checkpoint.
                Defaults to None.
        """
        from mmengine.runner.checkpoint import save_checkpoint

        state_dict = dict()
        state_dict['state_dict'] = self.model_state_dict()

        # save optimizer state dict
        if save_optimizer and hasattr(self, 'optim_wrapper'):
            state_dict['optimizer'] = self.optim_state_dict()

        # save param scheduler state dict
        if save_param_scheduler and not hasattr(self, 'param_schedulers'):
            self.logger.warning(
                '`save_param_scheduler` is True but strategy has no '
                'param_schedulers attribute, so skip saving parameter '
                'schedulers')
            save_param_scheduler = False

        if save_param_scheduler:
            state_dict['param_schedulers'] = self.scheduler_state_dict(
            )  # type: ignore  # noqa: E501

        # save extra checkpoint passed by users
        if extra_ckpt is None:
            extra_ckpt = dict()
        if 'meta' not in extra_ckpt:
            extra_ckpt['meta'] = dict()
        extra_ckpt['meta'].update(
            seed=self.seed,
            time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            mmengine=mmengine.__version__ + get_git_hash(),
        )

        state_dict.update(extra_ckpt)

        # users can do some modification before saving checkpoint
        if callback is not None:
            callback(state_dict)

        save_checkpoint(state_dict, filename)
