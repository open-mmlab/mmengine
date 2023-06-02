# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch.nn as nn

from mmengine.device import get_device
from mmengine.optim import OptimWrapper, _ParamScheduler
from mmengine.registry import STRATEGIES
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
