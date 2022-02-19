# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
from typing import Callable, List

import torch
import torch.nn as nn

from mmengine.registry import OPTIMIZER_CONSTRUCTORS, OPTIMIZERS


def register_torch_optimizers() -> List[str]:
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            OPTIMIZERS.register_module(module=_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers


TORCH_OPTIMIZERS = register_torch_optimizers()


def build_optimizer_constructor(cfg: dict) -> Callable:
    return OPTIMIZER_CONSTRUCTORS.build(cfg)


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimizerConstructor')
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_constructor = build_optimizer_constructor(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))
    optimizer = optim_constructor(model)
    return optimizer
