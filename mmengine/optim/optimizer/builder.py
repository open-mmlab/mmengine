# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
from typing import List

import torch
import torch.nn as nn

from mmengine.registry import OPTIMIZER_CONSTRUCTORS, OPTIMIZERS
from .optimizer_wrapper import OptimizerWrapper


def register_torch_optimizers() -> List[str]:
    """Register optimizers in ``torch.optim`` to the ``OPTIMIZERS`` registry.

    Returns:
        List[str]: A list of registered optimizers' name.
    """
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


def build_optimizer_wrapper(model: nn.Module, cfg: dict) -> OptimizerWrapper:
    """Build function of optimizer.

    If ``constructor`` is set in the ``cfg``, this method will build an
    optimizer constructor, and use optimizer constructor to build the
    optimizer. If ``constructor`` is not set, the
    ``DefaultOptimizerConstructor`` will be used by default.

    Args:
        model (nn.Module): Model to be optimized.
        cfg (dict): Config of optimizer and optimizer constructor.

    Returns:
        torch.optim.Optimizer: The built optimizer.
    """
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimizerConstructor')
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_constructor = OPTIMIZER_CONSTRUCTORS.build(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))
    optimizer = optim_constructor(model)
    return optimizer
