# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
from typing import List, Union

import torch
import torch.nn as nn

from mmengine.config import Config, ConfigDict
from mmengine.registry import OPTIMIZER_WRAPPER_CONSTRUCTORS, OPTIMIZERS
from .optimizer_wrapper_dict import OptimWrapper


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


def build_optimizer_wrapper(
        model: nn.Module, cfg: Union[dict, Config,
                                     ConfigDict]) -> OptimWrapper:
    """Build function of OptimWrapper.

    If ``constructor`` is set in the ``cfg``, this method will build an
    optimizer wrapper constructor, and use optimizer wrapper constructor to
    build the optimizer wrapper. If ``constructor`` is not set, the
    ``DefaultOptimWrapperConstructor`` will be used by default.

    Args:
        model (nn.Module): Model to be optimized.
        cfg (dict): Config of optimizer, optimizer_wrapper and optimizer
            constructor.

    Returns:
        OptimWrapper: The built optimizer wrapper.
    """
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimWrapperConstructor')
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_wrapper_constructor = OPTIMIZER_WRAPPER_CONSTRUCTORS.build(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))
    optimizer_wrapper = optim_wrapper_constructor(model)
    return optimizer_wrapper
