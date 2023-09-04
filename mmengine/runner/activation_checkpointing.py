# Copyright (c) OpenMMLab. All rights reserved.
from functools import wraps
from operator import attrgetter
from typing import List, Union

import torch
from torch.utils.checkpoint import checkpoint


def wrap_forward(forward):

    @wraps(forward)
    def wrapper(*args):
        return checkpoint(forward, *args)

    return wrapper


def turn_on_activation_checkpointing(model: torch.nn.Module,
                                     modules: Union[List[str], str]):

    if isinstance(modules, str):
        modules = [modules]
    for module_name in modules:
        module = attrgetter(module_name)(model)
        module.forward = wrap_forward(module.forward)
