# Copyright (c) OpenMMLab. All rights reserved.

from mmengine.registry import OPTIM_WRAPPERS
from .optimizer_wrapper import OptimWrapper


@OPTIM_WRAPPERS.register_module()
class DSOptimWrapper(OptimWrapper):

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def update_params(self, loss, model) -> None:
        model.backward(loss)
        model.step()
