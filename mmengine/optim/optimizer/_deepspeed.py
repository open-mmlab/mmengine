# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List

from mmengine.registry import OPTIM_WRAPPERS
from .base import BaseOptimWrapper


@OPTIM_WRAPPERS.register_module()
class DeepSpeedOptimWrapper(BaseOptimWrapper):

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def update_params(self, loss, model) -> None:  # type: ignore
        model.backward(loss)
        self.step(model)

    def step(self, model):
        model.step()

    def get_lr(self) -> Dict[str, List[float]]:
        """Get the learning rate of the optimizer.

        Provide unified interface to get learning rate of optimizer.

        Returns:
            Dict[str, List[float]]:
            param_groups learning rate of the optimizer.
        """
        res = {}
        res['lr'] = [group['lr'] for group in self.optimizer.param_groups]

        return res

    def get_momentum(self) -> Dict[str, List[float]]:
        """Get the momentum of the optimizer.

        Provide unified interface to get momentum of optimizer.

        Returns:
            Dict[str, List[float]]: Momentum of the optimizer.
        """
        momentum = []
        for group in self.optimizer.param_groups:
            # Get momentum of SGD.
            if 'momentum' in group.keys():
                momentum.append(group['momentum'])
            # Get momentum of Adam.
            elif 'betas' in group.keys():
                momentum.append(group['betas'][0])
            else:
                momentum.append(0)
        return dict(momentum=momentum)

    def state_dict(self) -> dict:
        raise NotImplementedError()

    def load_state_dict(self, state_dict: dict) -> None:
        raise NotImplementedError()
