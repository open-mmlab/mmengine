# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmengine.registry import OPTIM_WRAPPERS
from .base import BaseOptimWrapper


@OPTIM_WRAPPERS.register_module()
class DeepSpeedOptimWrapper(BaseOptimWrapper):

    def __init__(self, optimizer):
        super().__init__(optimizer)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            raise ValueError('model attribute should be set before accessing.')
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def update_params(self, loss) -> None:  # type: ignore
        """Update parameters in :attr:`optimizer`."""
        self.backward(loss)
        self.step()

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """"Perform gradient back propagation."""
        self.model.backward(loss)

    def zero_grad(self, **kwargs) -> None:
        raise NotImplementedError(
            'DeepSpeedOptimWrapper does not support zero_grad method '
            'currently.')

    def step(self, **kwargs):
        self.model.step()

    def state_dict(self) -> dict:
        state_dict = {}
        if self.base_param_settings is not None:
            state_dict['base_param_settings'] = self.base_param_settings

        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        base_param_settings = state_dict.pop('base_param_settings', None)

        if base_param_settings is not None:
            self.base_param_settings = base_param_settings
