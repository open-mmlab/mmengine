# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Union

import torch

from mmengine.optim.optimizer._deepspeed import DeepSpeedOptimWrapper
from mmengine.registry import MODEL_WRAPPERS

try:
    from deepspeed.runtime.engine import DeepSpeedEngine
except ImportError:
    DeepSpeedEngine = None


@MODEL_WRAPPERS.register_module()
class MMDeepSpeedEngineWrapper:

    def __init__(
        self,
        *,
        model: DeepSpeedEngine,
        inputs_to_half: Optional[List[Union[int, str]]] = None,
    ):
        self.model = model
        self._inputs_to_half = inputs_to_half

    def __getattr__(self, name):
        return getattr(self.model, name)

    def train_step(
        self,
        data: Union[dict, tuple, list],
        optim_wrapper: DeepSpeedOptimWrapper,
    ) -> Dict[str, torch.Tensor]:
        data = self.model.module.data_preprocessor(data, training=True)
        data = self._cast_inputs_half(data)
        losses = self._run_forward(data, mode='loss')
        parsed_loss, log_vars = self.model.module.parse_losses(losses)
        optim_wrapper.update_params(parsed_loss)

        return log_vars

    def val_step(self, data: Union[dict, tuple, list]) -> list:
        """Gets the prediction of module during validation process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.model.module.data_preprocessor(data, False)
        data = self._cast_inputs_half(data)
        return self._run_forward(data, mode='predict')

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """Gets the predictions of module during testing process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.model.module.data_preprocessor(data, False)
        data = self._cast_inputs_half(data)
        return self._run_forward(data, mode='predict')

    def _run_forward(self, data: Union[dict, tuple, list], mode: str) -> Any:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = self.model(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self.model(*data, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results

    def _cast_inputs_half(self, inputs: Union[list, tuple, dict, None]):
        """Cast inputs to half precision if needed.

        Args:
            inputs (list or tuple or dict or None): Inputs to be casted.

        Returns:
            list or tuple or dict or None: Casted inputs.
        """
        if self._inputs_to_half is None:
            return inputs

        if isinstance(inputs, (list, tuple)):
            new_inputs = []
            for i, v in enumerate(inputs):
                if i in self._inputs_to_half:
                    new_inputs.append(v.half())
                else:
                    new_inputs.append(v)
            return inputs.__class__(new_inputs)
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                if k in self._inputs_to_half:
                    inputs[k] = v.half()
            return inputs
        else:
            raise TypeError('inputs should be list, tuple or dict, '
                            f'but got {type(inputs)}')
