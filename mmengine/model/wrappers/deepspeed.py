# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Union

import deepspeed
import torch
from deepspeed.runtime.engine import DeepSpeedEngine

from mmengine.optim import OptimWrapper
from mmengine.registry import MODEL_WRAPPERS
from mmengine.utils import digit_version


@MODEL_WRAPPERS.register_module()
class MMDeepSpeedEngine(DeepSpeedEngine):

    def __init__(
        self,
        args=None,
        model=None,
        optimizer=None,
        model_parameters=None,
        training_data=None,
        lr_scheduler=None,
        mpu=None,
        dist_init_required=None,
        collate_fn=None,
        config=None,
        config_class=None,
        dont_change_device=False,
        inputs_to_half: Optional[List[Union[int, str]]] = None,
    ):

        if digit_version(deepspeed.__version__) >= digit_version('0.9.0'):
            from deepspeed.runtime.config import DeepSpeedConfig
            config_class = DeepSpeedConfig(config, mpu)

            super().__init__(
                args=args,
                model=model,
                optimizer=optimizer,
                model_parameters=model_parameters,
                training_data=training_data,
                lr_scheduler=lr_scheduler,
                mpu=mpu,
                dist_init_required=dist_init_required,
                collate_fn=collate_fn,
                config=config,
                config_class=config_class,
                dont_change_device=dont_change_device)
        else:
            super().__init__(
                args=args,
                model=model,
                optimizer=optimizer,
                model_parameters=model_parameters,
                training_data=training_data,
                lr_scheduler=lr_scheduler,
                mpu=mpu,
                dist_init_required=dist_init_required,
                collate_fn=collate_fn,
                config=config,
                dont_change_device=dont_change_device)

        self._inputs_to_half = inputs_to_half

    def train_step(
        self,
        data: Union[dict, tuple, list],
        optim_wrapper: OptimWrapper,
    ) -> Dict[str, torch.Tensor]:
        data = self.module.data_preprocessor(data, training=True)
        data = self._cast_inputs_half(data)
        losses = self._run_forward(data, mode='loss')
        parsed_loss, log_vars = self.module.parse_losses(losses)
        optim_wrapper.update_params(parsed_loss, model=self)

        return log_vars

    def val_step(self, data: Union[dict, tuple, list]) -> list:
        """Gets the prediction of module during validation process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.module.data_preprocessor(data, False)
        data = self._cast_inputs_half(data)
        return self._run_forward(data, mode='predict')

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """Gets the predictions of module during testing process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.module.data_preprocessor(data, False)
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
            results = self(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results

    def _cast_inputs_half(self, inputs):
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
