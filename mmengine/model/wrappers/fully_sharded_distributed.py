# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, List

import torch
import torch.nn as nn
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch, CPUOffload, FullyShardedDataParallel)

from mmengine.data import BaseDataElement
from mmengine.optim import OptimWrapper
from mmengine.registry import MODEL_WRAPPERS, Registry

# support customize fsdp policy
FSDP_WRAP_POLICY = Registry('fsdp wrap policy')


@MODEL_WRAPPERS.register_module()
class MMFullyShardedDataParallel(FullyShardedDataParallel):

    def __init__(self,
                 module: nn.Module,
                 process_group=None,
                 cpu_offload=None,
                 fsdp_auto_wrap_policy=None,
                 backward_prefetch=None):

        if cpu_offload:
            cpu_offload = str(cpu_offload)
            assert cpu_offload in ['True', 'False'], \
                '`cpu_offload` should be either `True` or `False`,' \
                f' but get {cpu_offload}'

            cpu_offload = CPUOffload(offload_params=(cpu_offload == 'True'))

        if fsdp_auto_wrap_policy:
            assert fsdp_auto_wrap_policy in FSDP_WRAP_POLICY, \
                f'`FSDP_WRAP_POLICY` has no function {fsdp_auto_wrap_policy}'
            fsdp_auto_wrap_policy = FSDP_WRAP_POLICY.get(
                'fsdp_auto_wrap_policy')
            if isinstance(fsdp_auto_wrap_policy, Callable):  # type: ignore
                raise TypeError(
                    '`fsdp_auto_wrap_policy` needs to be `Callable`,'
                    f' but has type {type(fsdp_auto_wrap_policy)} ')

        if backward_prefetch:
            backward_prefetch = str(backward_prefetch)
            assert backward_prefetch in ['pre', 'post'], \
                '`backward_prefetch` should be either `pre` or `post`,' \
                f' but get {backward_prefetch}'
            if backward_prefetch == 'pre':
                backward_prefetch = BackwardPrefetch.BACKWARD_PRE
            else:
                backward_prefetch = BackwardPrefetch.BACKWARD_POST

        super().__init__(module, process_group, cpu_offload,
                         fsdp_auto_wrap_policy, backward_prefetch)

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Interface for model forward, backward and parameters updating during
        training process.

        :meth:`train_step` will perform the following steps in order:

        - If :attr:`module` defines the preprocess method,
            call ``module.preprocess`` to pre-processing data.
        - Call ``module.forward(**data)`` and get losses.
        - Parse losses.
        - Call ``optim_wrapper.optimizer_step`` to update parameters.
        - Return log messages of losses.

        Args:
            data (List[dict]): Data sampled by dataloader.
            optim_wrapper (OptimWrapper): A wrapper of optimizer to
                update parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            batch_inputs, data_samples = self.module.data_preprocessor(
                data, training=True)
            losses = self(batch_inputs, data_samples, mode='loss')
        parsed_loss, log_vars = self.module.parse_losses(losses)
        optim_wrapper.update_params(parsed_loss)
        return log_vars

    def val_step(self, data: List[dict]) -> List[BaseDataElement]:
        """Gets the prediction of module during validation process.

        Args:
            data (List[dict]): Data sampled by dataloader.

        Returns:
            List[BaseDataElement] or dict: The predictions of given data.
        """
        inputs, data_sample = self.module.data_preprocessor(data, False)
        return self(inputs, data_sample, mode='predict')

    def test_step(self, data: List[dict]) -> List[BaseDataElement]:
        """Gets the predictions of module during testing process.

        Args:
            data: Data sampled by dataloader.

        Returns:
            List[BaseDataElement]: The predictions of given data.
        """
        inputs, data_sample = self.module.data_preprocessor(data, False)
        return self(inputs, data_sample, mode='predict')
