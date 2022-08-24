# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch, CPUOffload, FullyShardedDataParallel)

from mmengine.optim import OptimWrapper
from mmengine.registry import MODEL_WRAPPERS, Registry
from mmengine.structures import BaseDataElement

# support customize fsdp policy
FSDP_WRAP_POLICYS = Registry('fsdp wrap policy')


@MODEL_WRAPPERS.register_module()
class MMFullyShardedDataParallel(FullyShardedDataParallel):
    """A wrapper for sharding Module parameters across data parallel workers.

    Different from FullyShardedDataParallel, MMFullyShardedDataParallel
    implements three methods :meth:`train_step`, :meth:`val_step` and
    :meth:`test_step`, which will be called by ``train_loop``, ``val_loop``
    and ``test_loop``.

    - ``train_step``: Called by ``runner.train_loop``, and implement
      default model forward, gradient back propagation, parameter updating
      logic.

    - ``val_step``: Called by ``runner.val_loop`` and get the inference
      results. Specially, since MMFullyShardedDataParallel will wrap model
      recursively, it may cause some problem if one just use
      ``BaseModel.val_step`` to implement ``val_step`` here. To avoid that,
      ``val_step`` will call methods of :obj:`BaseModel` to pre-process
      data first, and use ``FullyShardedDataParallel.forward`` to get result.

    - ``test_step``: Called by ``runner.test_loop`` and get the inference
      results. Its logic is equivalent to ``val_loop``.

    Args:
        module (nn.Module): module to be wrapped with FSDP.
        process_group (Optional[ProcessGroup]): process group for sharding.
        cpu_offload (Optional[Union[bool,CPUOffload]]):
            CPU offloading config.
            Different from FullyShardedDataParallel,Since it can be set by
            users' pre-defined config in MMEngine,its type is expected to be
            `None`, `bool` or `CPUOffload`.

            Currently, only parameter and gradient CPU offload is supported.
            It can be enabled via passing in
            ``cpu_offload=CPUOffload(offload_params=True)``. Note that this
            currently implicitly enables gradient offloading to CPU in order
            for params and grads to be on same device to work with optimizer.
            This API is subject to change. Default is ``None`` in which case
            there will be no offloading.
        fsdp_auto_wrap_policy: (Optional[Union[str,Callable]]):
            Specifying a policy to recursively wrap layers with FSDP.
            Different from FullyShardedDataParallel, Since it can be set by
            users' pre-defined config in MMEngine, its type is expected to be
            `None`, `str` or `Callable`. If it's `str`, then
            MMFullyShardedDataParallel will try to get specified method in
            ``FSDP_WRAP_POLICYS`` registry,and this method will be passed to
            FullyShardedDataParallel to finally initialize model.

            Note that this policy currently will only apply to child modules of
            the passed in module. The remainder modules are always wrapped in
            the returned FSDP root instance.
            ``default_auto_wrap_policy`` written in
            ``torch.distributed.fsdp.wrap`` is an example of
            ``fsdp_auto_wrap_policy`` callable, this policy wraps layers with
            parameter sizes larger than 100M. Users can supply the customized
            ``fsdp_auto_wrap_policy`` callable that should accept following
            arguments: ``module: nn.Module``, ``recurse: bool``,
            ``unwrapped_params: int``, extra customized arguments could be
            added to the customized ``fsdp_auto_wrap_policy`` callable as well.

            Example::

                >>> def custom_auto_wrap_policy(
                >>>     module: nn.Module,
                >>>     recurse: bool,
                >>>     unwrapped_params: int,
                >>>     # These are customizable for this policy function.
                >>>     min_num_params: int = int(1e8),
                >>> ) -> bool:
                >>>     return unwrapped_params >= min_num_params

        backward_prefetch: (Optional[Union[str,BackwardPrefetch]]):
            Different from FullyShardedDataParallel, Since it will be set by
            users' pre-defined config in MMEngine,its type is expected to be
            `None`, `str` or `BackwardPrefetch`.

            This is an experimental feature that is subject to change in the
            the near future. It allows users to enable two different
            backward_prefetch algorithms to help backward communication and
            computation overlapping.
            Pros and cons of each algorithm is explained in class
            ``BackwardPrefetch``.
    """

    def __init__(
        self,
        module: nn.Module,
        process_group: Optional[ProcessGroup] = None,
        cpu_offload: Optional[Union[bool, CPUOffload]] = None,
        fsdp_auto_wrap_policy: Optional[Union[str, Callable]] = None,
        backward_prefetch: Optional[Union[str, BackwardPrefetch]] = None,
    ):

        if cpu_offload is not None:
            if isinstance(cpu_offload, bool):
                cpu_offload = CPUOffload(offload_params=cpu_offload)
            elif not isinstance(cpu_offload, CPUOffload):
                raise TypeError(
                    '`cpu_offload` should be `None`, `bool`'
                    f'or `CPUOffload`, but has type {type(cpu_offload)}')

        if fsdp_auto_wrap_policy is not None:
            if isinstance(fsdp_auto_wrap_policy, str):
                assert fsdp_auto_wrap_policy in FSDP_WRAP_POLICYS, \
                    '`FSDP_WRAP_POLICYS` has no ' \
                    f'function {fsdp_auto_wrap_policy}'
                fsdp_auto_wrap_policy = FSDP_WRAP_POLICYS.get(  # type: ignore
                    fsdp_auto_wrap_policy)
                if not isinstance(fsdp_auto_wrap_policy,
                                  Callable):  # type: ignore
                    raise TypeError(
                        'Registered `fsdp_auto_wrap_policy` needs to be '
                        '`Callable`, but has type '
                        f'{type(fsdp_auto_wrap_policy)}')
            elif not isinstance(fsdp_auto_wrap_policy,
                                Callable):  # type: ignore
                raise TypeError(
                    '`fsdp_auto_wrap_policy` should be `None`, `str` '
                    'or `Callable`, but has type '
                    f'{type(fsdp_auto_wrap_policy)}')

        if backward_prefetch is not None:
            if isinstance(backward_prefetch, str):
                assert backward_prefetch in ['pre', 'post'], \
                    '`backward_prefetch` should be either `pre` or `post`,' \
                    f' but get {backward_prefetch}'
                if backward_prefetch == 'pre':
                    backward_prefetch = BackwardPrefetch.BACKWARD_PRE
                else:
                    backward_prefetch = BackwardPrefetch.BACKWARD_POST
            elif not isinstance(backward_prefetch, BackwardPrefetch):
                raise TypeError('`backward_prefetch` should be `None`, `str` '
                                'or `BackwardPrefetch`, but has type '
                                f'{type(backward_prefetch)}')

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
