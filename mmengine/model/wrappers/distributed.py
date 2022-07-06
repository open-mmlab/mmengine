# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
from torch.nn.parallel.distributed import DistributedDataParallel

from mmengine.data import BaseDataElement
from mmengine.optim import OptimWrapper
from mmengine.registry import MODEL_WRAPPERS
from ..utils import detect_anomalous_params


@MODEL_WRAPPERS.register_module()
class MMDistributedDataParallel(DistributedDataParallel):
    """A distributed model wrapper used for training,testing and validation in
    loop.

    Different from DistributedDataParallel, MMDistributedDataParallel
    implements three methods :meth:`train_step`, :meth:`val_step` and
    :meth:`test_step`, which will be called by ``train_loop``, ``val_loop``
    and ``test_loop``.

    - ``train_step``: Called by ``runner.train_loop``, and implement
      default model forward, gradient back propagation, parameter updating
      logic. To take advantage of DistributedDataParallel's automatic gradient
      synchronization, ``train_step`` calls ``DistributedDataParallel.forward``
      to calculate the losses, and call other methods of :obj:`BaseModel` to
      pre-process data and parse losses. Finally, update model parameters by
      :obj:``OptimWrapper`` and return the loss dictionary used for logging.

    - ``val_step``: Called by ``runner.val_loop`` and get the inference
      results. Since there is no gradient synchronization requirement,
      this procedure is equivalent to ``BaseModel.val_step``

    - ``test_step``: Called by ``runner.test_loop``, equivalent ``val_step``.

    Args:
        detect_anomalous_params (bool): This option is only used for
            debugging which will slow down the training speed.
            Detect anomalous parameters that are not included in
            the computational graph with `loss` as the root.
            There are two cases

                - Parameters were not used during
                  forward pass.
                - Parameters were not used to produce
                  loss.
            Default: False.

        **kwargs: keyword arguments passed to ``DistributedDataParallel``.

            - device_ids (List[int] or torch.device, optional): CUDA devices
              for module.
            - output_device (int or torch.device, optional): Device location of
              output for single-device CUDA modules.
            - dim (int): Defaults to 0.
            - broadcast_buffers (bool): Flag that enables syncing (
                broadcasting) buffers of the module at beginning of the
                ``forward`` function. Defaults to True
            - find_unused_parameters (bool): Whether to find parameters of
              module, which are not in the forward graph. Defaults to False.
            - process_group (ProcessGroup, optional): The process group to be
              used for distributed data all-reduction.
            - bucket_cap_mb (int): bucket size in MegaBytes (MB). Defaults
              to 25.
            - check_reduction (bool): This argument is deprecated. Defaults
              to False.
            - gradient_as_bucket_view (bool): Defaults to False.
            - static_graph (bool): Defaults to False.

    See more information about arguments in `https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel`_  # noqa E501

    Note:
        If model has multiple submodules and each module has
        separate optimization strategies,
        :class:`MMSeparateDistributedDataParallel` should be used to wrap
        the model.

    Note:
        If model itself has custom optimization strategy, rather than
        simply forward model and update model. A custom model wrapper
        inherit from ``MMDistributedDataParallel`` should be defined and
        override the ``train_step`` method.
    """

    def __init__(self,
                 module,
                 detect_anomalous_params: bool = False,
                 **kwargs):
        super().__init__(module=module, **kwargs)
        self.detect_anomalous_params = detect_anomalous_params

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
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            batch_inputs, data_samples = self.module.data_preprocessor(
                data, training=True)
            losses = self(batch_inputs, data_samples, mode='loss')
        if self.detect_anomalous_params:
            detect_anomalous_params(losses, model=self)
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
        return self.module.val_step(data)

    def test_step(self, data: List[dict]) -> List[BaseDataElement]:
        """Gets the predictions of module during testing process.

        Args:
            data: Data sampled by dataloader.

        Returns:
            List[BaseDataElement]: The predictions of given data.
        """
        return self.module.test_step(data)
