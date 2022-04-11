# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional, Sequence

import torch
from torch.nn.parameter import Parameter
from torch.nn.utils import clip_grad

from mmengine.registry import HOOKS
from .hook import Hook

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class OptimizerHook(Hook):
    """A hook contains custom operations for the optimizer.

    Args:
        grad_clip (dict, optional): A config dict to control the clip_grad.
            Defaults to None.
        detect_anomalous_params (bool): This option is only used for
            debugging which will slow down the training speed.
            Detect anomalous parameters that are not included in
            the computational graph with ``loss`` as the root.
            There are two cases
                - Parameters were not used during
                  forward pass.
                - Parameters were not used to produce
                  loss.
            Defaults to False.
    """

    priority = 'HIGH'

    def __init__(self,
                 grad_clip: Optional[dict] = None,
                 detect_anomalous_params: bool = False) -> None:
        self.grad_clip = grad_clip
        self.detect_anomalous_params = detect_anomalous_params

    def clip_grads(self, params: List[Parameter]) -> Optional[torch.Tensor]:
        """Clip the gradients of parameters.

        Args:
            params (list[Parameter]): Model's parameters.

        Returns:
            Optional[torch.Tensor]: Total norm of the parameters if there is
            at least one param requiring gradient, else None.
        """
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)
        return None

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """All operations need to be finished after each training iteration.

        This function will finish following 3 operations:

        - Detect any anomalous parameters which are not included in the
          training graph. (optional)

        - Compute the gradient of model parameters.

        - Clip the gradients of each parameter. (optional)

        - Update model parameters with gradients.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                In order to keep this interface consistent with other hooks,
                we keep ``data_batch`` here. Defaults to None.
            outputs (dict, optional): Outputs from model.
                In order to keep this interface consistent with other hooks,
                we keep ``outputs`` here. Defaults to None.
        """
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        runner.outputs['loss'].backward()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()

    def detect_anomalous_parameters(self, loss: torch.Tensor, runner) -> None:
        """Detect anomalous parameters that are not included in the graph.

        Args:
            loss (torch.Tensor): The loss of current iteration.
            runner (Runner): The runner of the training process.
        """
        logger = runner.logger
        parameters_in_graph = set()
        visited = set()

        def traverse(grad_fn):
            if grad_fn is None:
                return
            if grad_fn not in visited:
                visited.add(grad_fn)
                if hasattr(grad_fn, 'variable'):
                    parameters_in_graph.add(grad_fn.variable)
                parents = grad_fn.next_functions
                if parents is not None:
                    for parent in parents:
                        grad_fn = parent[0]
                        traverse(grad_fn)

        traverse(loss.grad_fn)
        for n, p in runner.model.named_parameters():
            if p not in parameters_in_graph and p.requires_grad:
                logger.log(
                    level=logging.ERROR,
                    msg=f'{n} with shape {p.size()} is not '
                    f'in the computational graph \n')
