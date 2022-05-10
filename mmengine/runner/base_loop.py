# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Dict, Union

import torch
from torch.utils.data import DataLoader


class BaseLoop(metaclass=ABCMeta):
    """Base loop class.

    All subclasses inherited from ``BaseLoop`` should overwrite the
    :meth:`run` method.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
    """

    def __init__(self, runner, dataloader: Union[DataLoader, Dict]) -> None:
        self._runner = runner

        if isinstance(dataloader, dict):
            self.dataloader = runner.build_dataloader(dataloader)
        else:
            self.dataloader = dataloader

    @property
    def runner(self):
        return self._runner

    @abstractmethod
    def run(self) -> None:
        """Execute loop."""

    def _parse_losses(self, losses: Dict[str, torch.Tensor]):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        # Deprecated model output.
        if 'log_vars' in losses:
            return losses

        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        return dict(loss=loss, log_vars=log_vars)
