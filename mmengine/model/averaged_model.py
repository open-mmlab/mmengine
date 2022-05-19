# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from abc import abstractmethod
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmengine.registry import MODELS


class BaseAveragedModel(nn.Module):
    """A base class for averaging model weights.

    Weight averaging, such as SWA and EMA, is a widely used technique for
    training neural networks. This class implements the averaging process
    for a model. All subclasses must implement the `avg_func` method.
    This class creates a copy of the provided module :attr:`model`
    on the device :attr:`device` and allows computing running averages of the
    parameters of the :attr:`model`.
    The code is referenced from: https://github.com/pytorch/pytorch/blob/master/torch/optim/swa_utils.py

    In mmengine, we provide two ways to use the model averaging:
    1. Use the model averaging module in hook:
        We provide an EMAHook to apply the model averaging during training.
        Add ``custom_hooks=[dict(type='EMAHook')]`` to the config or the runner.
        The hook is implemented in mmengine/hooks/ema_hook.py

    2. Use the model averaging module directly in the algorithm. Take the ema
       teacher in semi-supervise as an example:
        >>> from mmengine.model import ExponentialMovingAverage
        >>> student = ResNet(depth=50)
        >>> # use ema model as teacher
        >>> ema_teacher = ExponentialMovingAverage(student)

    Args:
        model (nn.Module): The model to be averaged.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """  # noqa: E501

    def __init__(self,
                 model: nn.Module,
                 interval: int = 1,
                 device: Optional[torch.device] = None,
                 update_buffers: bool = False) -> None:
        super().__init__()
        self.module = deepcopy(model)
        self.interval = interval
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('steps',
                             torch.tensor(0, dtype=torch.long, device=device))
        self.update_buffers = update_buffers

    @abstractmethod
    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> Tensor:
        """Compute the average of the parameters. All subclasses must implement
        this method.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """

    def forward(self, *args, **kwargs):
        """Forward method of the averaged model."""
        return self.module(*args, **kwargs)

    def update_parameters(self, model: nn.Module) -> None:
        """Update the parameters of the model. This method will execute the
        ``avg_func`` to compute the new parameters and update the model's
        parameters.

        Args:
            model (nn.Module): The model whose parameters will be averaged.
        """
        if self.steps % self.interval == 0:
            avg_param = (
                itertools.chain(self.module.parameters(),
                                self.module.buffers())
                if self.update_buffers else self.parameters())
            src_param = (
                itertools.chain(model.parameters(), model.buffers())
                if self.update_buffers else model.parameters())
            for p_avg, p_src in zip(avg_param, src_param):
                device = p_avg.device
                p_src_ = p_src.detach().to(device)
                if self.steps == 0:
                    p_avg.detach().copy_(p_src_)
                else:
                    p_avg.detach().copy_(
                        self.avg_func(p_avg.detach(), p_src_,
                                      self.steps.to(device)))
        self.steps += 1


@MODELS.register_module()
class StochasticWeightAverage(BaseAveragedModel):
    """Implements the stochastic weight averaging (SWA) of the model.

    Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
    Wider Optima and Better Generalization, UAI 2018.
    <https://arxiv.org/abs/1803.05407>`_ by Pavel Izmailov, Dmitrii
    Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson.
    """

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> Tensor:
        """Compute the average of the parameters using stochastic weight
        average.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        Returns:
            Tensor: The averaged parameters.
        """
        return averaged_param + (source_param - averaged_param) / (
            steps // self.interval + 1)


@MODELS.register_module()
class ExponentialMovingAverage(BaseAveragedModel):
    """Implements the exponential moving average (EMA) of the model.

    All parameters are updated by the formula as below:

        .. math::

            Xema\_{t+1} = (1 - \text{momentum}) \times
            Xema\_{t} +  \text{momentum} \times X_t

    Args:
        model (nn.Module): The model to be averaged.
        momentum (float): The momentum used for updating ema parameter.
            Ema's parameter are updated with the formula:
           `averaged_param = (1-momentum) * averaged_param + momentum *
           source_param`. Defaults to 0.0002.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """  # noqa: W605

    def __init__(self,
                 model: nn.Module,
                 momentum: float = 0.0002,
                 interval: int = 1,
                 device: Optional[torch.device] = None,
                 update_buffers: bool = False) -> None:
        super().__init__(model, interval, device, update_buffers)
        assert 0.0 < momentum < 1.0, 'momentum must be in range (0.0, 1.0)'\
                                     f'but got {momentum}'
        self.momentum = momentum

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> Tensor:
        """Compute the moving average of the parameters using exponential
        moving average.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        Returns:
            Tensor: The averaged parameters.
        """
        return averaged_param * (1 -
                                 self.momentum) + source_param * self.momentum


@MODELS.register_module()
class MomentumAnnealingEMA(ExponentialMovingAverage):
    """Exponential moving average (EMA) with momentum annealing strategy.

    Args:
        model (nn.Module): The model to be averaged.
        momentum (float): The momentum used for updating ema parameter.
            Ema's parameter are updated with the formula:
           `averaged_param = (1-momentum) * averaged_param + momentum *
           source_param`. Defaults to 0.0002.
        gamma (int): Use a larger momentum early in training and gradually
            annealing to a smaller value to update the ema model smoothly. The
            momentum is calculated as max(momentum, gamma / (gamma + steps))
            Defaults to 100.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """

    def __init__(self,
                 model: nn.Module,
                 momentum: float = 0.0002,
                 gamma: int = 100,
                 interval: int = 1,
                 device: Optional[torch.device] = None,
                 update_buffers: bool = False) -> None:
        super().__init__(
            model=model,
            momentum=momentum,
            interval=interval,
            device=device,
            update_buffers=update_buffers)
        assert gamma > 0, f'gamma must be greater than 0, but got {gamma}'
        self.gamma = gamma

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> Tensor:
        """Compute the moving average of the parameters using the linear
        momentum strategy.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        Returns:
            Tensor: The averaged parameters.
        """
        momentum = max(self.momentum, self.gamma / (self.gamma + self.steps))
        return averaged_param * (1 - momentum) + source_param * momentum
