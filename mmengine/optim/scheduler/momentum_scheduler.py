# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch

from mmengine.registry import PARAM_SCHEDULERS
from .param_scheduler import (INF, ConstantParamScheduler,
                              CosineAnnealingParamScheduler,
                              ExponentialParamScheduler, LinearParamScheduler,
                              MultiStepParamScheduler, StepParamScheduler)


@PARAM_SCHEDULERS.register_module()
class ConstantMomentum(ConstantParamScheduler):
    """Decays the momentum value of each parameter group by a small constant
    factor until the number of epoch reaches a pre-defined milestone: ``end``.
    Notice that such decay can happen simultaneously with other changes to the
    momentum value from outside this scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): The number we multiply momentum until the milestone.
            Defaults to 1./3.
        begin (int): Step at which to start updating the momentum.
            Defaults to 0.
        end (int): Step at which to stop updating the momentum.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without state
            dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled momentum is updated by epochs.
            Defaults to True.
        verbose (bool): Whether to print the momentum for each update.
            Defaults to False.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 factor: float = 1.0 / 3,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        super().__init__(
            optimizer,
            param_name='momentum',
            factor=factor,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)


@PARAM_SCHEDULERS.register_module()
class CosineAnnealingMomentum(CosineAnnealingParamScheduler):
    r"""Set the momentum of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial value and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    Notice that because the schedule
    is defined recursively, the momentum can be simultaneously modified
    outside this scheduler by other operators. If the momentum is set
    solely by this scheduler, the momentum at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this
    only implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum momentum value. Defaults to 0.
        begin (int): Step at which to start updating the momentum.
            Defaults to 0.
        end (int): Step at which to stop updating the momentum.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled momentum is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the momentum for each update.
            Defaults to False.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: int = 0,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        super().__init__(
            optimizer,
            param_name='momentum',
            T_max=T_max,
            eta_min=eta_min,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)


@PARAM_SCHEDULERS.register_module()
class ExponentialMomentum(ExponentialParamScheduler):
    """Decays the momentum of each parameter group by gamma every epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of momentum value decay.
        begin (int): Step at which to start updating the momentum.
            Defaults to 0.
        end (int): Step at which to stop updating the momentum.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled momentum is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the momentum for each update.
            Defaults to False.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 gamma: float,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        super().__init__(
            optimizer,
            param_name='momentum',
            gamma=gamma,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)


@PARAM_SCHEDULERS.register_module()
class LinearMomentum(LinearParamScheduler):
    """Decays the momentum of each parameter group by linearly changing
    small multiplicative factor until the number of epoch reaches a pre-defined
    milestone: ``end``.

    Notice that such decay can happen simultaneously with other changes to the
    momentum from outside this scheduler.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply momentum in the
            first epoch. The multiplication factor changes towards end_factor
            in the following epochs. Defaults to 1./3.
        end_factor (float): The number we multiply momentum at the end
            of linear changing process. Defaults to 1.0.
        begin (int): Step at which to start updating the momentum.
            Defaults to 0.
        end (int): Step at which to stop updating the momentum.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled momentum is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the momentum for each update.
            Defaults to False.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 start_factor: float = 1.0 / 3,
                 end_factor: float = 1.0,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        super().__init__(
            optimizer,
            param_name='momentum',
            start_factor=start_factor,
            end_factor=end_factor,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)


@PARAM_SCHEDULERS.register_module()
class MultiStepMomentum(MultiStepParamScheduler):
    """Decays the specified momentum in each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the momentum from outside this
    scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of momentum value decay.
            Defaults to 0.1.
        begin (int): Step at which to start updating the momentum.
            Defaults to 0.
        end (int): Step at which to stop updating the momentum.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled momentum is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the momentum for each update.
            Defaults to False.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 milestones: List[int],
                 gamma: float = 0.1,
                 last_step: int = -1,
                 begin: int = 0,
                 end: int = INF,
                 by_epoch: bool = True,
                 verbose: bool = False):
        super().__init__(
            optimizer,
            param_name='momentum',
            milestones=milestones,
            gamma=gamma,
            last_step=last_step,
            begin=begin,
            end=end,
            by_epoch=by_epoch,
            verbose=verbose)


@PARAM_SCHEDULERS.register_module()
class StepMomentum(StepParamScheduler):
    """Decays the momentum of each parameter group by gamma every step_size
    epochs. Notice that such decay can happen simultaneously with other changes
    to the momentum from outside this scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of momentum value decay.
        gamma (float): Multiplicative factor of momentum value decay.
            Defaults to 0.1.
        begin (int): Step at which to start updating the momentum.
            Defaults to 0.
        end (int): Step at which to stop updating the momentum.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled momentum is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the momentum for each update.
            Defaults to False.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 step_size: int,
                 gamma: float = 0.1,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        super().__init__(
            optimizer,
            param_name='momentum',
            step_size=step_size,
            gamma=gamma,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)
