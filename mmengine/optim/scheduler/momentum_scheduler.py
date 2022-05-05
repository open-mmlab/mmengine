# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import PARAM_SCHEDULERS
from .param_scheduler import (ConstantParamScheduler,
                              CosineAnnealingParamScheduler,
                              ExponentialParamScheduler, LinearParamScheduler,
                              MultiStepParamScheduler, PolyParamScheduler,
                              StepParamScheduler)


class MomentumSchedulerMixin:
    """A mixin class for momentum schedulers."""

    def __init__(self, optimizer, *args, **kwargs):
        super().__init__(optimizer, 'momentum', *args, **kwargs)

    @classmethod
    def build_iter_based(cls, optimizer, *args, **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        return super().build_iter_based(optimizer, 'momentum', *args, **kwargs)


@PARAM_SCHEDULERS.register_module()
class ConstantMomentum(MomentumSchedulerMixin, ConstantParamScheduler):
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


@PARAM_SCHEDULERS.register_module()
class CosineAnnealingMomentum(MomentumSchedulerMixin,
                              CosineAnnealingParamScheduler):
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


@PARAM_SCHEDULERS.register_module()
class ExponentialMomentum(MomentumSchedulerMixin, ExponentialParamScheduler):
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


@PARAM_SCHEDULERS.register_module()
class LinearMomentum(MomentumSchedulerMixin, LinearParamScheduler):
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


@PARAM_SCHEDULERS.register_module()
class MultiStepMomentum(MomentumSchedulerMixin, MultiStepParamScheduler):
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


@PARAM_SCHEDULERS.register_module()
class StepMomentum(MomentumSchedulerMixin, StepParamScheduler):
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


@PARAM_SCHEDULERS.register_module()
class PolyMomentum(MomentumSchedulerMixin, PolyParamScheduler):
    """Decays the momentum of each parameter group in a polynomial decay
    scheme.

    Notice that such decay can happen simultaneously with other changes to the
    parameter value from outside this scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        eta_min (float): Minimum momentum at the end of scheduling.
            Defaults to 0.
        power (float): The power of the polynomial. Defaults to 1.0.
        begin (int): Step at which to start updating the parameters.
            Defaults to 0.
        end (int): Step at which to stop updating the parameters.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.
    """
