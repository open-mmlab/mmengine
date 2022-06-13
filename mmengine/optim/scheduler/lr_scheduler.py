# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import PARAM_SCHEDULERS
from .param_scheduler import (ConstantParamScheduler,
                              CosineAnnealingParamScheduler,
                              ExponentialParamScheduler, LinearParamScheduler,
                              MultiStepParamScheduler, OneCycleParamScheduler,
                              PolyParamScheduler, StepParamScheduler)


class LRSchedulerMixin:
    """A mixin class for learning rate schedulers."""

    def __init__(self, optimizer, *args, **kwargs):
        super().__init__(optimizer, 'lr', *args, **kwargs)


@PARAM_SCHEDULERS.register_module()
class ConstantLR(LRSchedulerMixin, ConstantParamScheduler):
    """Decays the learning rate value of each parameter group by a small
    constant factor until the number of epoch reaches a pre-defined milestone:
    ``end``. Notice that such decay can happen simultaneously with other
    changes to the learning rate value from outside this scheduler.

    Args:
        optimizer (Optimizer or OptimWrapper): Wrapped optimizer.
        factor (float): The number we multiply learning rate until the
            milestone. Defaults to 1./3.
        begin (int): Step at which to start updating the learning rate.
            Defaults to 0.
        end (int): Step at which to stop updating the learning rate.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without state
            dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled learning rate is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the learning rate for each update.
            Defaults to False.
    """


@PARAM_SCHEDULERS.register_module()
class CosineAnnealingLR(LRSchedulerMixin, CosineAnnealingParamScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
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
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this
    only implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer or OptimWrapper): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Defaults to 0.
        begin (int): Step at which to start updating the learning rate.
            Defaults to 0.
        end (int): Step at which to stop updating the learning rate.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled learning rate is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the learning rate for each update.
            Defaults to False.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """


@PARAM_SCHEDULERS.register_module()
class ExponentialLR(LRSchedulerMixin, ExponentialParamScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.

    Args:
        optimizer (Optimizer or OptimWrapper): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        begin (int): Step at which to start updating the learning rate.
            Defaults to 0.
        end (int): Step at which to stop updating the learning rate.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled learning rate is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the learning rate for each update.
            Defaults to False.
    """


@PARAM_SCHEDULERS.register_module()
class LinearLR(LRSchedulerMixin, LinearParamScheduler):
    """Decays the learning rate of each parameter group by linearly changing
    small multiplicative factor until the number of epoch reaches a pre-defined
    milestone: ``end``.

    Notice that such decay can happen simultaneously with other changes to the
    learning rate from outside this scheduler.
    Args:
        optimizer (Optimizer or OptimWrapper): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the
            first epoch. The multiplication factor changes towards end_factor
            in the following epochs. Defaults to 1./3.
        end_factor (float): The number we multiply learning rate at the end
            of linear changing process. Defaults to 1.0.
        begin (int): Step at which to start updating the learning rate.
            Defaults to 0.
        end (int): Step at which to stop updating the learning rate.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled learning rate is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the learning rate for each update.
            Defaults to False.
    """


@PARAM_SCHEDULERS.register_module()
class MultiStepLR(LRSchedulerMixin, MultiStepParamScheduler):
    """Decays the specified learning rate in each parameter group by gamma once
    the number of epoch reaches one of the milestones. Notice that such decay
    can happen simultaneously with other changes to the learning rate from
    outside this scheduler.

    Args:
        optimizer (Optimizer or OptimWrapper): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Defaults to 0.1.
        begin (int): Step at which to start updating the learning rate.
            Defaults to 0.
        end (int): Step at which to stop updating the learning rate.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled learning rate is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the learning rate for each update.
            Defaults to False.
    """


@PARAM_SCHEDULERS.register_module()
class StepLR(LRSchedulerMixin, StepParamScheduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler.

    Args:
        optimizer (Optimizer or OptimWrapper): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Defaults to 0.1.
        begin (int): Step at which to start updating the learning rate.
            Defaults to 0.
        end (int): Step at which to stop updating the learning rate.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled learning rate is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the learning rate for each update.
            Defaults to False.
    """


@PARAM_SCHEDULERS.register_module()
class PolyLR(LRSchedulerMixin, PolyParamScheduler):
    """Decays the learning rate of each parameter group in a polynomial decay
    scheme.

    Notice that such decay can happen simultaneously with other changes to the
    parameter value from outside this scheduler.

    Args:
        optimizer (Optimizer or OptimWrapper): Wrapped optimizer.
        eta_min (float): Minimum learning rate at the end of scheduling.
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


@PARAM_SCHEDULERS.register_module()
class OneCycleLR(LRSchedulerMixin, OneCycleParamScheduler):
    r"""Sets the learning rate of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    The 1cycle learning rate policy changes the learning rate after every
    batch. `step` should be called after a batch has been used for training.

    This scheduler is not chainable.

    Note also that the total number of steps in the cycle can be determined in
    one of two ways (listed in order of precedence):

    #. A value for total_steps is explicitly provided.
    #. A number of epochs (epochs) and a number of steps per epoch
       (steps_per_epoch) are provided.
       In this case, the number of total steps is inferred by
       total_steps = epochs * steps_per_epoch

    You must either provide a value for total_steps or provide a value for both
    epochs and steps_per_epoch.

    The default behaviour of this scheduler follows the fastai implementation
    of 1cycle, which claims that "unpublished work has shown even better
    results by using only two phases". To mimic the behaviour of the original
    paper instead, set ``three_phase=True``.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        eta_max (float or list): Upper parameter value boundaries in the cycle
            for each parameter group.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by
            providing a value for epochs and steps_per_epoch.
            Default to None.
        pct_start (float): The percentage of the cycle (in number of steps)
            spent increasing the learning rate.
            Default to 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: "cos" for cosine annealing,
            "linear" for linear annealing.
            Default to 'cos'
        div_factor (float): Determines the initial learning rate via
            initial_param = eta_max/div_factor
            Default to 25
        final_div_factor (float): Determines the minimum learning rate via
            eta_min = initial_param/final_div_factor
            Default to 1e4
        three_phase (bool): If ``True``, use a third phase of the schedule to
            annihilate the learning rate according to 'final_div_factor'
            instead of modifying the second phase (the first two phases will be
            symmetrical about the step indicated by 'pct_start').
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.

    .. _Super-Convergence\: Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120
    """# noqa E501
