# Copyright (c) OpenMMLab. All rights reserved.
import types
import math
from torch._six import inf
from functools import wraps
import warnings
import weakref
from collections import Counter
from bisect import bisect_right

from torch.optim import Optimizer
import torch.optim.lr_scheduler


class _ParameterShceduler(object):
    def __init__(self,
                 optimizer,
                 param_name,
                 begin=0,
                 end=-1,
                 last_step=-1,
                 by_epoch=True,
                 verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.param_name = param_name
        if end != -1 and end <= begin:
            raise ValueError('end should be larger than begin')
        self.begin = begin
        self.end = end
        self.by_epoch = by_epoch
        # Initialize epoch and base learning rates
        if last_step == -1:
            for group in optimizer.param_groups:
                group.setdefault(f'initial_{param_name}', group[param_name])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if f'initial_{param_name}' not in group:
                    raise KeyError(f"param 'initial_{param_name}' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_values = [group[f'initial_{param_name}'] for group in optimizer.param_groups]
        self.last_step = last_step

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        # add counter to optimizer
        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0

        self._step_count = 0
        self.verbose = verbose

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_value(self):
        """ Return last computed value by current scheduler.
        """
        return self._last_value

    def _get_value(self):
        # Compute value using chainable form of the scheduler
        raise NotImplementedError

    def print_value(self, is_verbose, group, value):
        """Display the current learning rate.
        """
        if is_verbose:
            print('Adjusting learning rate'
                  ' of group {} to {:.4e}.'.format(group, value))

    def step(self):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1
        self.last_step += 1
        values = self._get_value()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, value = data
            param_group[self.param_name] = value
            self.print_value(self.verbose, i, value)

        self._last_value = [group[self.param_name] for group in self.optimizer.param_groups]


class StepParameterScheduler(_ParameterShceduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_step (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self,
                 optimizer,
                 param_name,
                 step_size,
                 gamma=0.1,
                 begin=0,
                 end=-1,
                 last_step=-1,
                 verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        super(StepParameterScheduler, self).__init__(optimizer=optimizer,
                                                     param_name=param_name,
                                                     begin=begin,
                                                     end=end,
                                                     last_step=last_step,
                                                     verbose=verbose)

    def _get_value(self):
        if (self.last_step == 0) or (self.last_step % self.step_size != 0):
            return [group[self.param_name] for group in self.optimizer.param_groups]
        return [group[self.param_name] * self.gamma
                for group in self.optimizer.param_groups]


class MultiStepParameterScheduler(_ParameterShceduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_step (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, param_name, milestones, gamma=0.1, last_step=-1, begin=0, end=-1, verbose=False):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super(MultiStepParameterScheduler, self).__init__(optimizer,
                                                          param_name=param_name,
                                                          begin=begin,
                                                          end=end,
                                                          last_step=last_step,
                                                          verbose=verbose)

    def _get_value(self):
        if self.last_step not in self.milestones:
            return [group[self.param_name] for group in self.optimizer.param_groups]
        return [group[self.param_name] * self.gamma ** self.milestones[self.last_step]
                for group in self.optimizer.param_groups]


class ConstantParameterScheduler(_ParameterShceduler):
    """Decays the learning rate of each parameter group by a small constant factor until the
    number of epoch reaches a pre-defined milestone: total_iters. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside this scheduler.
    When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): The number we multiply learning rate until the milestone. Default: 1./3.
        total_iters (int): The number of steps that the scheduler decays the learning rate.
            Default: 5.
        last_step (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025   if epoch == 0
        >>> # lr = 0.025   if epoch == 1
        >>> # lr = 0.025   if epoch == 2
        >>> # lr = 0.025   if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = ConstantLR(self.opt, factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, param_name, factor=1.0 / 3, begin=0, end=-1, last_step=-1, verbose=False):
        if factor > 1.0 or factor < 0:
            raise ValueError('Constant multiplicative factor expected to be between 0 and 1.')

        self.factor = factor
        self.total_iters = end - begin -1
        super(ConstantParameterScheduler, self).__init__(optimizer,
                                                         param_name=param_name,
                                                         begin=begin,
                                                         end=end,
                                                         last_step=last_step,
                                                         verbose=verbose)

    def _get_value(self):
        if self.last_step == 0:
            return [group[self.param_name] * self.factor for group in self.optimizer.param_groups]

        if (self.last_step > self.total_iters or
                (self.last_step != self.total_iters)):
            return [group[self.param_name] for group in self.optimizer.param_groups]

        if (self.last_step == self.total_iters):
            return [group[self.param_name] * (1.0 / self.factor) for group in self.optimizer.param_groups]


class ExponentialParameterScheduler(_ParameterShceduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_step (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, param_name, gamma, begin=0, end=-1, last_step=-1, verbose=False):
        self.gamma = gamma
        super(ExponentialParameterScheduler, self).__init__(optimizer,
                                                            param_name=param_name,
                                                            begin=begin,
                                                            end=end,
                                                            last_step=last_step,
                                                            verbose=verbose)

    def _get_value(self):
        if self.last_step == 0:
            return self.base_values
        return [group[self.param_name] * self.gamma
                for group in self.optimizer.param_groups]


class CosineAnnealingParameterScheduler(_ParameterShceduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
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

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_step (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, param_name, T_max, eta_min=0, begin=0, end=-1, last_step=-1, verbose=False):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingParameterScheduler, self).__init__(optimizer,
                                                                param_name=param_name,
                                                                begin=begin,
                                                                end=end,
                                                                last_step=last_step,
                                                                verbose=verbose)

    def _get_value(self):
        if self.last_step == 0:
            return self.base_values
        elif (self.last_step - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group[self.param_name] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_values, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_step / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_step - 1) / self.T_max)) *
                (group[self.param_name] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]


class LinearParameterScheduler(_ParameterShceduler):
    """Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_step (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025    if epoch == 0
        >>> # lr = 0.03125  if epoch == 1
        >>> # lr = 0.0375   if epoch == 2
        >>> # lr = 0.04375  if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = LinearLR(self.opt, start_factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, param_name, start_factor=1.0 / 3, end_factor=1.0, begin=0, end=-1, last_step=-1,
                 verbose=False):
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError('Starting multiplicative factor expected to be between 0 and 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError('Ending multiplicative factor expected to be between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = end - begin - 1
        super(LinearParameterScheduler, self).__init__(optimizer,
                                                       param_name=param_name,
                                                       begin=begin,
                                                       end=end,
                                                       last_step=last_step,
                                                       verbose=verbose)

    def _get_value(self):

        if self.last_step == 0:
            return [group[self.param_name] * self.start_factor for group in self.optimizer.param_groups]

        if self.last_step >= self.end:
            return [group[self.param_name] for group in self.optimizer.param_groups]

        return [group[self.param_name] * (1. + (self.end_factor - self.start_factor) /
                                          (self.total_iters * self.start_factor + (self.last_step - 1) * (self.end_factor - self.start_factor)))
                for group in self.optimizer.param_groups]
