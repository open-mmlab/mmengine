# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
import weakref
from collections import Counter
from functools import wraps
from typing import Callable, List

from torch.optim import Optimizer

from mmengine.registry import PARAM_SCHEDULERS

INF = int(1e9)


class _ParamScheduler:
    """Base class for parameter schedulers.

    It should be inherited by all schedulers that schedule parameters in the
    optimizer's ``param_groups``. All subclasses should overwrite the
    ``_get_value()`` according to their own schedule strategy.
    The implementation is motivated by
    https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        begin (int): Step at which to start updating the parameters.
            Defaults to 0.
        end (int): Step at which to stop updating the parameters.
            Defaults to INF.
        last_step (int): The index of last step. Used for resuming without
            state dict. Default value ``-1`` means the ``step`` function is
            never be called before. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.
    """  # noqa: E501

    def __init__(self,
                 optimizer: Optimizer,
                 param_name: str,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('``optimizer`` should be an Optimizer,'
                            'but got {}'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.param_name = param_name

        if end <= begin:
            raise ValueError('end should be larger than begin, but got'
                             ' begin={}, end={}'.format(begin, end))
        self.begin = begin
        self.end = end

        self.by_epoch = by_epoch

        assert isinstance(last_step, int) and last_step >= -1
        # Initialize valid step count and base values
        if last_step == -1:
            for group in optimizer.param_groups:
                # If the param is never be scheduled, record the current value
                # as the initial value.
                group.setdefault(f'initial_{param_name}', group[param_name])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if f'initial_{param_name}' not in group:
                    raise KeyError(
                        f"param 'initial_{param_name}' is not specified "
                        'in param_groups[{}] when resuming an optimizer'.
                        format(i))
        self.base_values = [
            group[f'initial_{param_name}'] for group in optimizer.param_groups
        ]
        self.last_step = last_step

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method: Callable):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)  # type: ignore
            # Get the unbound method for the same purpose.
            func = method.__func__  # type: ignore
            cls = instance_ref().__class__  # type: ignore
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._global_step += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True  # type: ignore
            return wrapper

        # add counter to optimizer
        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._global_step = -1

        self._global_step = -1
        self.verbose = verbose

        self.step()

    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which is not
        the optimizer.

        Returns:
            dict: scheduler state.
        """
        return {
            key: value
            for key, value in self.__dict__.items() if key != 'optimizer'
        }

    def load_state_dict(self, state_dict: dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_value(self):
        """Return the last computed value by current scheduler.

        Returns:
            list: A list of the last computed value of the optimizer's
            ``param_group``.
        """
        return self._last_value

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        raise NotImplementedError

    def print_value(self, is_verbose: bool, group: int, value: float):
        """Display the current parameter value.

        Args:
            is_verbose (bool): Whether to print the value.
            group (int): The index of the current ``param_group``.
            value (float): The parameter value.
        """
        if is_verbose:
            print('Adjusting parameter value'
                  ' of group {} to {:.4e}.'.format(group, value))

    def step(self):
        """Adjusts the parameter value of each parameter group based on the
        specified schedule."""
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._global_step == 0:
            if not hasattr(self.optimizer.step, '_with_counter'):
                warnings.warn(
                    'Seems like `optimizer.step()` has been overridden after'
                    'parameter value scheduler initialization. Please, make'
                    'sure to call `optimizer.step()` before'
                    '`scheduler.step()`. See more details at'
                    'https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate',  # noqa: E501
                    UserWarning)

            # Just check if there were two first scheduler.step() calls
            # before optimizer.step()
            elif self.optimizer._global_step < 0:
                warnings.warn(
                    'Detected call of `scheduler.step()` before'
                    '`optimizer.step()`. In PyTorch 1.1.0 and later, you'
                    'should call them in the opposite order: '
                    '`optimizer.step()` before `scheduler.step()`. '
                    'Failure to do this will result in PyTorch skipping '
                    'the first value of the parameter value schedule. '
                    'See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate',  # noqa: E501
                    UserWarning)
        self._global_step += 1

        # Compute parameter value per param group in the effective range
        if self.begin <= self._global_step < self.end:
            self.last_step += 1
            values = self._get_value()

            for i, data in enumerate(zip(self.optimizer.param_groups, values)):
                param_group, value = data
                param_group[self.param_name] = value
                self.print_value(self.verbose, i, value)

        self._last_value = [
            group[self.param_name] for group in self.optimizer.param_groups
        ]


@PARAM_SCHEDULERS.register_module()
class StepParamScheduler(_ParamScheduler):
    """Decays the parameter value of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the parameter value from outside this scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of parameter value decay.
        gamma (float): Multiplicative factor of parameter value decay.
            Defaults to 0.1.
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

    def __init__(self,
                 optimizer: Optimizer,
                 param_name: str,
                 step_size: int,
                 gamma: float = 0.1,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(
            optimizer=optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    def _get_value(self):
        if (self.last_step == 0) or (self.last_step % self.step_size != 0):
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]
        return [
            group[self.param_name] * self.gamma
            for group in self.optimizer.param_groups
        ]


@PARAM_SCHEDULERS.register_module()
class MultiStepParamScheduler(_ParamScheduler):
    """Decays the specified parameter in each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the parameter from outside this
    scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of parameter value decay.
            Defaults to 0.1.
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

    def __init__(self,
                 optimizer: Optimizer,
                 param_name: str,
                 milestones: List[int],
                 gamma: float = 0.1,
                 last_step: int = -1,
                 begin: int = 0,
                 end: int = INF,
                 by_epoch: bool = True,
                 verbose: bool = False):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super().__init__(
            optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    def _get_value(self):
        if self.last_step not in self.milestones:
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]
        return [
            group[self.param_name] *
            self.gamma**self.milestones[self.last_step]
            for group in self.optimizer.param_groups
        ]


@PARAM_SCHEDULERS.register_module()
class ConstantParamScheduler(_ParamScheduler):
    """Decays the parameter value of each parameter group by a small constant
    factor until the number of epoch reaches a pre-defined milestone: ``end``.
    Notice that such decay can happen simultaneously with other changes to the
    parameter value from outside this scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): The number we multiply parameter value until the
            milestone. Defaults to 1./3.
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

    def __init__(self,
                 optimizer: Optimizer,
                 param_name: str,
                 factor: float = 1.0 / 3,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        if factor > 1.0 or factor < 0:
            raise ValueError(
                'Constant multiplicative factor should between 0 and 1.')

        self.factor = factor
        self.total_iters = end - begin - 1
        super().__init__(
            optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    def _get_value(self):
        if self.last_step == 0:
            return [
                group[self.param_name] * self.factor
                for group in self.optimizer.param_groups
            ]

        if (self.last_step > self.total_iters
                or (self.last_step != self.total_iters)):
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]

        if self.last_step == self.total_iters:
            return [
                group[self.param_name] * (1.0 / self.factor)
                for group in self.optimizer.param_groups
            ]


@PARAM_SCHEDULERS.register_module()
class ExponentialParamScheduler(_ParamScheduler):
    """Decays the parameter value of each parameter group by gamma every epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of parameter value decay.
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

    def __init__(self,
                 optimizer: Optimizer,
                 param_name: str,
                 gamma: float,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        self.gamma = gamma
        super().__init__(
            optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    def _get_value(self):
        if self.last_step == 0:
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]
        return [
            group[self.param_name] * self.gamma
            for group in self.optimizer.param_groups
        ]


@PARAM_SCHEDULERS.register_module()
class CosineAnnealingParamScheduler(_ParamScheduler):
    r"""Set the parameter value of each parameter group using a cosine annealing
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
    is defined recursively, the parameter value can be simultaneously modified
    outside this scheduler by other operators. If the parameter value is set
    solely by this scheduler, the parameter value at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this
    only implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum parameter value. Defaults to 0.
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

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self,
                 optimizer: Optimizer,
                 param_name: str,
                 T_max: int,
                 eta_min: float = 0.,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(
            optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    def _get_value(self):
        if self.last_step == 0:
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]
        elif (self.last_step - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group[self.param_name] + (base_value - self.eta_min) *
                (1 - math.cos(math.pi / self.T_max)) / 2
                for base_value, group in zip(self.base_values,
                                             self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * self.last_step / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_step - 1) / self.T_max)) *
                (group[self.param_name] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]


@PARAM_SCHEDULERS.register_module()
class LinearParamScheduler(_ParamScheduler):
    """Decays the parameter value of each parameter group by linearly changing
    small multiplicative factor until the number of epoch reaches a pre-defined
    milestone: ``end``.

    Notice that such decay can happen simultaneously with other changes to the
    parameter value from outside this scheduler.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply parameter value in the
            first epoch. The multiplication factor changes towards end_factor
            in the following epochs. Defaults to 1./3.
        end_factor (float): The number we multiply parameter value at the end
            of linear changing process. Defaults to 1.0.
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

    def __init__(self,
                 optimizer: Optimizer,
                 param_name: str,
                 start_factor: float = 1.0 / 3,
                 end_factor: float = 1.0,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError(
                'Starting multiplicative factor should between 0 and 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError(
                'Ending multiplicative factor should between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = end - begin - 1
        super().__init__(
            optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    def _get_value(self):

        if self.last_step == 0:
            return [
                group[self.param_name] * self.start_factor
                for group in self.optimizer.param_groups
            ]

        return [
            group[self.param_name] *
            (1. + (self.end_factor - self.start_factor) /
             (self.total_iters * self.start_factor + (self.last_step - 1) *
              (self.end_factor - self.start_factor)))
            for group in self.optimizer.param_groups
        ]
