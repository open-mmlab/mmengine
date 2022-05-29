# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Iterator, Tuple

from .optimizer_wrapper import OptimizerWrapper


class OptimizerWrapperDict:
    """A dictionary container of :obj:`OptimizerWrapper`.

    An :obj:`OptimizerWrapperDict` instance is composed by a dictionary of
    :obj:`OptimizerWrapper`, and can be accessed just like a dictionary. All
    optimizer wrapper instance should have the same type and all
    instance in the dictionary will call corresponding methods when
    :meth:`step`, :meth:`zero_gard`, and :meth:`update_params` of
    :obj:`OptimizerWrapper` is called.

    Warning:
        :obj:`OptimizerWrapperDict` will only call :meth:`backward` of the
        first instance in dictionary. Since the gradient back propagation
        process is only related to the ``loss`` itself, but not to the
        optimizer.

    Warning:
        :obj:`OptimizerWrapperDict` cannot enable a gradient accumulation
        context. The gradient accumulation should be
        enabled by accessing the corresponding :obj:`OptimizerWrapper`
        instance and call its :meth:`accumulate_grad`.

    Args:
        optimizer_wrappers (Dict[str, OptimizerWrapper]): A dictionary of
            ``OptimizerWrapper`` instance.
    """

    def __init__(self, optimizer_wrappers: Dict[str, OptimizerWrapper]):
        first_key = next(iter(optimizer_wrappers))
        first_optim_wrapper = optimizer_wrappers[first_key]
        optim_class = type(first_optim_wrapper)
        for key, value in optimizer_wrappers.items():
            assert type(value) == optim_class, (
                f'All optimizer wrappers should have the same type, but found'
                f' {key}: {type(value)} and {first_key}: {optim_class}')
            if value.accumulative_iters != 1:
                warnings.warn(
                    f'The `accumulative_iters` of {key} is '
                    f'{value.accumulative_iters}. OptimizerWrapperDict '
                    f'will not enable any `accumulate_grad` context of its '
                    f'optimizer wrappers. You should access the corresponding '
                    f'optimizer wrapper to enable the context.')
        self.optimizer_wrappers = optimizer_wrappers

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state dictionary from the ``state_dict``.

        Args:
            state_dict (dict): Each key-value pair in `state_dict` represents
                the name and the state dictionary of corresponding
                :obj:`OptimizerWrapper`.
        """
        for name, _state_dict in state_dict.items():
            assert name in self.optimizer_wrappers, (
                f'Mismatched `state_dict`! cannot found {name} in '
                'OptimizerWrapperDict')
            self.optimizer_wrappers[name].load_state_dict(_state_dict)

    def state_dict(self) -> dict:
        """Get the state dictionary of all optimizer wrappers.

        Returns:
            dict: Each key-value pair in the dictionary represents the name
            and state dictionary of corresponding :obj:`OptimizerWrapper`.
        """
        state_dict = dict()
        for name, optim_wrapper in self.optimizer_wrappers.items():
            state_dict[name] = optim_wrapper.state_dict()
        return state_dict

    def items(self) -> Iterator[Tuple[str, OptimizerWrapper]]:
        """A generator to get the name and corresponding
        :obj:`OptimizerWrapper`"""
        yield from self.optimizer_wrappers.items()

    def values(self) -> Iterator[OptimizerWrapper]:
        """A generator to get :obj:`OptimizerWrapper`"""
        yield from self.optimizer_wrappers.values()

    def keys(self) -> Iterator[str]:
        """A generator to get the name of :obj:`OptimizerWrapper`"""
        yield from self.optimizer_wrappers.keys()

    def __getitem__(self, key: str) -> OptimizerWrapper:
        assert key in self.optimizer_wrappers, (
            f'Cannot find {key} in OptimizerWrapperDict, please check '
            'your optimizer constructor.')
        return self.optimizer_wrappers[key]

    def __contains__(self, key: str) -> bool:
        return key in self.optimizer_wrappers

    def __len__(self) -> int:
        return len(self.optimizer_wrappers)

    def __repr__(self) -> str:
        desc = ''
        for name, optim_wrapper in self.optimizer_wrappers.items():
            desc += f'name: {name}\n'
            desc += repr(optim_wrapper)
        return desc
