# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from contextlib import ExitStack, contextmanager
from typing import Dict, Iterator, Tuple

import torch.nn as nn

from .optimizer_wrapper import OptimizerWrapper


class OptimizerWrapperDict:
    """A dictionary container of :obj:`OptimizerWrapper`.

    If runner is training with multiple optimizers,
    ``runner.optimizer_wrapper`` is an ``OptimizerWrapperDict`` instance, which
    is built by ``CustomOptimizerWrapperConstructor``.

    The ``OptimizerWrapper`` instance contained in ``OptimizerWrapperDict``
    can be accessed in the same way as `dict`.

    Consider runner and loop will call the following methods of
    ``OptimizerWrapper``: ``state_dict``, ``load_state_dict`` and
    ``accumulate_grad``. ``OptimizerWrapperDict`` implements these methods to
    make it compatible with single optimizer training.

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

    @contextmanager
    def accumulate_grad(self, model: nn.Module, cur_iter: int, max_iters: int):
        """Enable ``accumulate_grad`` contexts of all optimizer wrappers.

        Warning:
            Consider there is only single ``model`` arguments for all
            optimizer wrappers, all optimizer wrappers are working under the
            same ``model.no_sync`` context. For example, there is a model
            composed of model_a(optimizer_a) and model_b(optimizer_b).
            ``OptimizerWrapperDict.accumulate_grad`` will further
            call ``model.no_sync``, which will block the gradient
            synchronization of both a and b. If optimizer_a and
            optimizer_b have different ``accumulative_iters``, and want to
            block the gradient synchronization of model_a and model_b
            separately, the model should not implement the ``no_sync``
            method(or enable an empty context). The ``accumulate_grad`` context
            should be enabled inside the model by accessing corresponding
            optimizer wrapper.
        """
        with ExitStack() as stack:
            for optim_wrapper in self.optimizer_wrappers.values():
                stack.enter_context(
                    optim_wrapper.accumulate_grad(model, cur_iter, max_iters))
            yield

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
