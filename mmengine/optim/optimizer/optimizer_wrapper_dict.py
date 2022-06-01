# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from contextlib import ExitStack, contextmanager
from typing import Dict, Iterator, List, Tuple

import torch
import torch.nn as nn

from .optimizer_wrapper import OptimWrapper


class OptimWrapperDict(OptimWrapper):
    """A dictionary container of :obj:`OptimWrapper`.

    If runner is training with multiple optimizers, all optimizer wrappers
    should be managed by :obj:`OptimWrapperDict` which is built by
    ``CustomOptimWrapperConstructor``. ``OptimWrapperDict`` will load and save
    the state dictionary of all optimizer wrappers.

    Consider the semantic ambiguity of calling :meth:``update_params``,
    :meth:`backward` of all optimizer wrappers, ``OptimWrapperDict`` will not
    implement these methods.

    Examples:
        >>> import torch.nn as nn
        >>> from torch.optim import SGD
        >>> from mmengine.optim import OptimWrapperDict, OptimWrapper
        >>> model1 = nn.Linear(1, 1)
        >>> model2 = nn.Linear(1, 1)
        >>> optim_wrapper1 = OptimWrapper(SGD(model1.parameters(), lr=0.1))
        >>> optim_wrapper2 = OptimWrapper(SGD(model2.parameters(), lr=0.1))
        >>> optim_wrapper_dict = OptimWrapperDict(model1=optim_wrapper1,
        >>>                                       model2=optim_wrapper2)

    Note:
        The optimizer wrapper contained in ``OptimWrapperDict`` can be accessed
        in the same way as `dict`.

    Args:
        **optim_wrappers: A dictionary of ``OptimWrapper`` instance.
    """

    def __init__(self, **optim_wrapper_dict: OptimWrapper):
        first_key = next(iter(optim_wrapper_dict))
        first_optim_wrapper = optim_wrapper_dict[first_key]
        assert isinstance(first_optim_wrapper, OptimWrapper), (
            'Each argument of `OptimWrapperDict` must be an `OptimWrapper '
            'instance`')
        optim_wrapper_class = type(first_optim_wrapper)
        for key, value in optim_wrapper_dict.items():
            assert type(value) == optim_wrapper_class, (
                f'All optimizer wrappers should have the same type, but found'
                f' {key}: {type(value)} and {first_key}: {optim_wrapper_class}'
            )
            if value.accumulative_iters != 1:
                warnings.warn(
                    f'The `accumulative_iters` of {key} is '
                    f'{value.accumulative_iters}. OptimWrapperDict '
                    'will not enable any `accumulate_grad` context of its '
                    'optimizer wrappers. You should access the corresponding '
                    'optimizer wrapper to enable the context.')
        self.optim_wrappers = optim_wrapper_dict

    def update_params(self, loss: torch.Tensor) -> None:
        """Update all optimizer wrappers would lead to a duplicate backward
        errors, and OptimWrapperDict does not know which optimizer wrapper
        should be updated.

        Therefore, this method is not implemented. The optimizer wrapper of
        OptimWrapperDict should be accessed and call its `update_params.
        """
        raise NotImplementedError(
            'You should access the OptimWrapper of the '
            'OptimWrapperDict and call its `update_params`')

    def backward(self, loss: torch.Tensor) -> None:
        """Since OptimWrapperDict doesn't know which optimizer wrapper's
        backward method should be called (``loss_scaler`` maybe different in
        different :obj:AmpOptimWrapper), this method is not implemented.

        The optimizer wrapper of OptimWrapperDict should be accessed and call
        its `backward.
        """
        raise NotImplementedError('You should access the OptimWrapper of the '
                                  'OptimWrapperDict and call its `backward`')

    def step(self) -> None:
        """Since the backward method is not implemented, the step should not be
        implemented either."""
        raise NotImplementedError('You should access the OptimWrapper of the '
                                  'OptimWrapperDict and call its `step`')

    def zero_grad(self) -> None:
        """Set the gradients of all optimizer wrappers to zero."""
        for optim_wrapper in self.optim_wrappers.values():
            optim_wrapper.zero_grad()

    @contextmanager
    def precision_context(self):
        optim_wrapper = next(iter(self.optim_wrappers.values()))
        with optim_wrapper.precision_context():
            yield

    @contextmanager
    def accumulate_grad(self, model: nn.Module, cur_iter: int, max_iters: int):
        """Enable ``accumulate_grad`` contexts of all optimizer wrappers.

        Warning:
            Consider there is only one ``model`` arguments for all
            optimizer wrappers, all optimizer wrappers are working under the
            same ``model.no_sync`` context. For example, there is a model
            composed of model_a(optimizer_a) and model_b(optimizer_b).
            ``OptimWrapperDict.accumulate_grad`` will further
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
            for optim_wrapper in self.optim_wrappers.values():
                stack.enter_context(
                    optim_wrapper.accumulate_grad(model, cur_iter, max_iters))
            yield

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state dictionary from the ``state_dict``.

        Args:
            state_dict (dict): Each key-value pair in `state_dict` represents
                the name and the state dictionary of corresponding
                :obj:`OptimWrapper`.
        """
        for name, _state_dict in state_dict.items():
            assert name in self.optim_wrappers, (
                f'Mismatched `state_dict`! cannot found {name} in '
                'OptimWrapperDict')
            self.optim_wrappers[name].load_state_dict(_state_dict)

    def get_lr(self) -> Dict[str, List[float]]:
        """Get the learning rate of all optimizers.

        Returns:
            Dict[str, List[float]]: Learning rate of all optimizers.
        """
        lr_dict = dict()
        for name, optim_wrapper in self.optim_wrappers.items():
            lr_dict[f'{name}.lr'] = optim_wrapper.get_lr()['lr']
        return lr_dict

    def get_momentum(self) -> Dict[str, List[float]]:
        """Get the momentum of all optimizers.

        Returns:
            Dict[str, List[float]]: momentum of all optimizers.
        """
        momentum_dict = dict()
        for name, optim_wrapper in self.optim_wrappers.items():
            momentum_dict[f'{name}.momentum'] = optim_wrapper.get_momentum(
            )['momentum']
        return momentum_dict

    def state_dict(self) -> dict:
        """Get the state dictionary of all optimizer wrappers.

        Returns:
            dict: Each key-value pair in the dictionary represents the name
            and state dictionary of corresponding :obj:`OptimWrapper`.
        """
        state_dict = dict()
        for name, optim_wrapper in self.optim_wrappers.items():
            state_dict[name] = optim_wrapper.state_dict()
        return state_dict

    def items(self) -> Iterator[Tuple[str, OptimWrapper]]:
        """A generator to get the name and corresponding
        :obj:`OptimWrapper`"""
        yield from self.optim_wrappers.items()

    def values(self) -> Iterator[OptimWrapper]:
        """A generator to get :obj:`OptimWrapper`"""
        yield from self.optim_wrappers.values()

    def keys(self) -> Iterator[str]:
        """A generator to get the name of :obj:`OptimWrapper`"""
        yield from self.optim_wrappers.keys()

    def __getitem__(self, key: str) -> OptimWrapper:
        assert key in self.optim_wrappers, (
            f'Cannot find {key} in OptimWrapperDict, please check '
            'your optimizer constructor.')
        return self.optim_wrappers[key]

    def __contains__(self, key: str) -> bool:
        return key in self.optim_wrappers

    def __len__(self) -> int:
        return len(self.optim_wrappers)

    def __repr__(self) -> str:
        desc = ''
        for name, optim_wrapper in self.optim_wrappers.items():
            desc += f'name: {name}\n'
            desc += repr(optim_wrapper)
        return desc
