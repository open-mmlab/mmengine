# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Sequence, Tuple, Union

from torch.utils.data import DataLoader

from mmengine.data import BaseDataSample
from mmengine.evaluator import BaseEvaluator, build_evaluator
from mmengine.registry import LOOPS
from mmengine.utils import is_list_of
from .base_loop import BaseLoop


@LOOPS.register_module()
class EpochBasedTrainLoop(BaseLoop):
    """Loop for epoch-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): An iterator to generate one batch of
            training dataset each iteration.
        max_epoch (int): Total training epochs.
    """

    def __init__(self, runner, dataloader: Union[DataLoader, Dict],
                 max_epochs: int) -> None:
        super().__init__(runner, dataloader)
        self._max_epochs = max_epochs

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')

        while self.runner.epoch < self._max_epochs:
            self.run_epoch()

            if (self.runner.val_loop is not None and
                    self.runner.epoch % self.runner.val_loop.interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self.runner._epoch += 1

    def run_iter(self, idx,
                 data_batch: Sequence[Tuple[Any, BaseDataSample]]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[Tuple[Any, BaseDataSample]]): Batch of data
                from dataloader.
        """
        self.runner._inner_iter = idx

        self.runner.call_hook('before_train_iter', data_batch=data_batch)
        outputs = self.runner.model.train_step(data_batch)
        self.runner.call_hook(
            'after_train_iter', data_batch=data_batch, outputs=outputs)

        self.runner._iter += 1


@LOOPS.register_module()
class IterBasedTrainLoop(BaseLoop):
    """Loop for iter-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): An iterator to generate one batch of
            training dataset each iteration.
        max_iter (int): Total training iterations.
    """

    def __init__(self, runner, dataloader: Union[DataLoader, Dict],
                 max_iters: int) -> None:
        super().__init__(runner, dataloader)
        self._max_iters = max_iters

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')

        while self.runner._iter < self._max_iters:
            data_batch = next(self.dataloader)
            self.run_iter(data_batch)

            if (self.runner.val_loop is not None and
                    self.runner._iter % self.runner.val_loop.interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')

    def run_iter(self, data_batch: Sequence[Tuple[Any,
                                                  BaseDataSample]]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[Tuple[Any, BaseDataSample]]): Batch of data
                from dataloader.
        """
        self.runner.call_hook('before_train_iter', data_batch=data_batch)
        outputs = self.runner.model.train_step(data_batch)
        self.runner.call_hook(
            'after_train_iter', data_batch=data_batch, outputs=outputs)
        self.runner._iter += 1


@LOOPS.register_module()
class ValLoop(BaseLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict):  An iterator to generate one batch of
            validation dataset each iteration.
        evaluator (BaseEvaluator or dict or list): Used for computing metrics.
        interval (int): Validation interval. Defaults to 1.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[BaseEvaluator, Dict, List],
                 interval: int = 1) -> None:
        super().__init__(runner, dataloader)

        if isinstance(evaluator, dict) or is_list_of(evaluator, dict):
            self.evaluator = build_evaluator(evaluator)  # type: ignore
        else:
            self.evaluator = evaluator  # type: ignore

        self.interval = interval

    def run(self):
        """Launch validation."""
        self.runner.call_hook('before_val')

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        self.evaluator.evaluate(len(self.dataloader.dataset))

        self.runner.call_hook('after_val')

    def run_iter(self, idx, data_batch: Sequence[Tuple[Any, BaseDataSample]]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[Tuple[Any, BaseDataSample]]): Batch of data
                from dataloader.
        """
        self.runner.inner_iter = idx
        self.runner.call_hook('before_val_iter', data_batch=data_batch)
        outputs = self.runner.model(data_batch)
        self.evaluator.process(data_batch, outputs)
        self.runner.call_hook(
            'after_val_iter', data_batch=data_batch, outputs=outputs)


@LOOPS.register_module()
class TestLoop(BaseLoop):
    """Loop for test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict):  An iterator to generate one batch of
            test dataset each iteration.
        evaluator (BaseEvaluator or dict or list): Used for computing metrics.
    """

    def __init__(self, runner, dataloader: Union[DataLoader, Dict],
                 evaluator: Union[BaseEvaluator, Dict, List]):
        super().__init__(runner, dataloader)

        if isinstance(evaluator, dict) or is_list_of(evaluator, dict):
            self.evaluator = build_evaluator(evaluator)  # type: ignore
        else:
            self.evaluator = evaluator  # type: ignore

    def run(self) -> None:
        """Launch test."""
        self.runner.call_hook('before_test')

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        self.evaluator.evaluate(len(self.dataloader.dataset))

        self.runner.call_hook('after_test')

    def run_iter(self, idx,
                 data_batch: Sequence[Tuple[Any, BaseDataSample]]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[Tuple[Any, BaseDataSample]]): Batch of data
                from dataloader.
        """
        self.runner.inner_iter = idx
        self.runner.call_hook('before_test_iter', data_batch=data_batch)
        outputs = self.runner.model(data_batch)
        self.evaluator.process(data_batch, outputs)
        self.runner.call_hook(
            'after_test_iter', databatch=data_batch, outputs=outputs)
