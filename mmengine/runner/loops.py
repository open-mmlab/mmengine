# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Union

from torch.utils.data import DataLoader

from mmengine.data import BaseDataSample
from mmengine.evaluator import BaseEvaluator, build_evaluator
from mmengine.utils import is_list_of
from .base_loop import BaseLoop


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
        self.runner.call_hooks('before_run')

        while self.runner.epoch < self._max_epochs:
            self.run_epoch()

            if (self.runner.val_loop is not None and
                    self.runner.epoch % self.runner.val_loop.interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hooks('after_run')

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hooks('before_train_epoch')

        for idx, data_batch in enumerate(self.dataloader):
            # data_batch is a tuple containing input and data_samples
            self.run_iter(idx, data_batch)

        self.runner.call_hooks('after_train_epoch')
        self.runner.epoch += 1

    def run_iter(self, idx, data_batch: BaseDataSample) -> None:
        """Iterate one batch.

        Args:
            data_batch (BaseDataSample): Batch of data from dataloader.
        """
        self.runner.inner_iter = idx

        self.runner.call_hooks('before_train_iter', data_batch=data_batch)
        outputs = self.runner.model.train_step(data_batch)
        self.runner.call_hooks(
            'after_train_iter', data_batch=data_batch, outputs=outputs)

        self.runner._iter += 1


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
        self.runner.call_hooks('before_run')
        self.runner.call_hooks('before_train_epoch')

        while self.runner._iter < self._max_iters:
            data_batch = next(self.dataloader)
            # data_batch is a tuple containing input and data_samples
            self.run_iter(data_batch)

            if (self.runner.val_loop is not None and
                    self.runner._iter % self.runner.val_loop.interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hooks('after_train_epoch')
        self.runner.call_hooks('after_run')

    def run_iter(self, data_batch: BaseDataSample) -> None:
        """Iterate one batch.

        Args:
            data_batch (BaseDataSample): Batch of data from dataloader.
        """
        self.runner.call_hooks('before_train_iter', data_batch=data_batch)
        outputs = self.runner.model.train_step(data_batch)
        self.runner.call_hooks(
            'after_train_iter', data_batch=data_batch, outputs=outputs)
        self.runner._iter += 1


class ValLoop(BaseLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict):  An iterator to generate one batch of
            validation dataset each iteration.
        evaluator (BaseEvaluator or dict or list): Used for computing metrics.
    """

    def __init__(self, runner, dataloader: Union[DataLoader, Dict],
                 evaluator: Union[BaseEvaluator, Dict, List]) -> None:
        super().__init__(runner, dataloader)

        if isinstance(evaluator, dict) or is_list_of(evaluator, dict):
            self.evaluator = build_evaluator(evaluator)  # type: ignore
        else:
            self.evaluator = evaluator  # type: ignore

    def run(self):
        """Launch validation."""
        self.runner.call_hooks('before_val_epoch')

        for idx, data_batch in enumerate(self.dataloader):
            # data_batch is a tuple containing input and data_samples
            self.run_iter(idx, data_batch)

        # compute metrics
        self.evaluator.evaluate(len(self.dataloader.dataset))

        self.runner.call_hooks('after_val_epoch')

    def run_iter(self, idx, data_batch: BaseDataSample):
        """Iterate one batch.

        Args:
            data_batch (BaseDataSample): Batch of data from dataloader.
        """
        self.runner.inner_iter = idx
        self.runner.call_hooks('before_val_iter', data_batch=data_batch)
        outputs = self.runner.model(data_batch)
        self.evaluator.process(data_batch, outputs)
        self.runner.call_hooks(
            'after_val_iter', data_batch=data_batch, outputs=outputs)


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
        self.runner.call_hooks('before_test_epoch')

        for idx, data_batch in enumerate(self.dataloader):
            # data_batch is a tuple containing input and data_samples
            self.run_iter(idx, data_batch)

        # compute metrics
        self.evaluator.evaluate(len(self.dataloader.dataset))

        self.runner.call_hooks('after_test_epoch')

    def run_iter(self, idx, data_batch: BaseDataSample) -> None:
        """Iterate one batch.

        Args:
            data_batch (BaseDataSample): Batch of data from dataloader.
        """
        self.runner.inner_iter = idx
        self.runner.call_hooks('before_test_iter', data_batch=data_batch)
        outputs = self.runner.model(data_batch)
        self.evaluator.process(data_batch, outputs)
        self.runner.call_hooks(
            'after_test_iter', databatch=data_batch, outputs=outputs)
