# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Sequence, Union

import torch
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.registry import LOOPS
from mmengine.utils import is_list_of
from .base_loop import BaseLoop


@LOOPS.register_module()
class EpochBasedTrainLoop(BaseLoop):
    """Loop for epoch-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 max_epochs: int,
                 val_begin: int = 1,
                 val_interval: int = 1) -> None:
        super().__init__(runner, dataloader)
        self._max_epochs = max_epochs
        self._max_iters = max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            warnings.warn(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.')

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')

        while self._epoch < self._max_epochs:
            self.run_epoch()

            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        with self.runner.optim_wrapper.accumulate_grad(self.runner.model,
                                                       self._iter,
                                                       self._max_iters):
            # outputs should be a dict of loss.
            outputs = self.runner.model.train_step(
                data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1


@LOOPS.register_module()
class IterBasedTrainLoop(BaseLoop):
    """Loop for iter-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_iters (int): Total training iterations.
        val_begin (int): The iteration that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1000.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 max_iters: int,
                 val_begin: int = 1,
                 val_interval: int = 1000) -> None:
        super().__init__(runner, dataloader)
        self._max_iters = max_iters
        self._max_epochs = 1  # for compatibility with EpochBasedTrainLoop
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            warnings.warn(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.')
        # get the iterator of the dataloader
        self.dataloader_iterator = iter(self.dataloader)

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')
        while self._iter < self._max_iters:
            self.runner.model.train()

            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and self._iter % self.val_interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')

    def run_iter(self, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=self._iter, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        with self.runner.optim_wrapper.accumulate_grad(self.runner.model,
                                                       self._iter,
                                                       self._max_iters):
            # train_logs should be a dict of loss.
            train_logs = self.runner.model.train_step(
                data_batch, optim_wrapper=self.runner.optim_wrapper)
        self.runner.message_hub.update_info('train_logs', train_logs)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=self._iter,
            data_batch=data_batch,
            outputs=train_logs)
        self._iter += 1


@LOOPS.register_module()
class ValLoop(BaseLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
    """

    def __init__(self, runner, dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List]) -> None:
        super().__init__(runner, dataloader)

        if isinstance(evaluator, dict) or is_list_of(evaluator, dict):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            warnings.warn(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.')

    def run(self):
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        outputs = self.runner.model.val_step(data_batch)
        self.evaluator.process(data_batch, outputs)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


@LOOPS.register_module()
class TestLoop(BaseLoop):
    """Loop for test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
    """

    def __init__(self, runner, dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List]):
        super().__init__(runner, dataloader)

        if isinstance(evaluator, dict) or is_list_of(evaluator, dict):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            warnings.warn(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.')

    def run(self) -> None:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        predictions = self.runner.model.test_step(data_batch)
        self.evaluator.process(data_batch, predictions)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=predictions)
