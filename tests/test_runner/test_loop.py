# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mmengine.runner.loop import (EpochBasedTrainLoop, IterBasedTrainLoop,
                                  TestLoop, ValLoop)


class ToyDataset(Dataset):
    META = dict()  # type: ignore
    data = np.zeros((30, 1, 1, 1))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index])


class TestLoops(TestCase):

    def setUp(self) -> None:
        self.runner = Mock()
        self.runner.call_hooks = Mock()
        self.runner.model = Mock()
        self.runner.epoch = 0
        self.runner.iter = 0
        self.runner.inner_iter = 0
        self.runner.model.train_step = Mock()
        self.runner.model.val_step = Mock()

        self.evaluator = Mock()
        self.evaluator.process = Mock()
        self.evaluator.evaluate = Mock()

    def test_epoch_based_train_loop(self):
        train_loop = EpochBasedTrainLoop(
            runner=self.runner, loader=DataLoader(ToyDataset()), max_epoch=3)
        train_loop.run()
        assert train_loop.runner.epoch == 3
        assert train_loop.runner.iter == 90

    def test_iter_based_train_loop(self):
        train_loop = IterBasedTrainLoop(
            runner=self.runner, loader=DataLoader(ToyDataset()), max_iter=25)
        train_loop.run()
        assert train_loop.runner.epoch == 0
        assert train_loop.runner.iter == 25

    def test_val_loop(self):
        val_loop = ValLoop(
            runner=self.runner,
            loader=DataLoader(ToyDataset()),
            evaluator=self.evaluator)
        val_loop.run()

    def test_test_loop(self):
        test_loop = TestLoop(
            runner=self.runner,
            loader=DataLoader(ToyDataset()),
            evaluator=self.evaluator)
        test_loop.run()
