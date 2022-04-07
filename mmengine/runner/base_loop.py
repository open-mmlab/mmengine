# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, Union

from torch.utils.data import DataLoader


class BaseLoop(metaclass=ABCMeta):
    """Base loop class.

    All subclasses inherited from ``BaseLoop`` should overwrite the
    :meth:`run` method.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
    """

    def __init__(self, runner, dataloader: Union[DataLoader, Dict]) -> None:
        self._runner = runner

        if isinstance(dataloader, dict):
            self.dataloader = runner.build_dataloader(dataloader)
        else:
            self.dataloader = dataloader

    @property
    def runner(self):
        return self._runner

    @abstractmethod
    def run(self) -> None:
        """Execute loop."""
