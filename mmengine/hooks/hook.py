# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Sequence, Tuple

from mmengine.data import BaseDataSample

DATA_BATCH = Optional[Sequence[Tuple[Any, BaseDataSample]]]


class Hook:
    """Base hook class.

    All hooks should inherit from this class.
    """

    priority = 'NORMAL'

    def before_run(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before the training process.

        Args:
            runner (Runner): The runner of the training process.
        """
        pass

    def after_run(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after the training process.

        Args:
            runner (Runner): The runner of the training process.
        """
        pass

    def before_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        pass

    def after_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        pass

    def before_iter(self, runner, data_batch: DATA_BATCH = None) -> None:
        """All subclasses should override this method, if they need any
        operations before each iter.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional):
                Data from dataloader. Defaults to None.
        """
        pass

    def after_iter(self,
                   runner,
                   data_batch: DATA_BATCH = None,
                   outputs: Optional[Sequence[BaseDataSample]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each epoch.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional):
                Data from dataloader. Defaults to None.
            outputs (Sequence[BaseDataSample], optional): Outputs from model.
                Defaults to None.
        """
        pass

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        """All subclasses should override this method, if they need any
        operations before saving the checkpoint.

        Args:
            runner (Runner): The runner of the training process.
            checkpoints (dict): Model's checkpoint.
        """
        pass

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        """All subclasses should override this method, if they need any
        operations after loading the checkpoint.

        Args:
            runner (Runner): The runner of the training process.
            checkpoints (dict): Model's checkpoint.
        """
        pass

    def before_train_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self.before_epoch(runner)

    def before_val_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before each validation epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self.before_epoch(runner)

    def before_test_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before each test epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self.before_epoch(runner)

    def after_train_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self.after_epoch(runner)

    def after_val_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self.after_epoch(runner)

    def after_test_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after each test epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self.after_epoch(runner)

    def before_train_iter(self, runner, data_batch: DATA_BATCH = None) -> None:
        """All subclasses should override this method, if they need any
        operations before each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional):
                Data from dataloader. Defaults to None.
        """
        self.before_iter(runner, data_batch=None)

    def before_val_iter(self, runner, data_batch: DATA_BATCH = None) -> None:
        """All subclasses should override this method, if they need any
        operations before each validation iteration.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional):
                Data from dataloader. Defaults to None.
        """
        self.before_iter(runner, data_batch=None)

    def before_test_iter(self, runner, data_batch: DATA_BATCH = None) -> None:
        """All subclasses should override this method, if they need any
        operations before each test iteration.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional):
                Data from dataloader. Defaults to None.
        """
        self.before_iter(runner, data_batch=None)

    def after_train_iter(
            self,
            runner,
            data_batch: DATA_BATCH = None,
            outputs: Optional[Sequence[BaseDataSample]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional):
                Data from dataloader. Defaults to None.
            outputs (Sequence[BaseDataSample], optional): Outputs from model.
                Defaults to None.
        """
        self.after_iter(runner, data_batch=None, outputs=None)

    def after_val_iter(
            self,
            runner,
            data_batch: DATA_BATCH = None,
            outputs: Optional[Sequence[BaseDataSample]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation iteration.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional):
                Data from dataloader. Defaults to None.
            outputs (Sequence[BaseDataSample], optional): Outputs from
                model. Defaults to None.
        """
        self.after_iter(runner, data_batch=None, outputs=None)

    def after_test_iter(
            self,
            runner,
            data_batch: DATA_BATCH = None,
            outputs: Optional[Sequence[BaseDataSample]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each test iteration.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional):
                Data from dataloader. Defaults to None.
            outputs (Sequence[BaseDataSample], optional): Outputs from model.
                Defaults to None.
        """
        self.after_iter(runner, data_batch=None, outputs=None)

    def every_n_epochs(self, runner, n: int) -> bool:
        """Test whether or not current epoch can be evenly divided by n.

        Args:
            runner (Runner): The runner of the training process.
            n (int): Whether or not current epoch can be evenly divided by n.

        Returns:
            bool: whether or not current epoch can be evenly divided by n.
        """
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, runner, n: int) -> bool:
        """Test whether or not current inner iteration can be evenly divided by
        n.

        Args:
            runner (Runner): The runner of the training process.
            n (int): Whether or not current inner iteration can be evenly
                divided by n.

        Returns:
            bool: whether or not current inner iteration can be evenly
            divided by n.
        """
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, runner, n: int) -> bool:
        """Test whether or not current iteration can be evenly divided by n.

        Args:
            runner (Runner): The runner of the training process.
            n (int): Whether or not current iteration can be
                evenly divided by n.

        Returns:
            bool: Return True if the current iteration can be evenly divided
            by n, otherwise False.
        """
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, runner) -> bool:
        """Check whether the current epoch reaches the `max_epochs` or not.

        Args:
            runner (Runner): The runner of the training process.

        Returns:
            bool: whether the end of current epoch or not.
        """
        return runner.inner_iter + 1 == len(runner.data_loader)

    def is_last_epoch(self, runner) -> bool:
        """Test whether or not current epoch is the last epoch.

        Args:
            runner (Runner): The runner of the training process.

        Returns:
            bool: bool: Return True if the current epoch reaches the
            `max_epochs`, otherwise False.
        """
        return runner.epoch + 1 == runner._max_epochs

    def is_last_iter(self, runner) -> bool:
        """Test whether or not current epoch is the last iteration.

        Args:
            runner (Runner): The runner of the training process.

        Returns:
            bool: whether or not current iteration is the last iteration.
        """
        return runner.iter + 1 == runner._max_iters
