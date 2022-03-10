# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Sequence, Tuple, Union

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
            runner (Runner): The runner of the training/validation/testing
                process.
        """
        pass

    def after_run(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after the training process.

        Args:
            runner (Runner): The runner of the training/validation/testing
                process.
        """
        pass

    def before_train(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before train.

        Args:
            runner (Runner): The runner of the training process.
        """
        pass

    def after_train(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after train.

        Args:
            runner (Runner): The runner of the training process.
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
        self._before_epoch(runner, mode='train')

    def before_val_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before each validation epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self._before_epoch(runner, mode='val')

    def before_test_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before each test epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self._before_epoch(runner, mode='test')

    def after_train_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self._after_epoch(runner, mode='train')

    def after_val_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self._after_epoch(runner, mode='val')

    def after_test_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after each test epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self._after_epoch(runner, mode='test')

    def before_train_iter(self, runner, data_batch: DATA_BATCH = None) -> None:
        """All subclasses should override this method, if they need any
        operations before each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional):
                Data from dataloader. Defaults to None.
        """
        self._before_iter(runner, data_batch=None, mode='train')

    def before_val_iter(self, runner, data_batch: DATA_BATCH = None) -> None:
        """All subclasses should override this method, if they need any
        operations before each validation iteration.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional):
                Data from dataloader. Defaults to None.
        """
        self._before_iter(runner, data_batch=None, mode='val')

    def before_test_iter(self, runner, data_batch: DATA_BATCH = None) -> None:
        """All subclasses should override this method, if they need any
        operations before each test iteration.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional):
                Data from dataloader. Defaults to None.
        """
        self._before_iter(runner, data_batch=None, mode='test')

    def after_train_iter(self,
                         runner,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional):
                Data from dataloader. Defaults to None.
            outputs (dict, optional): Outputs from model.
                Defaults to None.
        """
        self._after_iter(runner, data_batch=None, outputs=None, mode='train')

    def after_val_iter(self,
                       runner,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence[BaseDataSample]] = None) \
            -> None:
        """All subclasses should override this method, if they need any
        operations after each validation iteration.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional):
                Data from dataloader. Defaults to None.
            outputs (dict or sequence, optional): Outputs from
                model. Defaults to None.
        """
        self._after_iter(runner, data_batch=None, outputs=None, mode='val')

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
            outputs (dict, optional): Outputs from model.
                Defaults to None.
        """
        self._after_iter(runner, data_batch=None, outputs=None, mode='test')

    def _before_epoch(self, runner, mode: str = 'train') -> None:
        """All subclasses should override this method, if they need any
        operations before each epoch.

        Args:
            runner (Runner): The runner of the training process.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        pass

    def _after_epoch(self, runner, mode: str = 'train') -> None:
        """All subclasses should override this method, if they need any
        operations after each epoch.

        Args:
            runner (Runner, ): The runner of the training process.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        pass

    def _before_iter(self, runner,
                     data_batch: DATA_BATCH = None,
                     mode: str = 'train') -> None:
        """All subclasses should override this method, if they need any
        operations before each iter.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional):
                Data from dataloader. Defaults to None.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        pass

    def _after_iter(self,
                    runner,
                    data_batch: DATA_BATCH = None,
                    outputs: Optional[Sequence[BaseDataSample]] = None,
                    mode: str = 'train') -> None:
        """All subclasses should override this method, if they need any
        operations after each epoch.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional):
                Data from dataloader. Defaults to None.
            outputs (Sequence[BaseDataSample], optional): Outputs from model.
                Defaults to None.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        pass

    def every_n_epochs(self, runner, n: int, mode: str = 'train') -> bool:
        """Test whether or not current epoch can be evenly divided by n.

        Args:
            runner (Runner): The runner of the training process.
            n (int): Whether or not current epoch can be evenly divided by n.
            mode (str): Current mode of runner. Defaults to 'train'.
        Returns:
            bool: whether or not current epoch can be evenly divided by n.
        """
        # TODO check train_loop type
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, runner, n: int, mode: str = 'train') -> bool:
        """Test whether or not current inner iteration can be evenly divided by
        n.

        Args:
            runner (Runner): The runner of the training process.
            n (int): Whether or not current inner iteration can be evenly
                divided by n.
            mode (str): Current mode of runner. Defaults to 'train'.

        Returns:
            bool: whether or not current inner iteration can be evenly
            divided by n.
        """
        # TODO check train_loop type
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, runner, n: int, mode: str = 'train') -> bool:
        """Test whether or not current iteration can be evenly divided by n.

        Args:
            runner (Runner): The runner of the training process.
            n (int): Whether or not current iteration can be
                evenly divided by n.
            mode (str): Current mode of runner. Defaults to 'train'.

        Returns:
            bool: Return True if the current iteration can be evenly divided
            by n, otherwise False.
        """
        # TODO check train_loop type
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, runner, mode: str = 'train') -> bool:
        """Check whether the current epoch reaches the `max_epochs` or not.

        Args:
            runner (Runner): The runner of the training process.
            mode (str): Current mode of runner. Defaults to 'train'.

        Returns:
            bool: whether the end of current epoch or not.
        """
        if mode == 'train':
            data_loader = runner.train_dataloader
        else:
            data_loader = runner.val_dataloader
        return runner.inner_iter + 1 == len(data_loader)

    def is_last_epoch(self, runner, mode: str = 'train') -> bool:
        """Test whether or not current epoch is the last epoch.

        Args:
            runner (Runner): The runner of the training process.
            mode (str): Current mode of runner. Defaults to 'train'.

        Returns:
            bool: bool: Return True if the current epoch reaches the
            `max_epochs`, otherwise False.
        """
        # TODO check train_loop type
        return runner.epoch + 1 == runner.train_loop._max_epochs

    def is_last_iter(self, runner, mode: str = 'train') -> bool:
        """Test whether or not current epoch is the last iteration.

        Args:
            runner (Runner): The runner of the training process.
            mode (str): Current mode of runner. Defaults to 'train'.

        Returns:
            bool: whether or not current iteration is the last iteration.
        """
        # TODO check train_loop type
        if mode == 'train':
            _max_iters = runner.train_loop._max_iters
        elif mode == 'val':
            _max_iters = runner.val_loop._max_iters
        elif mode == 'test':
            _max_iters = runner.test_loop._max_iters
        else:
            raise ValueError('mode should be train, val or test')
        return runner.iter + 1 == _max_iters
