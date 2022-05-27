# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Union

from mmengine.data import BaseDataElement

DATA_BATCH = Optional[Sequence[dict]]


class Hook:
    """Base hook class.

    All hooks should inherit from this class.
    """

    priority = 'NORMAL'

    def before_run(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before the training validation or testing process.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
        """
        pass

    def after_run(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before the training validation or testing process.

        Args:
            runner (Runner): The runner of the training, validation or testing
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

    def before_val(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before validation.

        Args:
            runner (Runner): The runner of the validation process.
        """
        pass

    def after_val(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after validation.

        Args:
            runner (Runner): The runner of the validation process.
        """
        pass

    def before_test(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before testing.

        Args:
            runner (Runner): The runner of the testing process.
        """
        pass

    def after_test(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after testing.

        Args:
            runner (Runner): The runner of the testing process.
        """
        pass

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        """All subclasses should override this method, if they need any
        operations before saving the checkpoint.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            checkpoint (dict): Model's checkpoint.
        """
        pass

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        """All subclasses should override this method, if they need any
        operations after loading the checkpoint.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            checkpoint (dict): Model's checkpoint.
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
            runner (Runner): The runner of the validation process.
        """
        self._before_epoch(runner, mode='val')

    def before_test_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before each test epoch.

        Args:
            runner (Runner): The runner of the testing process.
        """
        self._before_epoch(runner, mode='test')

    def after_train_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self._after_epoch(runner, mode='train')

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        self._after_epoch(runner, mode='val')

    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each test epoch.

        Args:
            runner (Runner): The runner of the testing process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on test dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        self._after_epoch(runner, mode='test')

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:
        """All subclasses should override this method, if they need any
        operations before each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
        """
        self._before_iter(
            runner, batch_idx=batch_idx, data_batch=data_batch, mode='train')

    def before_val_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None) -> None:
        """All subclasses should override this method, if they need any
        operations before each validation iteration.

        Args:
            runner (Runner): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
        """
        self._before_iter(
            runner, batch_idx=batch_idx, data_batch=data_batch, mode='val')

    def before_test_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None) -> None:
        """All subclasses should override this method, if they need any
        operations before each test iteration.

        Args:
            runner (Runner): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
        """
        self._before_iter(
            runner, batch_idx=batch_idx, data_batch=data_batch, mode='test')

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model.
                Defaults to None.
        """
        self._after_iter(
            runner,
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs,
            mode='train')

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence[BaseDataElement]] = None) \
            -> None:
        """All subclasses should override this method, if they need any
        operations after each validation iteration.

        Args:
            runner (Runner): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict or sequence, optional): Outputs from
                model. Defaults to None.
        """
        self._after_iter(
            runner,
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs,
            mode='val')

    def after_test_iter(
            self,
            runner,
            batch_idx: int,
            data_batch: DATA_BATCH = None,
            outputs: Optional[Sequence[BaseDataElement]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each test iteration.

        Args:
            runner (Runner): The runner of the training  process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model.
                Defaults to None.
        """
        self._after_iter(
            runner,
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs,
            mode='test')

    def _before_epoch(self, runner, mode: str = 'train') -> None:
        """All subclasses should override this method, if they need any
        operations before each epoch.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        pass

    def _after_epoch(self, runner, mode: str = 'train') -> None:
        """All subclasses should override this method, if they need any
        operations after each epoch.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        pass

    def _before_iter(self,
                     runner,
                     batch_idx: int,
                     data_batch: DATA_BATCH = None,
                     mode: str = 'train') -> None:
        """All subclasses should override this method, if they need any
        operations before each iter.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            batch_idx (int): The index of the current batch in the loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        pass

    def _after_iter(self,
                    runner,
                    batch_idx: int,
                    data_batch: DATA_BATCH = None,
                    outputs: Optional[Union[Sequence[BaseDataElement],
                                            dict]] = None,
                    mode: str = 'train') -> None:
        """All subclasses should override this method, if they need any
        operations after each epoch.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            batch_idx (int): The index of the current batch in the loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (Sequence[BaseDataElement], optional): Outputs from model.
                Defaults to None.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        pass

    def every_n_epochs(self, runner, n: int) -> bool:
        """Test whether current epoch can be evenly divided by n.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            n (int): Whether current epoch can be evenly divided by n.

        Returns:
            bool: Whether current epoch can be evenly divided by n.
        """
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, batch_idx: int, n: int) -> bool:
        """Test whether current inner iteration can be evenly divided by n.

        Args:
            batch_idx (int): Current batch index of the training, validation
                or testing loop.
            n (int): Whether current inner iteration can be evenly
                divided by n.

        Returns:
            bool: Whether current inner iteration can be evenly
            divided by n.
        """
        return (batch_idx + 1) % n == 0 if n > 0 else False

    def every_n_train_iters(self, runner, n: int) -> bool:
        """Test whether current training iteration can be evenly divided by n.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            n (int): Whether current iteration can be evenly divided by n.

        Returns:
            bool: Return True if the current iteration can be evenly divided
            by n, otherwise False.
        """
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, dataloader, batch_idx: int) -> bool:
        """Check whether the current iteration reaches the last iteration of
        the dataloader.

        Args:
            dataloader (Dataloader): The dataloader of the training,
                validation or testing process.
            batch_idx (int): The index of the current batch in the loop.
        Returns:
            bool: Whether reaches the end of current epoch or not.
        """
        return batch_idx + 1 == len(dataloader)

    def is_last_train_epoch(self, runner) -> bool:
        """Test whether current epoch is the last train epoch.

        Args:
            runner (Runner): The runner of the training process.

        Returns:
            bool: Whether reaches the end of training epoch.
        """
        return runner.epoch + 1 == runner.max_epochs

    def is_last_train_iter(self, runner) -> bool:
        """Test whether current iteration is the last train iteration.

        Args:
            runner (Runner): The runner of the training process.

        Returns:
            bool: Whether current iteration is the last train iteration.
        """
        return runner.iter + 1 == runner.max_iters
