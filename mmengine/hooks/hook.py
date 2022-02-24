# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmengine.data import BaseDataSample
from mmengine.registry import Registry

HOOKS = Registry('hook')


class Hook:
    """The base hooks class.

    All hooks should inherit from this class.
    """

    def before_run(self, runner: object) -> None:
        """All subclasses should override this function, if they need any
        operations before the training process."""
        pass

    def after_run(self, runner: object) -> None:
        """All subclasses should override this function, if they need any
        operations after the training process."""
        pass

    def before_epoch(self, runner: object) -> None:
        """All subclasses should override this function, if they need any
        operations before each epoch."""
        pass

    def after_epoch(self, runner: object) -> None:
        """All subclasses should override this function, if they need any
        operations after each epoch."""
        pass

    def before_iter(self,
                    runner: object,
                    data_batch: Sequence[BaseDataSample] = None) -> None:
        """All subclasses should override this function, if they need any
        operations before each iter."""
        pass

    def after_iter(self,
                   runner: object,
                   data_batch: Sequence[BaseDataSample] = None,
                   outputs: Sequence[BaseDataSample] = None) -> None:
        """All subclasses should override this function, if they need any
        operations after each epoch."""
        pass

    def before_save_checkpoint(self, runner: object, checkpoint: dict) -> None:
        """All subclasses should override this function, if they need any
        operations before saving the checkpoint."""
        pass

    def after_load_checkpoint(self, runner: object, checkpoint: dict) -> None:
        """All subclasses should override this function, if they need any
        operations after saving the checkpoint."""
        pass

    def before_train_epoch(self, runner: object) -> None:
        """All subclasses should override this function, if they need any
        operations before each training epoch."""
        self.before_epoch(runner)

    def before_val_epoch(self, runner: object) -> None:
        """All subclasses should override this function, if they need any
        operations before each validation epoch."""
        self.before_epoch(runner)

    def before_test_epoch(self, runner: object) -> None:
        """All subclasses should override this function, if they need any
        operations before each test epoch."""
        self.before_epoch(runner)

    def after_train_epoch(self, runner: object) -> None:
        """All subclasses should override this function, if they need any
        operations after each training epoch."""
        self.after_epoch(runner)

    def after_val_epoch(self, runner: object) -> None:
        """All subclasses should override this function, if they need any
        operations after each validation epoch."""
        self.after_epoch(runner)

    def after_test_epoch(self, runner: object) -> None:
        """All subclasses should override this function, if they need any
        operations after each test epoch."""
        self.after_epoch(runner)

    def before_train_iter(self,
                          runner: object,
                          data_batch: Sequence[BaseDataSample] = None) -> None:
        """All subclasses should override this function, if they need any
        operations before each training iteration."""
        self.before_iter(runner, data_batch=None)

    def before_val_iter(self,
                        runner: object,
                        data_batch: Sequence[BaseDataSample] = None) -> None:
        """All subclasses should override this function, if they need any
        operations before each validation iteration."""
        self.before_iter(runner, data_batch=None)

    def before_test_iter(self,
                         runner: object,
                         data_batch: Sequence[BaseDataSample] = None) -> None:
        """All subclasses should override this function, if they need any
        operations before each test iteration."""
        self.before_iter(runner, data_batch=None)

    def after_train_iter(self,
                         runner: object,
                         data_batch: Sequence[BaseDataSample] = None,
                         outputs: Sequence[BaseDataSample] = None) -> None:
        """All subclasses should override this function, if they need any
        operations after each training iteration."""
        self.after_iter(runner, data_batch=None, outputs=None)

    def after_val_iter(self,
                       runner: object,
                       data_batch: Sequence[BaseDataSample] = None,
                       outputs: Sequence[BaseDataSample] = None) -> None:
        """All subclasses should override this function, if they need any
        operations after each validation iteration."""
        self.after_iter(runner, data_batch=None, outputs=None)

    def after_test_iter(self,
                        runner: object,
                        data_batch: Sequence[BaseDataSample] = None,
                        outputs: Sequence[BaseDataSample] = None) -> None:
        """All subclasses should override this function, if they need any
        operations after each test iteration."""
        self.after_iter(runner, data_batch=None, outputs=None)

    def every_n_epochs(self, runner: object, n: int) -> bool:
        """Test whether or not current epoch can be evenly divided by n."""
        return (runner.epoch + 1) % n == 0 if n > 0 else False  # type: ignore

    def every_n_inner_iters(self, runner: object, n: int) -> bool:
        """Test whether or not current inner iteration can be evenly divided by
        n."""
        return (runner.inner_iter +  # type: ignore
                1) % n == 0 if n > 0 else False

    def every_n_iters(self, runner: object, n: int) -> bool:
        """Test whether or not current iteration can be evenly divided by n."""
        return (runner.iter + 1) % n == 0 if n > 0 else False  # type: ignore

    def end_of_epoch(self, runner: object) -> bool:
        """Test whether the end of current epoch or not."""
        return runner.inner_iter + 1 == len(runner.data_loader)  # type: ignore

    def is_last_epoch(self, runner: object) -> bool:
        """Test whether or not current epoch is the last epoch."""
        return runner.epoch + 1 == runner._max_epochs  # type: ignore

    def is_last_iter(self, runner: object) -> bool:
        """Test whether or not current epoch is the last iteration."""
        return runner.iter + 1 == runner._max_iters  # type: ignore
