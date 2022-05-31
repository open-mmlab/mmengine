# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

from mmengine.optim import OptimWrapper, OptimWrapperDict
from mmengine.registry import HOOKS
from .hook import Hook

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class RuntimeInfoHook(Hook):
    """A hook that updates runtime information into message hub.

    E.g. ``epoch``, ``iter``, ``max_epochs``, and ``max_iters`` for the
    training state. Components that cannot access the runner can get runtime
    information through the message hub.
    """

    priority = 'VERY_HIGH'

    def before_run(self, runner) -> None:
        """Initialize runtime information."""
        runner.message_hub.update_info('epoch', runner.epoch)
        runner.message_hub.update_info('iter', runner.iter)
        runner.message_hub.update_info('max_epochs', runner.max_epochs)
        runner.message_hub.update_info('max_iters', runner.max_iters)

    def before_train(self, runner) -> None:
        """Update resumed training state."""
        runner.message_hub.update_info('epoch', runner.epoch)
        runner.message_hub.update_info('iter', runner.iter)

    def before_train_epoch(self, runner) -> None:
        """Update current epoch information before every epoch."""
        runner.message_hub.update_info('epoch', runner.epoch)

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:
        """Update current iter and learning rate information before every
        iteration."""
        runner.message_hub.update_info('iter', runner.iter)
        if not isinstance(runner.optim_wrapper, OptimWrapperDict):
            # Since `OptimWrapperDict` inherits from `OptimWrapper`,
            # `isinstance(self.optim_wrapper, OptimWrapper)` cannot tell
            # whether `self.optim_wrapper` is an `OptimizerWrapper` or
            # `OptimWrapperDict` instance. Therefore, here we simply check
            # self.optim_wrapper is not an `OptimWrapperDict` instance and
            # then assert it is an OptimWrapper instance.
            assert isinstance(runner.optim_wrapper, OptimWrapper)
            runner.message_hub.update_scalar(
                'train/lr', runner.optim_wrapper.param_groups[0]['lr'])
        else:
            for name, optim_wrapper in runner.optim_wrapper.items():
                runner.message_hub.update_scalar(
                    f'train/{name}.lr', optim_wrapper.param_groups[0]['lr'])

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Update ``log_vars`` in model outputs every iteration."""
        if outputs is not None:
            for key, value in outputs['log_vars'].items():
                runner.message_hub.update_scalar(f'train/{key}', value)

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
        if metrics is not None:
            for key, value in metrics.items():
                runner.message_hub.update_scalar(f'val/{key}', value)

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
        if metrics is not None:
            for key, value in metrics.items():
                runner.message_hub.update_scalar(f'test/{key}', value)
