# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

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
        """Update current iter information before every iteration."""
        runner.message_hub.update_info('iter', runner.iter)
        runner.message_hub.update_scalar(
            'train/lr', runner.optimizer.param_groups[0]['lr'])

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Update ``log_vars`` in model outputs every iteration."""
        if outputs is not None:
            for key, value in outputs['log_vars'].items():
                runner.message_hub.update_scalar(f'train/{key}', value)

    def after_val_epoch(self, runner, metrics: Dict[str, float]) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float]): Evaluation results of all metrics
                on validation dataset. The keys are the names of the metrics,
                and the values are corresponding results.
        """
        for key, value in metrics.items():
            runner.message_hub.update_scalar(f'val/{key}', value)

    def after_test_epoch(self, runner, metrics: Dict[str, float]) -> None:
        """All subclasses should override this method, if they need any
        operations after each test epoch.

        Args:
            runner (Runner): The runner of the testing process.
            metrics (Dict[str, float]): Evaluation results of all metrics
                on test dataset. The keys are the names of the metrics, and
                the values are corresponding results.
        """
        for key, value in metrics.items():
            runner.message_hub.update_scalar(f'test/{key}', value)
