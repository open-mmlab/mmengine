# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Optional, Union

from mmengine.registry import HOOKS
from .hook import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """Early stop the training when the metric reached a plateau.

    Args:
        metric (str): The metric key to decide early stopping.
        rule (str, optional): Comparison rule. Options are 'greater',
            'less'. Defaults to None.
        delta(float, optional): Minimum difference to continue the training.
            Defaults to 0.01.
        pool_size (int, optional): The number of experiments to consider.
            Defaults to 5.
        patience (int, optional): Maximum number of tolerance.
            Defaults to 0.
    """
    priority = 'LOWEST'

    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    _default_greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
        'mAcc', 'aAcc'
    ]
    _default_less_keys = ['loss']

    def __init__(
        self,
        metric: str,
        rule: str = None,
        delta: float = 0.1,
        pool_size: int = 5,
        patience: int = 0,
    ):

        self.metric = metric
        if metric in self._default_greater_keys:
            rule = 'greater'
        elif metric in self._default_less_keys:
            rule = 'less'
        assert rule in ['greater', 'less'], \
            '`rule` should be either \'greater\' or \'less\'.'
        self.rule = rule
        self.delta = delta
        self.pool_size = pool_size
        self.patience = patience
        self.count = 0

        self.pool_values: List[float] = []

    def before_run(self, runner) -> None:
        """Check `stop_training` variable in `runner.train_loop`.

        Args:
            runner (Runner): The runner of the training process.
        """

        assert hasattr(runner.train_loop, 'stop_training'), \
            '`train_loop` should contain `stop_training` variable.'

    def after_val_epoch(self, runner, metrics):
        """Decide whether to stop the training process.

        Args:
            runner (Runner): The runner of the training process.
            metrics (dict): Evaluation results of all metrics
        """

        if self.metric not in metrics:
            warnings.warn(
                f'Skip early stopping process since the evaluation '
                f'results ({metrics.keys()}) do not include `metric` '
                f'({self.metric}).')
            return

        latest_value = metrics[self.metric]
        compare = self.rule_map[self.rule]

        self.pool_values.append(latest_value)

        if self.rule == 'greater':
            # maintain largest values
            self.pool_values = sorted(self.pool_values)[-self.pool_size:]
        else:
            # maintain smalleast values
            self.pool_values = sorted(self.pool_values)[:self.pool_size]

        if len(self.pool_values) == self.pool_size and compare(
                sum(self.pool_values) / self.pool_size + self.delta,
                latest_value):

            self.count += 1

            if self.count >= self.patience:
                runner.train_loop.stop_training = True
                runner.logger.info(
                    'The metric reached a plateau. '
                    'This training process will be stopped early.')
        else:
            self.count = 0
