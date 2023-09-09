# Copyright (c) OpenMMLab. All rights reserved.

from typing import Callable, Dict, List, Optional, Sequence, Union

from mmengine.hooks import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]


class ReportingHook(Hook):
    """Auxiliary hook to report the score to tuner. The ReportingHook maintains
    a "scoreboard" which keeps track of the monitored metrics' scores during
    the training process. The scores are aggregated based on the method
    specified by the 'report_op' parameter. If tuning limit is specified, this
    hook will mark the loop to stop.

    Args:
        monitor (str): The monitored metric key prefixed with either 'train/'
            or 'val/' to indicate the specific phase where the metric should
            be monitored. For instance, 'train/loss' will monitor the 'loss'
            metric during the training phase, and 'val/accuracy' will monitor
            the 'accuracy' metric during the validation phase.
            The actual metric key (i.e., the part following the prefix)
            should correspond to a key in the logs produced during
            training or validation.
        tuning_iter (int, optional): The iteration limit to stop tuning.
            Defaults to None.
        tuning_epoch (int, optional): The epoch limit to stop tuning.
            Defaults to None.
        report_op (str, optional): The method for aggregating scores
            in the scoreboard. Accepts the following options:
            - 'latest': Returns the most recent score in the scoreboard.
            - 'mean': Returns the mean of all scores in the scoreboard.
            - 'max': Returns the highest score in the scoreboard.
            - 'min': Returns the lowest score in the scoreboard.
            Defaults to 'latest'.
        max_scoreboard_len (int, optional):
            Specifies the maximum number of scores that can be retained
            on the scoreboard, helping to manage memory and computational
            overhead. Defaults to 1024.
    """

    report_op_supported: Dict[str, Callable[[List[float]], float]] = {
        'latest': lambda x: x[-1],
        'mean': lambda x: sum(x) / len(x),
        'max': max,
        'min': min
    }

    def __init__(self,
                 monitor: str,
                 tuning_iter: Optional[int] = None,
                 tuning_epoch: Optional[int] = None,
                 report_op: str = 'latest',
                 max_scoreboard_len: int = 1024):
        if not monitor.startswith('train/') and not monitor.startswith('val/'):
            raise ValueError("The 'monitor' parameter should start "
                             "with 'train/' or 'val/' to specify the phase.")
        if report_op not in self.report_op_supported:
            raise ValueError(f'report_op {report_op} is not supported')
        if tuning_iter is not None and tuning_epoch is not None:
            raise ValueError(
                'tuning_iter and tuning_epoch cannot be set at the same time')
        self.monitor_prefix, self.monitor_metric = monitor.split('/', 1)
        self.report_op = report_op
        self.tuning_iter = tuning_iter
        self.tuning_epoch = tuning_epoch

        self.max_scoreboard_len = max_scoreboard_len
        self.scoreboard: List[float] = []

    def _append_score(self, score: float):
        """Append the score to the scoreboard."""
        self.scoreboard.append(score)
        if len(self.scoreboard) > self.max_scoreboard_len:
            self.scoreboard.pop(0)

    def _should_stop(self, runner):
        """Check if the training should be stopped.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.tuning_iter is not None:
            if runner.iter + 1 >= self.tuning_iter:
                return True
        elif self.tuning_epoch is not None:
            if runner.epoch + 1 >= self.tuning_epoch:
                return True
        else:
            return False

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[Union[dict, Sequence]] = None,
                         mode: str = 'train') -> None:
        """Record the score after each iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        if self.monitor_prefix != 'train':
            return
        tag, _ = runner.log_processor.get_log_after_iter(
            runner, batch_idx, 'train')
        score = tag.get(self.monitor_metric)
        if not isinstance(score, (int, float)):
            raise ValueError(f"The monitored value '{self.monitor_metric}' "
                             'should be a number.')
        self._append_score(score)

        if self._should_stop(runner):
            runner.train_loop.stop_training = True

    def after_train_epoch(self, runner) -> None:
        """Record the score after each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self._should_stop(runner):
            runner.train_loop.stop_training = True

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """Record the score after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        if self.monitor_prefix != 'val' or metrics is None:
            return
        score = metrics.get(self.monitor_metric)
        if not isinstance(score, (int, float)):
            raise ValueError(f"The monitored value '{self.monitor_metric}' "
                             'should be a number.')
        self._append_score(score)

    def report_score(self) -> Optional[float]:
        """Aggregate the scores in the scoreboard.

        Returns:
            Optional[float]: The aggregated score.
        """
        if not self.scoreboard:
            score = None
        else:
            operation = self.report_op_supported[self.report_op]
            score = operation(self.scoreboard)
        return score

    @classmethod
    def register_report_op(cls, name: str, func: Callable[[List[float]],
                                                          float]):
        """Register a new report operation.

        Args:
            name (str): The name of the report operation.
            func (Callable[[List[float]], float]): The function to aggregate
                the scores.
        """
        cls.report_op_supported[name] = func

    def clear(self):
        """Clear the scoreboard."""
        self.scoreboard.clear()
