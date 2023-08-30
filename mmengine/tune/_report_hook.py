# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List, Optional, Sequence, Union

from mmengine.hooks import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]


class ReportingHook(Hook):
    """Auxiliary hook to report the score to tuner.

    If tuning limit is specified, this hook will mark the loop to stop.

    Args:
        monitor (str): The monitored metric key to report.
        tuning_iter (int, optional): The iteration limit to stop tuning.
            Defaults to None.
        tuning_epoch (int, optional): The epoch limit to stop tuning.
            Defaults to None.
        report_op (str, optional): The operation to report the score.
            Options are 'latest', 'mean'. Defaults to 'latest'.
        max_scoreboard_len (int, optional):
            The maximum length of the scoreboard.
    """

    report_op_supported = ['latest', 'mean']

    def __init__(self,
                 monitor: str,
                 tuning_iter: Optional[int] = None,
                 tuning_epoch: Optional[int] = None,
                 report_op: str = 'latest',
                 max_scoreboard_len: int = 1024):
        assert report_op in self.report_op_supported, \
            f'report_op {report_op} is not supported'
        if tuning_iter is not None and tuning_epoch is not None:
            raise ValueError(
                'tuning_iter and tuning_epoch cannot be set at the same time')
        self.report_op = report_op
        self.tuning_iter = tuning_iter
        self.tuning_epoch = tuning_epoch

        self.monitor = monitor
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

        tag, _ = runner.log_processor.get_log_after_iter(
            runner, batch_idx, 'train')
        score = tag.get(self.monitor, None)
        if score is not None:
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
        if metrics is None:
            return
        score = metrics.get(self.monitor, None)
        if score is not None:
            self._append_score(score)

    def report_score(self) -> Optional[float]:
        """Aggregate the scores in the scoreboard.

        Returns:
            Optional[float]: The aggregated score.
        """
        if not self.scoreboard:
            score = None
        elif self.report_op == 'latest':
            score = self.scoreboard[-1]
        else:
            score = sum(self.scoreboard) / len(self.scoreboard)
        return score

    def clear(self):
        """Clear the scoreboard."""
        self.scoreboard.clear()
