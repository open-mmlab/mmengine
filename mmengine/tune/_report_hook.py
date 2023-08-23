# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List, Optional, Sequence, Union

from mmengine.hooks import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]


class ReportingHook(Hook):

    def __init__(self,
                 monitor: str,
                 tuning_iter: Optional[int] = None,
                 tuning_epoch: Optional[int] = None,
                 report_op: str = 'latest',
                 max_scoreboard_len: int = 1024):
        self.has_limit = tuning_iter is not None or tuning_epoch is not None
        assert report_op in ['latest',
                             'mean'], f'report_op {report_op} is not supported'
        self.report_op = report_op
        self.tuning_iter = tuning_iter
        self.tuning_epoch = tuning_epoch

        self.monitor = monitor
        self.max_scoreboard_len = max_scoreboard_len
        self.scoreboard: List[float] = []

    def _append_score(self, score):
        self.scoreboard.append(score)
        if len(self.scoreboard) > self.max_scoreboard_len:
            self.scoreboard.pop(0)

    def _mark_stop(self, runner):
        if self.tuning_iter is not None:
            if runner.iter > self.tuning_iter:
                runner.train_loop.stop_training = True
        if self.tuning_epoch is not None:
            if runner.epoch > self.tuning_epoch:
                runner.train_loop.stop_training = True

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[Union[dict, Sequence]] = None,
                         mode: str = 'train') -> None:

        tag, _ = runner.log_processor.get_log_after_iter(
            runner, batch_idx, 'train')
        score = tag.get(self.monitor, None)
        if score is not None:
            self._append_score(score)

        if self.has_limit:
            self._mark_stop(runner)

    def after_train_epoch(self, runner) -> None:
        if self.has_limit:
            self._mark_stop(runner)

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        if metrics is None:
            return
        score = metrics.get(self.monitor, None)
        if score is not None:
            self._append_score(score)

    def report_score(self) -> Optional[float]:
        if not self.scoreboard:
            score = None
        elif self.report_op == 'latest':
            score = self.scoreboard[-1]
        else:
            score = sum(self.scoreboard) / len(self.scoreboard)
        return score

    def clear_scoreboard(self):
        self.scoreboard = []
