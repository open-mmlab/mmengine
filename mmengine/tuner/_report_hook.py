# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
import math

from typing import Dict, Union, Sequence, Optional, List

DATA_BATCH = Optional[Union[dict, tuple, list]]

class ReportingHook(Hook):

    _max_history = 1024
    
    def __init__(self,
        monitor: str,
        rule: str,
        tuning_iter: int = 0,
        tunning_epoch: int = 0,
        report_op: str = 'latest'
    ):
        assert rule in ['greater', 'less'], f'rule {rule} is not supported'
        self.rule = rule    
        assert (tuning_iter == 0 and tunning_epoch > 0) or (tunning_epoch == 0 and tuning_iter > 0), 'tuning_iter and tuning_epoch should be set only one'
        assert report_op in ['latest', 'mean'], f'report_op {report_op} is not supported'
        self.report_op = report_op
        self.tuning_iter = tuning_iter
        self.tuning_epoch = tunning_epoch
        self.enabled_by_epoch = self._tuning_epoch != 0
        
        self.monitor = monitor
        self.history = []

    def _append_score(self, score):
        self.history.append(score)
        if len(self.history) > self._max_history:
            self.history.pop(0)

    def after_train_iter(
        self,
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
        if self.enabled_by_epoch:
            return
        if runner.iter + 1 == self.tuning_iter:
            runner.train_loop.stop_training = True

    def after_train_epoch(self, runner) -> None:
        if not self.enabled_by_epoch:
            return
        if runner.epoch + 1 == self.tuning_epoch:
            runner.train_loop.stop_training = True

    def  after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        if metrics is None:
            return
        score = metrics.get(self.monitor, None)
        if score is not None:
            self._append_score(score)

    def report_score(self):

        if self.report_op == 'latest':
            score = self.history[-1]
            if math.isnan(score) or math.isinf(score):
                if self.rule == 'greater':
                    score = float('-inf')
                else:
                    score = float('inf')

        elif self.report_op == 'mean':
            if any(math.isnan(s) or math.isinf(s) for s in self.history):
                if self.rule == 'greater':
                    score = float('-inf')
                else:
                    score = float('inf')
            else:
                score = sum(self.history) / len(self.history)
        return score

    def clear_history(self):
        self.history = []