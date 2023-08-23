# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List, Sequence, Tuple, Union

from mmengine.config import Config, ConfigDict
from mmengine.dist import (broadcast_object_list, init_dist, is_distributed,
                           is_main_process)
from mmengine.logging import MMLogger
from mmengine.runner import Runner
from ._report_hook import ReportingHook
from .searchers import HYPER_SEARCHERS, _Searcher


class Tuner:
    rules_supported = ['greater', 'less']

    def __init__(self,
                 runner_cfg: Union[Dict, Config, ConfigDict],
                 hparam_spec: Dict[str, Dict],
                 monitor: str,
                 rule: str,
                 num_trials: int,
                 tuning_iter: int = 0,
                 tunning_epoch: int = 0,
                 report_op: str = 'latest',
                 searcher_type: str = 'NevergradSearcher',
                 **searcher_kwargs):

        self._runner_cfg = runner_cfg.copy()
        self._hparam_spec = hparam_spec
        self._monitor = monitor

        if rule not in self.rules_supported:
            raise ValueError(f'Rule {rule} is not supported')
        self._rule = rule

        self._num_trials = num_trials
        self._tuning_iter = tuning_iter
        self._tuning_epoch = tunning_epoch
        self._reporting_op = report_op
        self._history: List[Tuple[Dict, float]] = []

        launcher = self._runner_cfg.get('launcher', 'none')
        self._distributed = launcher != 'none'
        if self._distributed and not is_distributed():
            env_cfg = runner_cfg.get('env_cfg', {})
            dist_cfg = env_cfg.get('dist_cfg', {})
            init_dist(launcher, **dist_cfg)
        self._logger = MMLogger.get_instance(
            'Tuner', log_level='INFO', distributed=self._distributed)
        self._logger.info(
            f'Tuner initialized with rule: {rule} and monitor: {monitor}')
        self._searcher = self._build_searcher(searcher_type, **searcher_kwargs)

    @property
    def hparam_spec(self) -> Dict[str, Dict]:
        return self._hparam_spec

    @property
    def monitor(self) -> str:
        return self._monitor

    @property
    def rule(self) -> str:
        return self._rule

    @property
    def num_trials(self) -> int:
        return self._num_trials

    @property
    def tuning_iter(self) -> int:
        return self._tuning_iter

    @property
    def tuning_epoch(self) -> int:
        return self._tuning_epoch

    @property
    def reporting_op(self) -> str:
        return self._reporting_op

    @property
    def history(self) -> List[Tuple[Dict, float]]:
        return self._history

    @staticmethod
    def inject_config(cfg, key, value):
        key = key.split('.')
        suffix = ''
        for item in key[:-1]:
            if isinstance(cfg, Sequence) and not isinstance(cfg, str):
                cfg = cfg[int(item)]
            else:
                assert item in cfg, f'key {item} is not in {cfg}'
                cfg = cfg[item]
            suffix += f'{item}.'
        assert key[-1] in cfg, f'attribute {key[-1]} is not in cfg{suffix}'
        cfg[key[-1]] = value
        return

    def _build_searcher(self,
                        searcher_type: str = 'nevergrad',
                        **kwargs) -> _Searcher:
        self._logger.info(f'Building searcher of type: {searcher_type}')
        build_config = dict(
            type=searcher_type,
            rule=self.rule,
            hparam_spec=self.hparam_spec,
            num_trials=self._num_trials)
        build_config.update(kwargs)
        return HYPER_SEARCHERS.build(build_config)

    def _run_trial(self) -> Tuple[Dict, float]:
        if is_main_process():
            hparams_to_broadcast = [self._searcher.suggest()]
        else:
            hparams_to_broadcast = [None]  # type: ignore
        broadcast_object_list(hparams_to_broadcast, src=0)
        hparam = hparams_to_broadcast[0]
        for k, v in hparam.items():
            self.inject_config(self._runner_cfg, k, v)
        runner = Runner.from_cfg(self._runner_cfg)
        report_hook = ReportingHook(self._monitor, self._tuning_iter,
                                    self._tuning_epoch, self._reporting_op)
        runner.register_hook(report_hook, priority='VERY_LOW')
        default_score = float('inf') if self._rule == 'less' else -float('inf')
        try:
            runner.train()
            score = report_hook.report_score()
            if score is None or math.isnan(score) or math.isinf(score):
                score = default_score
            scores_to_broadcast = [score]
        except Exception:
            scores_to_broadcast = [default_score]
        broadcast_object_list(scores_to_broadcast, src=0)
        score = scores_to_broadcast[0]
        if is_main_process():
            self._searcher.record(hparam, score)
        return hparam, score

    def tune(self) -> Dict[str, Union[dict, float]]:
        self._logger.info(f'Starting tuning for {self._num_trials} trials...')
        for trail_idx in range(self._num_trials):
            hparam, score = self._run_trial()
            self._history.append((hparam, score))
            self._logger.info(
                f'Trial [{trail_idx + 1}/{self._num_trials}] finished.' +
                f' Score obtained: {score}' +
                f' Hyperparameters used: {hparam}')

        best_hparam: dict
        best_score: float
        if self._rule == 'greater':
            best_hparam, best_score = max(self._history, key=lambda x: x[1])
        else:
            best_hparam, best_score = min(self._history, key=lambda x: x[1])
        self._logger.info(f'Best hyperparameters obtained: {best_hparam}')
        self._logger.info(f'Best score obtained: {best_score}')
        self._logger.info('Tuning completed.')
        return dict(hparam=best_hparam, score=best_score)

    def clean_up(self):
        self._history = []
