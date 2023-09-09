# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from mmengine.config import Config, ConfigDict
from mmengine.dist import (all_reduce, broadcast_object_list, init_dist,
                           is_distributed, is_main_process)
from mmengine.logging import MMLogger
from ._report_hook import ReportingHook
from .searchers import HYPER_SEARCHERS, Searcher

ConfigType = Union[Dict, Config, ConfigDict]


class Tuner:
    """A helper for hyperparameter tuning.

    By specifying a hyperparameter search space and a monitor metric,
    this `Tuner` will automatically identify the optimal hyperparameters
    for the runner.

    The `Tuner` employs black-box optimization techniques, ensuring
    a systematic search for the best hyperparameters within a defined space.
    The tuning process iteratively:

        1. Searches for hyperparameters considering
            the outcomes of previous trials.
        2. Constructs and trains the runner using the given hyperparameters.
        3. Assesses the performance of the trained runner's model
            and logs it in the searcher.

    Upon the conclusion of all trials, the hyperparameters that yielded
    the peak performance are returned.

    Args:
        runner_cfg (ConfigType):
            Configuration for the runner.
        hparam_spec (Dict[str, Dict]):
            The hyperparameter search space definition.
        monitor (str): The metric to be monitored during the tuning process.
        rule (str): The criterion used to determine the best hyperparameters.
                    Only 'greater' and 'less' are currently supported.
        num_trials (int): Total number of trials to execute.
        tuning_iter (Optional[int]): The maximum iterations for each trial.
            If specified, tuning stops after reaching this limit.
            Default is None, indicating no specific iteration limit.
        tuning_epoch (Optional[int]): The maximum epochs for each trial.
            If specified, tuning stops after reaching this number of epochs.
            Default is None, indicating no epoch limit.
        report_op (str):
            Operation mode for metric reporting. Default is 'latest'.
        searcher_cfg (ConfigType): Configuration for the searcher.
            Default is `dict(type='RandomSearcher')`.

    Note:
        The black-box optimization depends on external packages,
        such as `nevergrad`. Ensure the necessary packages are installed
        before using.

    Example:
        >>> from mmengine.tune import Tuner
        >>> runner_config = {"...": "..."}
        >>> hparam_spec = {
        >>>     'optim_wrapper.optimizer.lr': {
        >>>         'type': 'continuous',
        >>>         'lower': 1e-5,
        >>>         'upper': 1e-3
        >>>     }
        >>> }
        >>> tuner = Tuner(
        >>>     runner_cfg,
        >>>     hparam_spec=hparam_spec,
        >>>     monitor='train/loss',
        >>>     rule='less',
        >>>     num_trials=32,
        >>> )
        >>> result = tuner.tune()
        >>> print(result['hparam'])
        >>> print(result['score'])
    """
    rules_supported = ['greater', 'less']

    def __init__(self,
                 runner_cfg: ConfigType,
                 hparam_spec: Dict[str, Dict],
                 monitor: str,
                 rule: str,
                 num_trials: int,
                 tuning_iter: Optional[int] = None,
                 tuning_epoch: Optional[int] = None,
                 report_op: str = 'latest',
                 searcher_cfg: ConfigType = dict(type='RandomSearcher')):

        self._runner_cfg = runner_cfg.copy()
        self._hparam_spec = hparam_spec
        self._monitor = monitor

        if rule not in self.rules_supported:
            raise ValueError(f'Rule {rule} is not supported')
        self._rule = rule

        self._num_trials = num_trials
        self._tuning_iter = tuning_iter
        self._tuning_epoch = tuning_epoch
        self._reporting_op = report_op
        self._history: List[Tuple[Dict, float]] = []

        # Initialize distributed environment if necessary
        # This adjustment ensures consistent hyperparameter searching and
        # performance recording across all processes.
        launcher = self._runner_cfg.get('launcher', 'none')
        self._distributed = launcher != 'none'
        if self._distributed and not is_distributed():
            env_cfg = runner_cfg.get('env_cfg', {})
            dist_cfg = env_cfg.get('dist_cfg', {})
            init_dist(launcher, **dist_cfg)

        # Build logger to record tuning process
        self._logger = MMLogger.get_instance(
            'Tuner', log_level='INFO', distributed=self._distributed)
        self._logger.info(
            f'Tuner initialized with rule: {rule} and monitor: {monitor}')

        # Build searcher to search for optimal hyperparameters
        self._searcher = self._build_searcher(searcher_cfg)

    @property
    def hparam_spec(self) -> Dict[str, Dict]:
        """str: The hyperparameter search space definition."""
        return self._hparam_spec

    @property
    def monitor(self) -> str:
        """str: The metric to be monitored during the tuning process."""
        return self._monitor

    @property
    def rule(self) -> str:
        """str: The criterion used to determine the best hyperparameters."""
        return self._rule

    @property
    def num_trials(self) -> int:
        """int: Total number of trials to execute."""
        return self._num_trials

    @property
    def tuning_iter(self) -> Optional[int]:
        """Optional[int]: The maximum iterations for each trial.
            If specified, tuning
        """
        return self._tuning_iter

    @property
    def tuning_epoch(self) -> Optional[int]:
        """Optional[int]: The maximum epochs for each trial.
            If specified, tuning
        """
        return self._tuning_epoch

    @property
    def reporting_op(self) -> str:
        """str: Operation mode for metric reporting. Default is 'latest'."""
        return self._reporting_op

    @property
    def history(self) -> List[Tuple[Dict, float]]:
        """List[Tuple[Dict, float]]: The history of hyperparameters and
        scores."""
        return self._history

    @property
    def searcher(self) -> Searcher:
        """Searcher: The searcher used for hyperparameter tuning."""
        return self._searcher

    @staticmethod
    def inject_config(cfg: ConfigType, key: str, value: Any):
        """Inject a value into a config.

        The name can be multi-level, like 'optimizer.lr'.

        Args:
            cfg (ConfigType): The config to be injected.
            key (str): The key of the value to be injected.
            value (Any): The value to be injected.
        """
        keys = key.split('.')
        for k in keys[:-1]:
            if isinstance(cfg, list):
                idx = int(k)
                if idx >= len(cfg) or idx < 0:
                    raise KeyError(f'Index {idx} is out of range in {cfg}')
                cfg = cfg[idx]
            else:
                if k not in cfg:
                    raise KeyError(f"Key '{k}' not found in {cfg}")
                cfg = cfg[k]

        if isinstance(cfg, list):
            idx = int(keys[-1])
            if idx >= len(cfg) or idx < 0:
                raise KeyError(f'Index {idx} is out of range in {cfg}')
            cfg[idx] = value
        else:
            if keys[-1] not in cfg:
                raise KeyError(f"Key '{keys[-1]}' not found in {cfg}")
            else:
                cfg[keys[-1]] = value
        return

    def _build_searcher(self, searcher_cfg: ConfigType) -> Searcher:
        """Build searcher from searcher_cfg.

        An Example of ``searcher_cfg``::

            searcher_cfg = dict(
                type='NevergradSearcher',
                solver_type='CMA'
            )

        Args:
            searcher_cfg (ConfigType): The searcher config.
        """
        searcher_cfg = searcher_cfg.copy()
        self._logger.info(f'Building searcher of type: {searcher_cfg["type"]}')
        searcher_cfg.update(
            dict(
                rule=self.rule,
                hparam_spec=self.hparam_spec,
                num_trials=self._num_trials))
        return HYPER_SEARCHERS.build(searcher_cfg)

    def _run_trial(self) -> Tuple[Dict, float, Optional[Exception]]:
        """Retrieve hyperparameters from searcher and run a trial."""

        # Retrieve hyperparameters for the trial:
        # 1. Only the main process executes the searcher to avoid any conflicts
        #   and ensure integrity.
        # 2. Once retrieved, the hyperparameters are broadcasted to all other
        #   processes ensuring every process has the same set of
        #   hyperparameters for this trial.
        from mmengine.runner import Runner

        if is_main_process():
            hparams_to_broadcast = [self._searcher.suggest()]
        else:
            hparams_to_broadcast = [None]  # type: ignore
        broadcast_object_list(hparams_to_broadcast, src=0)
        hparam = hparams_to_broadcast[0]

        # Inject hyperparameters into runner config.
        for k, v in hparam.items():
            self.inject_config(self._runner_cfg, k, v)
        runner = Runner.from_cfg(self._runner_cfg)
        report_hook = ReportingHook(self._monitor, self._tuning_iter,
                                    self._tuning_epoch, self._reporting_op)
        runner.register_hook(report_hook, priority='VERY_LOW')
        default_score = float('inf') if self._rule == 'less' else -float('inf')

        # Run a trial.
        # If an exception occurs during the trial, the score is set
        # to default_score.
        score: float
        error: Optional[Exception] = None
        try:
            runner.train()
            score = report_hook.report_score()
            if score is None or math.isnan(score) or math.isinf(score):
                score = default_score
        except Exception as e:
            score = default_score
            error = e

        # Synchronize and average scores across all processes
        score_tensor = torch.tensor(score)
        all_reduce(score_tensor, op='mean')
        score = score_tensor.item()

        if is_main_process():
            self._searcher.record(hparam, score)
        return hparam, score, error

    def tune(self) -> Dict[str, Union[Dict[str, Any], float]]:
        """Launch tuning.

        Returns:
        Dict[str, Union[Dict[str, Any], float]]:
            A dictionary containing the best hyperparameters under the key
            'hparam' and the corresponding score under the key 'score'.
        """
        self._logger.info(f'Starting tuning for {self._num_trials} trials...')
        for trail_idx in range(self._num_trials):
            hparam, score, error = self._run_trial()
            log_msg = f'Trial [{trail_idx + 1}/{self._num_trials}]'
            if error is not None:
                log_msg += f' failed. Error: {error}'
            else:
                log_msg += f' finished. Score obtained: {score}'
            log_msg += f' Hyperparameters used: {hparam}'
            self._logger.info(log_msg)
            self._history.append((hparam, score))

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

    def clear(self):
        """Clear the history of hyperparameters and scores."""
        self._history.clear()
