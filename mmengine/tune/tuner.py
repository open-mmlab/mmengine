# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import TensorPipeRpcBackendOptions, is_available

from mmengine.config import Config, ConfigDict
from mmengine.dist import (broadcast_object_list, get_rank, get_world_size,
                           init_dist, is_distributed, is_main_process)
from mmengine.logging import MMLogger
from mmengine.runner import Runner
from ._report_hook import ReportingHook
from .searchers import HYPER_SEARCHERS, _Searcher


class Tuner:
    rules_supported = ['greater', 'less']
    rpc_worker_name = 'RPC_WORKER{}'

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
                 rpc_port: int = 29501,
                 **searcher_kwargs):
        assert is_available(), 'torch.distributed.rpc is not available.'

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
        self._init_rpc(rpc_port)
        self._rpc_port = rpc_port
        self._logger = MMLogger.get_instance('Tuner', log_level='INFO')
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
    def get_rpc_worker_name(rank) -> str:
        return Tuner.rpc_worker_name.format(rank)

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

    @staticmethod
    def run_trial(runner_cfg, monitor, rule, tuning_iter, tunning_epoch,
                  report_op):
        os.environ['LOCAL_RANK'] = '0'
        runner = Runner.from_cfg(runner_cfg)
        report_hook = ReportingHook(monitor, rule, tuning_iter, tunning_epoch,
                                    report_op)
        runner.register_hook(report_hook, priority='VERY_LOW')
        runner.train()
        return report_hook.report_score()

    def _init_rpc(self, rpc_port: int):
        rank = get_rank()
        world_size = get_world_size()
        rpc_init_method: str
        if self._distributed:
            rpc_init_method = 'env://'
        else:
            rpc_init_method = f'tcp://localhost:{rpc_port}'
        rpc_backend_options = TensorPipeRpcBackendOptions(
            init_method=rpc_init_method,
            devices=[int(os.environ.get('LOCAL_RANK', rank))],
        )

        rpc.init_rpc(
            Tuner.get_rpc_worker_name(rank),
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options)

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

    def _get_score_from_futures(self, futs) -> float:
        try:
            return torch.futures.wait_all(futs)[0]
        except Exception:
            return float('-inf') if self._rule == 'greater' else float('inf')

    def _submit(self):

        if is_main_process():
            hparam = self._searcher.suggest()
            for k, v in hparam.items():
                self.inject_config(self._runner_cfg, k, v)
            temp_dir = tempfile.TemporaryDirectory()
            self._runner_cfg['work_dir'] = temp_dir.name

            futs = []
            for rank in range(get_world_size()):
                fut = rpc.rpc_async(
                    Tuner.get_rpc_worker_name(rank),
                    Tuner.run_trial,
                    args=(self._runner_cfg, self._monitor, self._rule,
                          self._tuning_iter, self._tuning_epoch,
                          self._reporting_op))
                futs.append(fut)
            score = self._get_score_from_futures(futs)
            self._logger.info(f'Trial completed with score: {score}')
            self._searcher.record(hparam, score)
            temp_dir.cleanup()
        else:
            hparam = None
            score = None
        broadcast_object_list([hparam, score], src=0)
        self._history.append((hparam, score))

    def tune(self) -> Dict[str, Union[dict, float]]:
        self._logger.info(f'Starting tuning for {self._num_trials} trials...')
        for _ in range(self._num_trials):
            self._submit()

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
