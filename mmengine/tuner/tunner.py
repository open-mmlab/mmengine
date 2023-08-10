# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmengine.runner import Runner
from mmengine.config import Config, ConfigDict
from mmengine.dist import init_dist, broadcast_object_list, is_main_process

from tying import Dict, Union, Sequence, Optional, List

from mmengine.dist import is_distributed

from ._report_hook import ReportingHook

class Tuner:

    def __init__(self, runner_cfg: Union[Dict, Config, ConfigDict], hparam_spec: Dict[str, Dict], monitor: str, rule: str, num_trials: int, tuning_iter: int = 0, tunning_epoch: int = 0, report_op: str = 'latest', searcher: str = 'nevergrad', **searcher_kwargs):
        self._runner_cfg = runner_cfg.copy()
        self._hparam_spec = hparam_spec
        self._monitor = monitor
        assert rule in  ['greater', 'less'], f'rule {rule} is not supported'
        self._rule = rule
        self._num_trials = num_trials
        self._searcher = self._build_searcher(searcher, **searcher_kwargs)
        self._reporting_hook = ReportingHook(monitor, rule, tuning_iter, tunning_epoch, report_op)
        self._history = []

        launcher = self._runner_cfg.get('launcher', 'none')
        env_cfg = self._runner_cfg.get('env_cfg', {})
        self._distributed: bool
        if launcher == 'none':
            self._distributed = False
        else:
            self._distributed = True
        if self.distributed and not is_distributed():
            dist_cfg: dict = env_cfg.get('dist_cfg', {})
            init_dist(self.launcher, **dist_cfg)

    def _build_searcher(self, searcher: str = 'nevergrad', **kwargs):
        if searcher == 'nevergrad':
            from .searcher import NevergradSearcher
            searcher = NevergradSearcher(self._mode, self._hparam_spec, self._num_trials, **kwargs)
        elif searcher == 'skopt':
            from .searcher import SkoptSearcher
            searcher = SkoptSearcher(self._mode, self._hparam_spec, self._num_trials, **kwargs)
        elif searcher == 'hyperopt':
            from .searcher import HyperoptSearcher
            searcher = HyperoptSearcher(self._mode, self._hparam_spec, self._num_trials, **kwargs)
        else:
            raise NotImplementedError(f'searcher {searcher} is not implemented')
        return searcher
        
    @staticmethod
    def inject_config(cfg, key, value):
        key = key.split('.')
        suffix = ''
        for item in key[:-1]:
            if isinstance(cfg, Sequence) and not isinstance(cfg, str):
                item = cfg[int(item)]
            else:
                assert item in cfg, f'key {key} is not in cfg'
                item = cfg[item]
            suffix += f'{item}.'
        assert key[-1] in cfg, f'attribute {key[-1]} is not in cfg{suffix}' 
        cfg[key[-1]] = value
        return

    def tune(self):
        for _ in range(self._num_trials):
            if is_main_process():
                hparam = [self._searcher.suggest()]
            else:
                hparam = [None]
            broadcast_object_list(hparam) 
            # Sync hparam if distributed
            for k, v in hparam[0].items():
                self.inject_config(self._runner_cfg, k, v)
            runner = Runner.from_cfg(self._runner_cfg)
            runner.register_hook(self._reporting_hook, priority='VERY_LOW')
            score: float
            try:
                runner.train()
                score = [self._reporting_hook.get_score()]
            except Exception as e:
                if self._rule == 'greater':
                    score = [float('-inf')]
                else:
                    score = [float('inf')]
            finally:
                broadcast_object_list(score)
                self._searcher.record(hparam[0], score[0])
                runner = self.tear_down_trial(runner)
                self._history.append((hparam[0], score[0]))

        beset_hparam: dict
        if self._rule == 'greater':
            beset_hparam = max(self._history, key=lambda x: x[1])[0]
        else:
            beset_hparam = min(self._history, key=lambda x: x[1])[0]
        return beset_hparam
            
    def tear_down_trial(self, runner):
        del runner
        torch.cuda.empty_cache()
        self._reporting_hook.clear_history()


def find_optimial_lr(
    runner_cfg: Union[Dict, Config, ConfigDict],
    monitor: str = 'loss',
    rule: str = 'less',
    num_trials: int = 32,
    lower_lr: Optional[float] = 1e-6, upper_lr: Optional[float] = 1e-2, lr_choices : Optional[List[float]] = None, tuning_iter: int = 1e4, tunning_epoch: int = 0, report_op: str = 'latest', searcher: str = 'nevergrad', **searcher_kwargs 
):
    is_discrete = lr_choices is not None
    assert (lower_lr is None and upper_lr is None and lr_choices is not None) or (lower_lr is not None and upper_lr is not None and lr_choices is None), 'lower_lr and upper_lr should be set only one'
    hparam_spec: dict
    if is_discrete:
        hparam_spec = {
            'optimizer.lr': {
                'type': 'discrete',
                'values': lr_choices
            }
        }
    else:
        hparam_spec = {
            'optimizer.lr': {
                'type': 'continuous',
                'lower': lower_lr,
                'upper': upper_lr
            }
        }

    tunner = Tuner(
        runner_cfg,
        hparam_spec=hparam_spec,
        monitor=monitor,
        rule=rule,
        num_trials=num_trials,
        tuning_iter=tuning_iter,
        tunning_epoch=tunning_epoch,
        report_op=report_op,
        searcher=searcher,
        **searcher_kwargs
    )
    return tunner.tune()