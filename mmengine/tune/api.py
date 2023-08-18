# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

from mmengine.config import Config, ConfigDict
from .tuner import Tuner


def find_optimial_lr(runner_cfg: Union[Dict, Config, ConfigDict],
                     monitor: str = 'loss',
                     rule: str = 'less',
                     num_trials: int = 32,
                     lower_lr: Optional[float] = 1e-6,
                     upper_lr: Optional[float] = 1e-2,
                     lr_choices: Optional[List[float]] = None,
                     tuning_iter: int = 0,
                     tunning_epoch: int = 0,
                     report_op: str = 'latest',
                     searcher_type: str = 'nevergrad',
                     **searcher_kwargs) -> Dict[str, Union[dict, float]]:
    is_discrete = lr_choices is not None
    if is_discrete:
        assert lower_lr is None and upper_lr is None, \
            'lower_lr and upper_lr should be None if lr_choices is not None'
    else:
        assert lower_lr is not None and upper_lr is not None, \
            'lower_lr and upper_lr should set if lr_choices is None'
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

    tuner = Tuner(
        runner_cfg,
        hparam_spec=hparam_spec,
        monitor=monitor,
        rule=rule,
        num_trials=num_trials,
        tuning_iter=tuning_iter,
        tunning_epoch=tunning_epoch,
        report_op=report_op,
        searcher_type=searcher_type,
        **searcher_kwargs)
    return tuner.tune()
