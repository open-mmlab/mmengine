# Copyright (c) OpenMMLab. All rights reserved.
from .tunner import Tuner

from typing import Dict, List, Optional, Union, Tuple

from mmengine.config import Config, ConfigDict


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
                     searcher: str = 'nevergrad',
                     **searcher_kwargs) -> Tuple[dict, float]:
    is_discrete = lr_choices is not None
    assert (lower_lr is None and upper_lr is None and lr_choices
            is not None) or (lower_lr is not None and upper_lr is not None
                             and lr_choices is None
                             ), 'lower_lr and upper_lr should be set only one'
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
        **searcher_kwargs)
    return tunner.tune()