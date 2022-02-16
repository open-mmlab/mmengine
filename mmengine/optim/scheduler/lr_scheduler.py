# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch

from mmengine.registry import SCHEDULERS
from .param_scheduler import (INF, ConstantParamScheduler,
                              CosineAnnealingParamScheduler,
                              ExponentialParamScheduler, LinearParamScheduler,
                              MultiStepParamScheduler, StepParamScheduler)


@SCHEDULERS.register_module()
class ConstantLR(ConstantParamScheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 factor: float = 1.0 / 3,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        super().__init__(
            optimizer,
            param_name='lr',
            factor=factor,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)


@SCHEDULERS.register_module()
class CosineAnnealingLR(CosineAnnealingParamScheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: int = 0,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        super().__init__(
            optimizer,
            param_name='lr',
            T_max=T_max,
            eta_min=eta_min,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)


@SCHEDULERS.register_module()
class ExponentialLR(ExponentialParamScheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 gamma: float,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        super().__init__(
            optimizer,
            param_name='lr',
            gamma=gamma,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)


@SCHEDULERS.register_module()
class LinearLR(LinearParamScheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 start_factor: float = 1.0 / 3,
                 end_factor: float = 1.0,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        super().__init__(
            optimizer,
            param_name='lr',
            start_factor=start_factor,
            end_factor=end_factor,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)


@SCHEDULERS.register_module()
class MultiStepLR(MultiStepParamScheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 milestones: List[int],
                 gamma: float = 0.1,
                 last_step: int = -1,
                 begin: int = 0,
                 end: int = INF,
                 by_epoch: bool = True,
                 verbose: bool = False):
        super().__init__(
            optimizer,
            param_name='lr',
            milestones=milestones,
            gamma=gamma,
            last_step=last_step,
            begin=begin,
            end=end,
            by_epoch=by_epoch,
            verbose=verbose)


@SCHEDULERS.register_module()
class StepLR(StepParamScheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 step_size: int,
                 gamma: float = 0.1,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        super().__init__(
            optimizer,
            param_name='lr',
            step_size=step_size,
            gamma=gamma,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)
