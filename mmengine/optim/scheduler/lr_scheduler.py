# Copyright (c) OpenMMLab. All rights reserved.
from .param_scheduler import (INF, ConstantParamScheduler,
                              CosineAnnealingParamScheduler,
                              ExponentialParamScheduler, LinearParamScheduler,
                              MultiStepParamScheduler, StepParamScheduler)


class ConstantLR(ConstantParamScheduler):

    def __init__(self,
                 optimizer,
                 factor=1.0 / 3,
                 begin=0,
                 end=INF,
                 last_step=-1,
                 verbose=False):
        super().__init__(
            optimizer,
            param_name='lr',
            factor=factor,
            begin=begin,
            end=end,
            last_step=last_step,
            verbose=verbose)


class CosineAnnealingLR(CosineAnnealingParamScheduler):

    def __init__(self,
                 optimizer,
                 T_max,
                 eta_min=0,
                 begin=0,
                 end=INF,
                 last_step=-1,
                 verbose=False):
        super().__init__(
            optimizer,
            param_name='lr',
            T_max=T_max,
            eta_min=eta_min,
            begin=begin,
            end=end,
            last_step=last_step,
            verbose=verbose)


class ExponentialLR(ExponentialParamScheduler):

    def __init__(self,
                 optimizer,
                 gamma,
                 begin=0,
                 end=INF,
                 last_step=-1,
                 verbose=False):
        super().__init__(
            optimizer,
            param_name='lr',
            gamma=gamma,
            begin=begin,
            end=end,
            last_step=last_step,
            verbose=verbose)


class LinearLR(LinearParamScheduler):

    def __init__(self,
                 optimizer,
                 start_factor=1.0 / 3,
                 end_factor=1.0,
                 begin=0,
                 end=INF,
                 last_step=-1,
                 verbose=False):
        super().__init__(
            optimizer,
            param_name='lr',
            start_factor=start_factor,
            end_factor=end_factor,
            begin=begin,
            end=end,
            last_step=last_step,
            verbose=verbose)


class MultiStepLR(MultiStepParamScheduler):

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 last_step=-1,
                 begin=0,
                 end=INF,
                 verbose=False):
        super().__init__(
            optimizer,
            param_name='lr',
            milestones=milestones,
            gamma=gamma,
            last_step=last_step,
            begin=begin,
            end=end,
            verbose=verbose)


class StepLR(StepParamScheduler):

    def __init__(self,
                 optimizer,
                 step_size,
                 gamma=0.1,
                 begin=0,
                 end=INF,
                 last_step=-1,
                 by_epoch=True,
                 verbose=False):
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
