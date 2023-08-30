# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from mmengine.dataset import DefaultSampler
from mmengine.hooks import EMAHook
from mmengine.model import MomentumAnnealingEMA
from mmengine.runner import FlexibleRunner
from mmengine.testing.runner_test_case import ToyDataset, ToyMetric

with read_base():
    from ._base_.base_model import *
    from ._base_.default_runtime import *
    from ._base_.scheduler import *

param_scheduler.milestones = [2, 4]


train_dataloader = dict(
    dataset=dict(type=ToyDataset),
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_size=3,
    num_workers=0)

val_dataloader = dict(
    dataset=dict(type=ToyDataset),
    sampler=dict(type=DefaultSampler, shuffle=False),
    batch_size=3,
    num_workers=0)

val_evaluator = [dict(type=ToyMetric)]

test_dataloader = dict(
    dataset=dict(type=ToyDataset),
    sampler=dict(type=DefaultSampler, shuffle=False),
    batch_size=3,
    num_workers=0)

test_evaluator = [dict(type=ToyMetric)]

custom_hooks = [
    dict(
        type=EMAHook,
        ema_type=MomentumAnnealingEMA,
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

runner_type = FlexibleRunner
