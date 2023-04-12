# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler
from mmengine.testing.runner_test_case import ToyDataset, ToyMetric

if '_base_':
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
