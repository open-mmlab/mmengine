# Copyright (c) OpenMMLab. All rights reserved.
from .registry import Registry

# TODO
RUNNERS = Registry('runner')
RUNNER_CONSTRUCTORS = Registry('runner constructor')
HOOKS = Registry('hook')

DATASETS = Registry('dataset')
SAMPLERS = Registry('sampler')
PIPELINES = Registry('pipeline')

MODELS = Registry('model')
WEIGHT_INITIALIZERS = Registry('weight initializer')

OPTIMIZERS = Registry('optimizer')
OPTIMIZER_CONSTRUCTORS = Registry('optimizer constructor')

TASK_UTILS = Registry('task util')
