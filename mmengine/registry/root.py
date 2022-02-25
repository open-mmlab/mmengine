# Copyright (c) OpenMMLab. All rights reserved.
"""MMEngine provides 11 root registries to support using modules across
projects.

More datails can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""

from .registry import Registry

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry('runner')
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry('runner constructor')
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry('hook')

# manage data-related modules
DATASETS = Registry('dataset')
DATA_SAMPLERS = Registry('data sampler')
TRANSFORMS = Registry('transform')

# mangage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model')
# mangage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry('model_wrapper')
# mangage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry('weight initializer')

# mangage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry('optimizer')
# manage constructors that customize the optimization hyperparameters.
OPTIMIZER_CONSTRUCTORS = Registry('optimizer constructor')
# mangage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry('parameter scheduler')

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry('task util')
