# Copyright (c) OpenMMLab. All rights reserved.
"""MMEngine provides 11 root registries to support cross-project calls.

More datails can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""

from .registry import Registry

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry('runner')
# mangage some constructors to custom runners
RUNNER_CONSTRUCTORS = Registry('runner constructor')
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry('hook')

# manage data-related modules
DATASETS = Registry('dataset')
SAMPLERS = Registry('sampler')
PIPELINES = Registry('pipeline')

# mangage all kinds of components to build networks like `DeformConv2d`
MODELS = Registry('model')
# mangage all kinds of weight initialization module like `Uniform`
WEIGHT_INITIALIZERS = Registry('weight initializer')

# mangage all kinds of optimizer like `SGD` and `Adam`
OPTIMIZERS = Registry('optimizer')
# mangage some constructors to custom optimizers
OPTIMIZER_CONSTRUCTORS = Registry('optimizer constructor')

# manage some components related to task like `AnchorGenerator`
TASK_UTILS = Registry('task util')
