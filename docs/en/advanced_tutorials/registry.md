# Registry

OpenMMLab supports a rich collection of algorithms and datasets, therefore, many modules with similar functionality are implemented. For example, the implementations of `ResNet` and `SE-ResNet` are based on the classes `ResNet` and `SEResNet`, respectively, which have similar functions and interfaces and belong to the model components of the algorithm library. To manage these functionally similar modules, MMEngine implements the [registry](mmengine.registry.Registry). Most of the algorithm libraries in OpenMMLab use `registry` to manage their modules, including [MMDetection](https://github.com/open-mmlab/mmdetection), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [MMPretrain](https://github.com/open-mmlab/mmpretrain) and [MMagic](https://github.com/open-mmlab/MMagic), etc.

## What is a registry

The [registry](mmengine.registry.Registry) in MMEngine can be considered as a union of a mapping table and a build function of modules. The mapping table maintains a mapping from strings to **classes or functions**, allowing the user to find the corresponding class or function with its name/notation. For example, the mapping from the string `"ResNet"` to the `ResNet` class. The module build function defines how to find the corresponding class or function based on a string and how to instantiate the class or call the function. For example, finding `nn.BatchNorm2d` and instantiating the `BatchNorm2d` module by the string `"bn"`, or finding the `build_batchnorm2d` function by the string `"build_batchnorm2d"` and then returning the result. The registries in MMEngine use the [build_from_cfg](mmengine.registry.build_from_cfg) function by default to find and instantiate the class or function corresponding to the string.

The classes or functions managed by a registry usually have similar interfaces and functionality, so the registry can be treated as an abstraction of those classes or functions. For example, the registry `MODELS` can be treated as an abstraction of all models, which manages classes such as `ResNet`, `SEResNet` and `RegNetX` and constructors such as `build_ResNet`, `build_SEResNet` and `build_RegNetX`.

## Getting started

There are three steps required to use the registry to manage modules in the codebase.

1. Create a registry.
2. Create a build method for instantiating the class (optional because in most cases you can just use the default method).
3. Add the module to the registry

Suppose we want to implement a series of activation modules and want to be able to switch to different modules by just modifying the configuration without modifying the code.

Let's create a registry first.

```python
from mmengine import Registry
# `scope` represents the domain of the registry. If not set, the default value is the package name.
# e.g. in mmdetection, the scope is mmdet
# `locations` indicates the location where the modules in this registry are defined.
# The Registry will automatically import the modules when building them according to these predefined locations.
ACTIVATION = Registry('activation', scope='mmengine', locations=['mmengine.models.activations'])
```

The module `mmengine.models.activations` specified by `locations` corresponds to the `mmengine/models/activations.py` file. When building modules with registry, the ACTIVATION registry will automatically import implemented modules from this file. Therefore, we can implement different activation layers in the `mmengine/models/activations.py` file, such as `Sigmoid`, `ReLU`, and `Softmax`.

```python
import torch.nn as nn

# use the register_module
@ACTIVATION.register_module()
class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print('call Sigmoid.forward')
        return x

@ACTIVATION.register_module()
class ReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        print('call ReLU.forward')
        return x

@ACTIVATION.register_module()
class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print('call Softmax.forward')
        return x
```

The key of using the registry module is to register the implemented modules into the `ACTIVATION` registry. With the `@ACTIVATION.register_module()` decorator added before the implemented module, the mapping between strings and classes or functions can be built and maintained by `ACTIVATION`. We can achieve the same functionality with `ACTIVATION.register_module(module=ReLU)` as well.

By registering, we can create a mapping between strings and classes or functions via `ACTIVATION`.

```python
print(ACTIVATION.module_dict)
# {
#     'Sigmoid': __main__.Sigmoid,
#     'ReLU': __main__.ReLU,
#     'Softmax': __main__.Softmax
# }
```

```{note}
The key to trigger the registry mechanism is to make the module imported.
There are three ways to register a module into the registry

1. Implement the module in the ``locations``. The registry will automatically import modules in the predefined locations. This is to ease the usage of algorithm libraries so that users can directly use ``REGISTRY.build(cfg)``.

2. Import the file manually. This is common when developers implement a new module in/out side the algorithm library.

3. Use ``custom_imports`` field in config. Please refer to [Importing custom Python modules](config.md#import-the-custom-module) for more details.
```

Once the implemented module is successfully registered, we can use the activation module in the configuration file.

```python
import torch

input = torch.randn(2)

act_cfg = dict(type='Sigmoid')
activation = ACTIVATION.build(act_cfg)
output = activation(input)
# call Sigmoid.forward
print(output)
```

We can switch to `ReLU` by just changing this configuration.

```python
act_cfg = dict(type='ReLU', inplace=True)
activation = ACTIVATION.build(act_cfg)
output = activation(input)
# call ReLU.forward
print(output)
```

If we want to check the type of input parameters (or any other operations) before creating an instance, we can implement a build method and pass it to the registry to implement a custom build process.

Create a `build_activation` function.

```python
def build_activation(cfg, registry, *args, **kwargs):
    cfg_ = cfg.copy()
    act_type = cfg_.pop('type')
    print(f'build activation: {act_type}')
    act_cls = registry.get(act_type)
    act = act_cls(*args, **kwargs, **cfg_)
    return act
```

Pass the `buid_activation` to `build_func`.

```python
ACTIVATION = Registry('activation', build_func=build_activation, scope='mmengine', locations=['mmengine.models.activations'])

@ACTIVATION.register_module()
class Tanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print('call Tanh.forward')
        return x

act_cfg = dict(type='Tanh')
activation = ACTIVATION.build(act_cfg)
output = activation(input)
# build activation: Tanh
# call Tanh.forward
print(output)
```

```{note}
In the above example, we demonstrate how to customize the method of building an instance of a class using the `build_func`.
This is similar to the default `build_from_cfg` method. In most cases, using the default method will be fine.
```

MMEngine's registry can register classes as well as functions.

```python
FUNCTION = Registry('function', scope='mmengine')

@FUNCTION.register_module()
def print_args(**kwargs):
    print(kwargs)

func_cfg = dict(type='print_args', a=1, b=2)
func_res = FUNCTION.build(func_cfg)
```

## Advanced usage

The registry in MMEngine supports hierarchical registration, which enables cross-project calls, meaning that modules from one project can be used in another project. Though there are other ways to implement this, the registry provides a much easier solution.

To easily make cross-library calls, MMEngine provides twenty two root registries, including:

- RUNNERS: the registry for Runner.
- RUNNER_CONSTRUCTORS: the constructors for Runner.
- LOOPS: manages training, validation and testing processes, such as `EpochBasedTrainLoop`.
- HOOKS: the hooks, such as `CheckpointHook`, and `ParamSchedulerHook`.
- DATASETS: the datasets.
- DATA_SAMPLERS: `Sampler` of `DataLoader`, used to sample the data.
- TRANSFORMS: various data preprocessing methods, such as `Resize`, and `Reshape`.
- MODELS: various modules of the model.
- MODEL_WRAPPERS: model wrappers for parallelizing distributed data, such as `MMDistributedDataParallel`.
- WEIGHT_INITIALIZERS: the tools for weight initialization.
- OPTIMIZERS: registers all `Optimizers` and custom `Optimizers` in PyTorch.
- OPTIM_WRAPPER: the wrapper for Optimizer-related operations such as `OptimWrapper`, and `AmpOptimWrapper`.
- OPTIM_WRAPPER_CONSTRUCTORS: the constructors for optimizer wrappers.
- PARAM_SCHEDULERS: various parameter schedulers, such as `MultiStepLR`.
- METRICS: the evaluation metrics for computing model accuracy, such as `Accuracy`.
- EVALUATOR: one or more evaluation metrics used to calculate the model accuracy.
- TASK_UTILS: the task-intensive components, such as `AnchorGenerator`, and `BboxCoder`.
- VISUALIZERS: the management drawing module that draws prediction boxes on images, such as `DetVisualizer`.
- VISBACKENDS: the backend for storing training logs, such as `LocalVisBackend`, and `TensorboardVisBackend`.
- LOG_PROCESSORS: controls the log statistics window and statistics methods, by default we use `LogProcessor`. You may customize `LogProcessor` if you have special needs.
- FUNCTIONS: registers various functions, such as `collate_fn` in `DataLoader`.
- INFERENCERS: registers inferencers of different tasks, such as `DetInferencer`, which is used to perform inference on the detection task.

### Use the module of the parent node

Let's define a `RReLU` module in `MMEngine` and register it to the `MODELS` root registry.

```python
import torch.nn as nn
from mmengine import Registry, MODELS

@MODELS.register_module()
class RReLU(nn.Module):
    def __init__(self, lower=0.125, upper=0.333, inplace=False):
        super().__init__()

    def forward(self, x):
        print('call RReLU.forward')
        return x
```

Now suppose there is a project called `MMAlpha`, which also defines a `MODELS` and sets its parent node to the `MODELS` of `MMEngine`, which creates a hierarchical structure.

```python
from mmengine import Registry, MODELS as MMENGINE_MODELS

MODELS = Registry('model', parent=MMENGINE_MODELS, scope='mmalpha', locations=['mmalpha.models'])
```

The following figure shows the hierarchy of `MMEngine` and `MMAlpha`.

<div align="center">
  <img src="https://user-images.githubusercontent.com/58739961/185307159-26dc5771-df77-4d03-9203-9c4c3197befa.png"/>
</div>

The [count_registered_modules](mmengine.registry.count_registered_modules) function can be used to print the modules that have been registered to MMEngine and their hierarchy.

```python
from mmengine.registry import count_registered_modules

count_registered_modules()
```

We define a customized `LogSoftmax` module in `MMAlpha` and register it to the `MODELS` in `MMAlpha`.

```python
@MODELS.register_module()
class LogSoftmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()

    def forward(self, x):
        print('call LogSoftmax.forward')
        return x
```

Here we use the `LogSoftmax` in the configuration of `MMAlpha`.

```python
model = MODELS.build(cfg=dict(type='LogSoftmax'))
```

We can also use the modules of the parent node `MMEngine` here in the `MMAlpha`.

```python
model = MODELS.build(cfg=dict(type='RReLU', lower=0.2))
# scope is optional
model = MODELS.build(cfg=dict(type='mmengine.RReLU'))
```

If no prefix is added, the `build` method will first find out if the module exists in the current node and return it if there is one. Otherwise, it will continue to look up the parent nodes or even the ancestor node until it finds the module. If the same module exists in both the current node and the parent nodes, we need to specify the `scope` prefix to indicate that we want to use the module of the parent nodes.

```python
import torch

input = torch.randn(2)
output = model(input)
# call RReLU.forward
print(output)
```

### How does the parent node know about child registry?

When working in our `MMAlpha` it might be necessary to use the `Runner` class defined in MMENGINE. This class is in charge of building most of the objects. If these objects are added to the child registry (`MMAlpha`), how is  `MMEngine` able to find them? It cannot, `MMEngine` needs to switch to the Registry from `MMEngine` to `MMAlpha` according to the scope which is defined in default_runtime.py for searching the target class.

We can also init the scope accordingly, see example below:

```python
from mmalpha.registry import MODELS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import init_default_scope
import torch.nn as nn

@MODELS.register_module()
class LogSoftmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()

    def forward(self, x):
        print('call LogSoftmax.forward')
        return x

# Works because we are using mmalpha registry
MODELS.build(dict(type="LogSoftmax"))

# Fails because mmengine registry does not know about stuff registered in mmalpha
MMENGINE_MODELS.build(dict(type="LogSoftmax"))

# Works because we are using mmalpha registry
init_default_scope('mmalpha')
MMENGINE_MODELS.build(dict(type="LogSoftmax"))
```

### Use the module of a sibling node

In addition to using the module of the parent nodes, users can also call the module of a sibling node.

Suppose there is another project called `MMBeta`, which, like `MMAlpha`, defines `MODELS` and set its parent node to `MMEngine`.

```python
from mmengine import Registry, MODELS as MMENGINE_MODELS

MODELS = Registry('model', parent=MMENGINE_MODELS, scope='mmbeta')
```

The following figure shows the registry structure of `MMAlpha` and `MMBeta`.

<div align="center">
  <img src="https://user-images.githubusercontent.com/58739961/185307738-9ddbce2d-f8b5-40c4-bf8f-603830ccc0dc.png"/>
</div>

Now we call the modules of `MMAlpha` in `MMBeta`.

```python
model = MODELS.build(cfg=dict(type='mmalpha.LogSoftmax'))
output = model(input)
# call LogSoftmax.forward
print(output)
```

Calling a module of a sibling node requires the `scope` prefix to be specified in `type`, so the above configuration requires the prefix `mmalpha`.

However, if you need to call several modules of a sibling node, each with a prefix, this requires a lot of modification. Therefore, `MMEngine` introduces the [DefaultScope](mmengine.registry.DefaultScope), with which `Registry` can easily support temporary switching of the current node to the specified node.

If you need to switch the current node to the specified node temporarily, just set `_scope_` to the scope of the specified node in `cfg`.

```python
model = MODELS.build(cfg=dict(type='LogSoftmax', _scope_='mmalpha'))
output = model(input)
# call LogSoftmax.forward
print(output)
```
