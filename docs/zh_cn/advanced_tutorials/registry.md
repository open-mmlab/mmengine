# 注册器（Registry）

OpenMMLab 的算法库支持了丰富的算法和数据集，因此实现了很多功能相近的模块。例如 ResNet 和 SE-ResNet 的算法实现分别基于 `ResNet` 和 `SEResNet` 类，这些类有相似的功能和接口，都属于算法库中的模型组件。为了管理这些功能相似的模块，MMEngine 实现了 [注册器](mmengine.registry.Registry)。OpenMMLab 大多数算法库均使用注册器来管理它们的代码模块，包括 [MMDetection](https://github.com/open-mmlab/mmdetection)， [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)，[MMPretrain](https://github.com/open-mmlab/mmpretrain) 和 [MMagic](https://github.com/open-mmlab/mmagic) 等。

## 什么是注册器

MMEngine 实现的[注册器](mmengine.registry.Registry)可以看作一个映射表和模块构建方法（build function）的组合。映射表维护了一个字符串到**类或者函数的映射**，使得用户可以借助字符串查找到相应的类或函数，例如维护字符串 `"ResNet"` 到 `ResNet` 类或函数的映射，使得用户可以通过 `"ResNet"` 找到 `ResNet` 类；而模块构建方法则定义了如何根据字符串查找到对应的类或函数以及如何实例化这个类或者调用这个函数，例如，通过字符串 `"bn"` 找到 `nn.BatchNorm2d` 并实例化 `BatchNorm2d` 模块；又或者通过字符串 `"build_batchnorm2d"` 找到 `build_batchnorm2d` 函数并返回该函数的调用结果。MMEngine 中的注册器默认使用 [build_from_cfg](mmengine.registry.build_from_cfg) 函数来查找并实例化字符串对应的类或者函数。

一个注册器管理的类或函数通常有相似的接口和功能，因此该注册器可以被视作这些类或函数的抽象。例如注册器 `MODELS` 可以被视作所有模型的抽象，管理了 `ResNet`，`SEResNet` 和 `RegNetX` 等分类网络的类以及 `build_ResNet`,  `build_SEResNet` 和 `build_RegNetX` 等分类网络的构建函数。

## 入门用法

使用注册器管理代码库中的模块，需要以下三个步骤。

1. 创建注册器
2. 创建一个用于实例化类的构建方法（可选，在大多数情况下可以只使用默认方法）
3. 将模块加入注册器中

假设我们要实现一系列激活模块并且希望仅修改配置就能够使用不同的激活模块而无需修改代码。

首先创建注册器，

```python
from mmengine import Registry
# scope 表示注册器的作用域，如果不设置，默认为包名，例如在 mmdetection 中，它的 scope 为 mmdet
# locations 表示注册在此注册器的模块所存放的位置，注册器会根据预先定义的位置在构建模块时自动 import
ACTIVATION = Registry('activation', scope='mmengine', locations=['mmengine.models.activations'])
```

`locations` 指定的模块 `mmengine.models.activations` 对应了 `mmengine/models/activations.py` 文件。在使用注册器构建模块的时候，ACTIVATION 注册器会自动从该文件中导入实现的模块。因此，我们可以在 `mmengine/models/activations.py` 文件中实现不同的激活函数，例如 `Sigmoid`，`ReLU` 和 `Softmax`。

```python
import torch.nn as nn

# 使用注册器管理模块
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

使用注册器管理模块的关键步骤是，将实现的模块注册到注册表 `ACTIVATION` 中。通过 `@ACTIVATION.register_module()` 装饰所实现的模块，字符串和类或函数之间的映射就可以由 `ACTIVATION` 构建和维护，我们也可以通过 `ACTIVATION.register_module(module=ReLU)` 实现同样的功能。

通过注册，我们就可以通过 `ACTIVATION` 建立字符串与类或函数之间的映射，

```python
print(ACTIVATION.module_dict)
# {
#     'Sigmoid': __main__.Sigmoid,
#     'ReLU': __main__.ReLU,
#     'Softmax': __main__.Softmax
# }
```

```{note}
只有模块所在的文件被导入时，注册机制才会被触发，用户可以通过三种方式将模块添加到注册器中：

1. 在 ``locations`` 指向的文件中实现模块。注册器将自动在预先定义的位置导入模块。这种方式是为了简化算法库的使用，以便用户可以直接使用 ``REGISTRY.build(cfg)``。

2. 手动导入文件。常用于用户在算法库之内或之外实现新的模块。

3. 在配置中使用 ``custom_imports`` 字段。 详情请参考[导入自定义Python模块](config.md#导入自定义-python-模块)。
```

模块成功注册后，我们可以通过配置文件使用这个激活模块。

```python
import torch

input = torch.randn(2)

act_cfg = dict(type='Sigmoid')
activation = ACTIVATION.build(act_cfg)
output = activation(input)
# call Sigmoid.forward
print(output)
```

如果我们想使用 `ReLU`，仅需修改配置。

```python
act_cfg = dict(type='ReLU', inplace=True)
activation = ACTIVATION.build(act_cfg)
output = activation(input)
# call ReLU.forward
print(output)
```

如果我们希望在创建实例前检查输入参数的类型（或者任何其他操作），我们可以实现一个构建方法并将其传递给注册器从而实现自定义构建流程。

创建一个构建方法，

```python

def build_activation(cfg, registry, *args, **kwargs):
    cfg_ = cfg.copy()
    act_type = cfg_.pop('type')
    print(f'build activation: {act_type}')
    act_cls = registry.get(act_type)
    act = act_cls(*args, **kwargs, **cfg_)
    return act
```

并将 `build_activation` 传递给 `build_func` 参数

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
在这个例子中，我们演示了如何使用参数 `build_func` 自定义构建类的实例的方法。
该功能类似于默认的 `build_from_cfg` 方法。在大多数情况下，使用默认的方法就可以了。
```

MMEngine 的注册器除了可以注册类，也可以注册函数。

```python
FUNCTION = Registry('function', scope='mmengine')

@FUNCTION.register_module()
def print_args(**kwargs):
    print(kwargs)

func_cfg = dict(type='print_args', a=1, b=2)
func_res = FUNCTION.build(func_cfg)
```

## 进阶用法

MMEngine 的注册器支持层级注册，利用该功能可实现跨项目调用，即可以在一个项目中使用另一个项目的模块。虽然跨项目调用也有其他方法的可以实现，但 MMEngine 注册器提供了更为简便的方法。

为了方便跨库调用，MMEngine 提供了 22 个根注册器：

- RUNNERS: Runner 的注册器
- RUNNER_CONSTRUCTORS: Runner 的构造器
- LOOPS: 管理训练、验证以及测试流程，如 `EpochBasedTrainLoop`
- HOOKS: 钩子，如 `CheckpointHook`, `ParamSchedulerHook`
- DATASETS: 数据集
- DATA_SAMPLERS: `DataLoader` 的 `Sampler`，用于采样数据
- TRANSFORMS: 各种数据预处理，如 `Resize`, `Reshape`
- MODELS: 模型的各种模块
- MODEL_WRAPPERS: 模型的包装器，如 `MMDistributedDataParallel`，用于对分布式数据并行
- WEIGHT_INITIALIZERS: 权重初始化的工具
- OPTIMIZERS: 注册了 PyTorch 中所有的 `Optimizer` 以及自定义的 `Optimizer`
- OPTIM_WRAPPER: 对 Optimizer 相关操作的封装，如 `OptimWrapper`，`AmpOptimWrapper`
- OPTIM_WRAPPER_CONSTRUCTORS: optimizer wrapper 的构造器
- PARAM_SCHEDULERS: 各种参数调度器，如 `MultiStepLR`
- METRICS: 用于计算模型精度的评估指标，如 `Accuracy`
- EVALUATOR: 用于计算模型精度的一个或多个评估指标
- TASK_UTILS: 任务强相关的一些组件，如 `AnchorGenerator`, `BboxCoder`
- VISUALIZERS: 管理绘制模块，如 `DetVisualizer` 可在图片上绘制预测框
- VISBACKENDS: 存储训练日志的后端，如 `LocalVisBackend`, `TensorboardVisBackend`
- LOG_PROCESSORS: 控制日志的统计窗口和统计方法，默认使用 `LogProcessor`，如有特殊需求可自定义 `LogProcessor`
- FUNCTIONS: 注册了各种函数，如 Dataloader 中传入的 `collate_fn`
- INFERENCERS: 注册了各种任务的推理器，如 `DetInferencer`，负责检测任务的推理

### 调用父节点的模块

`MMEngine` 中定义模块 `RReLU`，并往 `MODELS` 根注册器注册。

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

假设有个项目叫 `MMAlpha`，它也定义了 `MODELS`，并设置其父节点为 `MMEngine` 的 `MODELS`，这样就建立了层级结构。

```python
from mmengine import Registry, MODELS as MMENGINE_MODELS

MODELS = Registry('model', parent=MMENGINE_MODELS, scope='mmalpha', locations=['mmalpha.models'])
```

下图是 `MMEngine` 和 `MMAlpha` 的注册器层级结构。

<div align="center">
  <img src="https://user-images.githubusercontent.com/58739961/185307159-26dc5771-df77-4d03-9203-9c4c3197befa.png"/>
</div>

可以调用 [count_registered_modules](mmengine.registry.count_registered_modules) 函数打印已注册到 MMEngine 的模块以及层级结构。

```python
from mmengine.registry import count_registered_modules

count_registered_modules()
```

在 `MMAlpha` 中定义模块 `LogSoftmax`，并往 `MMAlpha` 的 `MODELS` 注册。

```python
@MODELS.register_module()
class LogSoftmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()

    def forward(self, x):
        print('call LogSoftmax.forward')
        return x
```

在 `MMAlpha` 中使用配置调用 `LogSoftmax`

```python
model = MODELS.build(cfg=dict(type='LogSoftmax'))
```

也可以在 `MMAlpha` 中调用父节点 `MMEngine` 的模块。

```python
model = MODELS.build(cfg=dict(type='RReLU', lower=0.2))
# 也可以加 scope
model = MODELS.build(cfg=dict(type='mmengine.RReLU'))
```

如果不加前缀，`build` 方法首先查找当前节点是否存在该模块，如果存在则返回该模块，否则会继续向上查找父节点甚至祖先节点直到找到该模块，因此，如果当前节点和父节点存在同一模块并且希望调用父节点的模块，我们需要指定 `scope` 前缀。

```python
import torch

input = torch.randn(2)
output = model(input)
# call RReLU.forward
print(output)
```

### 调用兄弟节点的模块

除了可以调用父节点的模块，也可以调用兄弟节点的模块。

假设有另一个项目叫 `MMBeta`，它和 `MMAlpha` 一样，定义了 `MODELS` 以及设置其父节点为 `MMEngine` 的 `MODELS`。

```python
from mmengine import Registry, MODELS as MMENGINE_MODELS

MODELS = Registry('model', parent=MMENGINE_MODELS, scope='mmbeta')
```

下图是 MMEngine，MMAlpha 和 MMBeta 的注册器层级结构。

<div align="center">
  <img src="https://user-images.githubusercontent.com/58739961/185307738-9ddbce2d-f8b5-40c4-bf8f-603830ccc0dc.png"/>
</div>

在 `MMBeta` 中调用兄弟节点 `MMAlpha` 的模块，

```python
model = MODELS.build(cfg=dict(type='mmalpha.LogSoftmax'))
output = model(input)
# call LogSoftmax.forward
print(output)
```

调用兄弟节点的模块需要在 `type` 中指定 `scope` 前缀，所以上面的配置需要加前缀 `mmalpha`。

如果需要调用兄弟节点的数个模块，每个模块都加前缀，这需要做大量的修改。于是 `MMEngine` 引入了 [DefaultScope](mmengine.registry.DefaultScope)，`Registry` 借助它可以很方便地支持临时切换当前节点为指定的节点。

如果需要临时切换当前节点为指定的节点，只需在 `cfg` 设置 `_scope_` 为指定节点的作用域。

```python
model = MODELS.build(cfg=dict(type='LogSoftmax', _scope_='mmalpha'))
output = model(input)
# call LogSoftmax.forward
print(output)
```
