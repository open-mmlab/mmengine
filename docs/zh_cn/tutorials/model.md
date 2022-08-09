# 模型

基于 Pytorch 构建模型时，我们会选择 [nn.Module](https://pytorch.org/docs/stable/nn.html?highlight=nn%20module#module-torch.nn.modules) 作为模型的基类，让模型能够轻松实现：

- 导出/加载/遍历模型参数，将模型参数转移至指定设备，设置模型的训练、测试状态等功能。
- `nn.Module` 能够将参数导出后传给接优化器[optimizer](https://pytorch.org/docs/stable/optim.html?highlight=optimizer#torch.optim.Optimizer)实现自动化的参数更新。
- 对接 [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html?highlight=distributeddataparallel#torch.nn.parallel.DistributedDataParallel)
  实现分布式训练。
- `nn.Mdoule` 能够将参数导出后，对接参数初始化模块 [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html?highlight=kaiming#torch.nn.init.kaiming_normal_)，轻松实现指定策略的参数初始化。

等一系列功能。正如上面提到的，`nn.Module` 对接参数初始化模块、优化器实现参数初始化、自动化的参数更新已经成为了使用 `nn.Module`
的标准流程，因此 MMEngine 在 `nn.Module` 的基础上做了进一步的抽象出了 基础模块（`BaseModule`） 和基础模型
（`BaseModel`），前者用于配置模型初始化策略，后者定义了管理模型训练、验证、测试、推理的基本流程。

## 基础模块（`BaseModule`）

MMEngine 抽象出基础模块来配置模型初始化相关的参数。基础模块继承自 `nn.Module`，不仅具备 `nn.Module`
的基本功能，还能根据传参实现相应的参数初始化逻辑。我们可以让模型继承基础模块，通过配置 `init_cfg`
实现自定义的参数初始化逻辑。

### 加载预训练权重

继承自基础模块的模型可以通过初始化阶段配置 `init_cfg`， 让后续模型在调用 `init_weights`
时加载预训练权重:

```python
import torch
import torch.nn as nn

from mmengine.model import BaseModule


class ToyNet(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv1 = nn.Linear(1, 1)

# 保存预训练权重
toy_net = ToyNet()
torch.save(toy_net.state_dict(), './pretrained.pth')
pretrained = './pretrained.pth'

# 配置加载预训练权重的初始化方式
toy_net = ToyNet(init_cfg=dict(
    type='Pretrained', checkpoint=pretrained))
# 加载权重
toy_net.init_weights()
```

当 `init_cfg` 是一个字典时，`type` 字段就表示一种初始化策略，上例中的 `Pretrained` 就表示
`PretrainedInit` 类，并且被注册到 `WEIGHT_INITIALIZERS` [注册器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html)。
`checkpoint` 是 `PretrainedInit` 的参数，用于指定模型加载的路径，可以是可以是本地磁盘路径，也可以是
url。

### 初始化策略

**1.按类型初始化**

有时候我们可能需要用不同的初始化策略去初始化不同模块，例如对卷积使用 `Kaiming` 初始化，对线性层使用 `Xavier`
初始化。此时我们可以让 `init_cfg` 是一个列表，其中的每一个元素都表示对某些层使用特定的初始化策略。

```python
import torch.nn as nn

from mmengine.model import BaseModule


class ToyNet(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.linear = nn.Linear(1, 1)
        self.conv = nn.Conv2d(1, 1, 1)

toy_net = ToyNet(
    init_cfg=[
        dict(type='Kaiming', layer='Conv2d'),
        dict(type='Xavier', layer='Linear')
    ],
)
toy_net.init_weights()
```

类似的，`layer` 参数也可以是一个列表，表示该初始化策略会作用于多个 `layer`

```python
toy_net = ToyNet(
    init_cfg=[
        dict(type='Kaiming', layer=['Conv2d', 'Linear'])
    ],
)
toy_net.init_weights()
```

**2.细粒度的初始化**

有时我们需要对同一类型的不同模块做不同初始化策略，例如我们有 `conv1` 和 `conv2` 两个类型同样为 `Conv2d`
的模块，需要二者的初始化方式分别为 `Kaiming` 初始化和 `Xavier` 初始化，这时候我们就需要配置 override 参数：

```python
import torch.nn as nn

from mmengine.model import BaseModule


class ToyNet(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)

toy_net = ToyNet(
    init_cfg=[
        dict(type='Kaiming', layer=['Conv2d'],
             override=dict(name='conv2', type='Xavier')),
    ],
)
toy_net.init_weights()
```

`override` 可以理解成一个二级的 `init_cfg`， 他同样可以是 `list` 或者 `dict`，也可以通过 `type`
字段指定初始化策略。不同的是 `override` 必须制定 `name`，`name` 相当于 `override`
参数的作用域，如上例中，`override` 的作用域为 `toy_net.conv2`， 我们
我们会以 `Xavier` 初始化策略初始化 `toy_net.conv2` 下的所有参数，而不会影响作用域以外的模块。

目前 MMEngine 支持以下初始化策略：

| Initializer        |    注册名    | 功能                                                                                    |
| :----------------- | :----------: | :-------------------------------------------------------------------------------------- |
| `constant_init`    |   Constant   | 将 weight 和 bias 初始化为指定常量                                                      |
| `XavierInit`       |    Xavier    | 将 weight 和 bias 以 [Xavier] 方式初始化                                                |
| `NormalInit`       |    Normal    | 将 weight 和 bias 以正太分布的方式初始化                                                |
| `TruncNormalInit`  | TruncNormal  | 将 weight 和 bias 以被截断的正太分布的方式初始化，参数 a 和 b 为正太分布的有效区域      |
| `UniformInit`      |   Uniform    | 将 weight 和 bias 以均匀分布的形式初始化，参数 a 和 b 为均匀分布的范围                  |
| `KaimingInit`      |   Kaiming    | 将 weight 和 bias 以 [Kaiming] 的方式初始化。                                           |
| `Caffe2XavierInit` | Caffe2Xavier | Caffe2 中 Xavier 初始化策略，在 Pytorch 中对应 `fan_in`, `normal` 模式的 Kaiming 初始化 |
| `PretrainedInit`   |  Pretrained  | 加载预训练权重                                                                          |

### 自定义的初始化策略

尽管 `init_cfg` 能够控制各个模块的初始化方式，但是在不扩展 `WEIGHT_INITIALIZERS`
的情况下，是无法初始化一些自定义模块的。对于这种情况，我们需要让自定义子模块实现 `init_weights` 方法。模型调用 `init_weights`
时，会链式的调用所有子模块的 `init_weights`。

```python
import torch.nn as nn
import torch

from mmengine.model import BaseModule


class ToyConv:
    def __init__(self):
        super().__init__()
        self.custom_weight = nn.Parameter(torch.empty(1, 1, 1, 1))
        self.custom_bias = nn.Parameter(torch.empty(1))

    def init_weights(self):
        with torch.no_grad():
            self.custom_weight = self.custom_weight.fill_(1)
            self.custom_bias = self.custom_bias.fill_(0)


class ToyNet(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)
        self.custom_conv = ToyConv()


toy_net = ToyNet(
    init_cfg=[
        dict(type='Kaiming', layer=['Conv2d'],
             override=dict(name='conv2', type='Xavier')),
    ],
)
toy_net.init_weights()
```

上例中，toy_net 会递归调用 `ToyConv` 的 `init_weights`，分别将参数 `custom_weight` 和
`custom_bias` 初始化为 1 和 0。

看到这里可能会疑惑，现在既可以通过配置 `init_cfg` 来选择初始化策略，也可以通过实现 `init_weights`
指定自定义的初始化方式，我们到底应该选择哪种方式呢？这里我们对二者的功能做了进一步的区分：

1. 配置 `init_cfg` 控制初始化

- 通常用于初始化一些比较底层的模块，例如卷积、线性层等。如果想通过 `init_cfg` 配置自定义模块的初始化方式，则需要将相应的初始化策略注册到
  `WEIGHT_INITIALIZERS` 里。
- 动态初始化，我们可以通过配置 `init_cfg` 动态的选择模型初始化方式。

2. 实现 `init_weights` 初始化

- 通常用于初始化一些自定义模块。相比于 `init_cfg` 粒度更粗。
- 有些模块可能不需要动态初始化特性，这时可以直接实现 `init_weights` 方法，而无需通过 `init_cfg` 配置。
- `init_weights` 的优先级更高，会**覆盖** `init_cfg` 中已经初始化后的结果。

```{note}
init_weights 的优先级比 `init_cfg` 高，如果 `init_cfg` 中已经指定了某个模块的初始化方式
```

```{note}
执行器会在 train() 函数中调用 init_weights。
```

## 基础模型

基于标准化的模型初始化流程，MMEngine 抽象出
`BaseModule`，让我们能够更加灵活的选择模型的初始化方式，同样的，基于标准化的模型训练流程，MMEngine
抽象出基础模型（`BaseModel`），让模型的核心代码更加集中，方便阅读和理解。尽管 Pytorch 对 `nn.Module`
的接口没有任何要求，但是我们认为一套标准的模型接口能够让我们更好的理解代码，因此 MMEngine 要求，只有符合基础模型接口约定的模型才能
在[执行器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html)中训练起来。

### 定义基础模型

#### Pytorch 标准训练流程

```python

```

[kaiming]: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
[xavier]: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
