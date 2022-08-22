# 初始化

基于 Pytorch 构建模型时，我们通常会选择 [nn.Module](https://pytorch.org/docs/stable/nn.html?highlight=nn%20module#module-torch.nn.modules) 作为模型的基类，搭配使用 Pytorch 的初始化模块 [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html?highlight=kaiming#torch.nn.init.kaiming_normal_)，完成模型的初始化。`MMEngine` 在此基础上抽象出基础模块（BaseModule） 和模块初始化函数，让我们能够更加方便灵活地初始化模型参数。

## 函数式初始化

Pytorch 的初始化模块提供了一系列**参数**初始化功能，例如我们想用正态分布初始化一个卷积层的权重（weight）层，常数初始化卷积的偏置（`bias`）层，我们就需要分别去初始化卷积的权重和偏置。

```python
from torch.nn.init import normal_, constant_
import torch.nn as nn

model = nn.Conv2d(1, 1, 1)
normal_(model.weight, mean=0, std=0.01)
constant_(model.bias, val=0)
```

```
Parameter containing:
tensor([0.], requires_grad=True)
```

上述流程实际上是卷积正态分布初始化的标准流程，MMEngine 在此基础上做了进一步的简化，实现了一系列常用**模块**的初始化函数。相比于上述 Pytorch 初始化模块的用法，MMEngine 提供的初始化函数可以直接初始化卷积模块，一行代码就能将权重按照正态分布初始化，偏置初始化为 0：

```python
from mmengine.model import normal_init

normal_init(model, mean=0, std=0.01, bias=0)
```

类似地，我们也可以用 [Kaiming](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) 初始化和 [Xavier](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) 初始化：

```python
from mmengine.model import kaiming_init, xavier_init

kaiming_init(model)
normal_init(model)
```

目前 MMEngine 提供了以下初始化函数：

| 初始化器                                                              | 功能                                                                                                                               |
| :-------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| [constant_init](../api.html#mmengine.model.constant_init)             | 将 weight 和 bias 初始化为指定常量，通常用于初始化卷积                                                                             |
| [xavier_init](../api.html#mmengine.model.xavier_init)                 | 将 weight 以 `Xavier` 方式初始化，将 bias 初始化成指定常量，通常用于初始化卷积                                                     |
| [normal_init](../api.html#mmengine.model.normal_init)                 | 将 weight 以正态分布的方式初始化，将 bias 初始化成指定常量，通常用于初始化卷积                                                     |
| [trunc_normal_init](../api.html#mmengine.model.trunc_normal_init)     | 将 weight 以被截断的正态分布的方式初始化，参数 a 和 b 为正态分布的有效区域；将 bias 初始化成指定常量，通常用于初始化 `transformer` |
| [uniform_init](../api.html#mmengine.model.uniform_init)               | 将 weight 以均匀分布的方式初始化，参数 a 和 b 为均匀分布的范围；将 bias 初始化为指定常量，通常用于初始化卷积                       |
| [kaiming_init](../api.html#mmengine.model.kaiming_init)               | 将 weight 以 `Kaiming` 方式初始化，将 bias 初始化成指定常量，通常用于初始化卷积                                                    |
| [caffe2_xavier_init](../api.html#mmengine.model.caffe2_xavier_init)   | Caffe2 中 Xavier 初始化方式，在 Pytorch 中对应 `fan_in`, `normal` 模式的 `Kaiming` 初始化，通常用于初始化卷积                      |
| [bias_init_with_prob](../api.html#mmengine.model.bias_init_with_prob) | 以概率值的形式初始化 bias                                                                                                          |

## 配置式初始化

`MMEngine` 为了让模型能够更加灵活的初始化权重，抽象出了模块基类 `BaseModule`。模块基类继承自 `nn.Module`，不仅具备 `nn.Module` 的基础功能时，还能够进行参数初始化。我们只需要让模型继承 `BaseModule`，并在实例化阶段配置构造参数 `init_cfg`，就能够控制模型中任意组件的初始化方式。`MMEngine` 将[函数式初始化](函数式初始化)中提到的函数进一步抽象成初始化器，作为 `init_cfg` 的配置项，目前实现了以下初始化器：

| 初始化器                                                        |    注册名    | 功能                                                                                                                               |
| :-------------------------------------------------------------- | :----------: | :--------------------------------------------------------------------------------------------------------------------------------- |
| [ConstantInit](../api.html#mmengine.model.ConstantInit)         |   Constant   | 将 weight 和 bias 初始化为指定常量，通常用于初始化卷积                                                                             |
| [XavierInit](../api.html#mmengine.model.XavierInit)             |    Xavier    | 将 weight `Xavier` 方式初始化，将 bias 初始化成指定常量，通常用于初始化卷积                                                        |
| [NormalInit](../api.html#mmengine.model.NormalInit)             |    Normal    | 将 weight 以正态分布的方式初始化，将 bias 初始化成指定常量，通常用于初始化卷积                                                     |
| [TruncNormalInit](../api.html#mmengine.model.TruncNormalInit)   | TruncNormal  | 将 weight 以被截断的正态分布的方式初始化，参数 a 和 b 为正态分布的有效区域；将 bias 初始化成指定常量，通常用于初始化 `transformer` |
| [UniformInit](../api.html#mmengine.model.UniformInit)           |   Uniform    | 将 weight 以均匀分布的方式初始化，参数 a 和 b 为均匀分布的范围；将 bias 初始化为指定常量，通常用于初始化卷积                       |
| [KaimingInit](../api.html#mmengine.model.KaimingInit)           |   Kaiming    | 将 weight 以 `Kaiming` 的方式初始化，将 bias 初始化成指定常量，通常用于初始化卷积                                                  |
| [Caffe2XavierInit](../api.html#mmengine.model.Caffe2XavierInit) | Caffe2Xavier | Caffe2 中 Xavier 初始化方式，在 Pytorch 中对应 `fan_in`, `normal` 模式的 `Kaiming` 初始化，，通常用于初始化卷                      |
| [PretrainedInit](../api.html#mmengine.model.PretrainedInit)     |  Pretrained  | 加载预训练权重                                                                                                                     |

模块基类接受 `init_cfg` 参数，继承自模块基类的模型可以在 `init_cfg` 里指定初始化器，选择相应的初始化方式。

假设我们定义了模型 `ToyNet`，它继承自模块基类（`BaseModule`），并在 `__init__` 里调用了 `BaseModule` 的 `__init__`。此时我们可以在 `ToyNet` 初始化时传入 `init_cfg` 参数来选择模型的初始化方式，`ToyNet` 实例化后再调用 `init_weights` 方法，完成权重的初始化。

### 使用预训练权重初始化

当 `init_cfg` 是一个字典时，`type` 字段就表示一种初始化器，它需要被注册到 `WEIGHT_INITIALIZERS` [注册器](./registry.md)。我们可以通过指定 `init_cfg=dict(type='Pretrained', checkpoint='path/to/ckpt')` 来加载预训练权重，其中 `Pretrained` 为 `PretrainedInit` 初始化器的缩写，这个映射名由 `WEIGHT_INITIALIZERS` 维护；`checkpoint` 是 `PretrainedInit` 的初始化参数，用于指定权重的加载路径，它可以是本地磁盘路径，也可以是 URL。

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
toy_net = ToyNet(init_cfg=dict(type='Pretrained', checkpoint=pretrained))
# 加载权重
toy_net.init_weights()
```

```
08/19 16:50:24 - mmengine - INFO - load model from: ./pretrained.pth
08/19 16:50:24 - mmengine - INFO - local loads checkpoint from path: ./pretrained.pth
```

### 常用的初始化方式

和使用 `PretrainedInit` 初始化器类似，如果我们想对卷积做 `Kaiming` 初始化，需要令 `init_cfg=dict(type='Kaiming', layer='Conv2d')`。这样模型初始化时，就会以 `Kaiming` 初始化的方式来初始化类型为 `Conv2d` 的模块。

有时候我们可能需要用不同的初始化方式去初始化不同类型的模块，例如对卷积使用 `Kaiming` 初始化，对线性层使用 `Xavier`
初始化。此时我们可以使 `init_cfg` 成为一个列表，，其中的每一个元素都表示对某些层使用特定的初始化方式。

```python
import torch.nn as nn

from mmengine.model import BaseModule


class ToyNet(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.linear = nn.Linear(1, 1)
        self.conv = nn.Conv2d(1, 1, 1)


# 对卷积做 Kaiming 初始化，现性层做 Xavier 初始化
toy_net = ToyNet(
    init_cfg=[
        dict(type='Kaiming', layer='Conv2d'),
        dict(type='Xavier', layer='Linear')
    ], )
toy_net.init_weights()
```

```
08/19 16:50:24 - mmengine - INFO -
linear.weight - torch.Size([1, 1]):
XavierInit: gain=1, distribution=normal, bias=0

08/19 16:50:24 - mmengine - INFO -
linear.bias - torch.Size([1]):
XavierInit: gain=1, distribution=normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv.weight - torch.Size([1, 1, 1, 1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0
```

类似地，`layer` 参数也可以是一个列表，表示列表中的多种不同的 `layer` 均使用 `type` 指定的初始化方式

```python
# 对卷积和线性层做 Kaiming 初始化
toy_net = ToyNet(init_cfg=[dict(type='Kaiming', layer=['Conv2d', 'Linear'])], )
toy_net.init_weights()
```

```
08/19 16:50:24 - mmengine - INFO -
linear.weight - torch.Size([1, 1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0

08/19 16:50:24 - mmengine - INFO -
linear.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv.weight - torch.Size([1, 1, 1, 1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0
```

#### 更细粒度的初始化

有时同一类型的不同模块有不同初始化方式，例如现在有 `conv1` 和 `conv2` 两个模块，他们的类型均为 `Conv2d`
。我们需要对 conv1 进行 `Kaiming` 初始化，conv2 进行 `Xavier` 初始化，则可以通过配置 `override` 参数来满足这样的需求：

```python
import torch.nn as nn

from mmengine.model import BaseModule


class ToyNet(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)


# 对 conv1 做卷积初始化，对 从 conv2 做 Xavier 初始化
toy_net = ToyNet(
    init_cfg=[
        dict(
            type='Kaiming',
            layer=['Conv2d'],
            override=dict(name='conv2', type='Xavier')),
    ], )
toy_net.init_weights()
```

```
08/19 16:50:24 - mmengine - INFO -
conv1.weight - torch.Size([1, 1, 1, 1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv1.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv2.weight - torch.Size([1, 1, 1, 1]):
XavierInit: gain=1, distribution=normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv2.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0
```

`override` 可以理解成一个嵌套的 `init_cfg`， 他同样可以是 `list` 或者 `dict`，也需要通过 `type`
字段指定初始化方式。不同的是 `override` 必须制定 `name`，`name` 相当于 `override`
的作用域，如上例中，`override` 的作用域为 `toy_net.conv2`，
我们会以 `Xavier` 初始化方式初始化 `toy_net.conv2` 下的所有参数，而不会影响作用域以外的模块。

### 自定义的初始化方式

尽管 `init_cfg` 能够控制各个模块的初始化方式，但是在不扩展 `WEIGHT_INITIALIZERS`
的情况下，我们是无法初始化一些自定义模块的，例如表格中提到的大多数初始化器，都需要对应的模块有 `weight` 和 `bias` 属性 。对于这种情况，我们建议让自定义模块实现 `init_weights` 方法。模型调用 `init_weights`
时，会链式地调用所有子模块的 `init_weights`。

假设我们定义了以下模块：

- 继承自 `nn.Module` 的 `ToyConv`，实现了 `init_weights` 方法，让 `custom_weight` 初始化为 1，`custom_bias` 初始化为 0
- 继承自模块基类的模型 `ToyNet`，且含有 `ToyConv` 子模块。

我们在调用 `ToyNet` 的 `init_weights` 方法时，会链式的调用的子模块 `ToyConv` 的 `init_weights` 方法，实现自定义模块的初始化。

```python
import torch
import torch.nn as nn

from mmengine.model import BaseModule


class ToyConv(nn.Module):

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
        dict(
            type='Kaiming',
            layer=['Conv2d'],
            override=dict(name='conv2', type='Xavier'))
    ])
# 链式调用 `ToyConv.init_weights()`，以自定义的方式初始化
toy_net.init_weights()
```

```
08/19 16:50:24 - mmengine - INFO -
conv1.weight - torch.Size([1, 1, 1, 1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv1.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv2.weight - torch.Size([1, 1, 1, 1]):
XavierInit: gain=1, distribution=normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv2.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0

08/19 16:50:24 - mmengine - INFO -
custom_conv.custom_weight - torch.Size([1, 1, 1, 1]):
Initialized by user-defined `init_weights` in ToyConv

08/19 16:50:24 - mmengine - INFO -
custom_conv.custom_bias - torch.Size([1]):
Initialized by user-defined `init_weights` in ToyConv
```

### 小结

最后我们对 `init_cfg` 和 `init_weights` 两种初始化方式做一些总结：

**1. 配置 `init_cfg` 控制初始化**

- 通常用于初始化一些比较底层的模块，例如卷积、线性层等。如果想通过 `init_cfg` 配置自定义模块的初始化方式，需要将相应的初始化器注册到 `WEIGHT_INITIALIZERS` 里。
- 动态初始化特性，初始化方式随 `init_cfg` 的值改变。

**2. 实现 `init_weights` 方法**

- 通常用于初始化自定义模块。相比于 `init_cfg` 的自定义初始化，实现 `init_weights` 方法更加简单，无需注册。然而，它的灵活性不及 `init_cfg`，无法动态地指定任意模块的初始化方式。

```{note}
init_weights 的优先级比 `init_cfg` 高，如果 `init_cfg` 中已经指定了某个模块的初始化方式
```

```{note}
执行器会在 train() 函数中调用 init_weights。
```
