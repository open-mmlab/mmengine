# 模型

MMEngine 在 `nn.Module` 的基础上进一步抽象出了模块基类（`BaseModule`） 和模型基类
（`BaseModel`），前者用于配置模型初始化方式，后者定义了模型训练、验证、测试、推理的基本流程。

## 模块基类（BaseModule）

神经网络模型有很多初始化方式，例如 [Xavier] 初始化，[Kaiming] 初始化。`MMEngine` 将不同的初始化方式抽象成初始化器，目前实现了以下初始化器：

| 初始化器           |    注册名    | 功能                                                                                    |
| :----------------- | :----------: | :-------------------------------------------------------------------------------------- |
| `ConstantInit`     |   Constant   | 将 weight 和 bias 初始化为指定常量                                                      |
| `XavierInit`       |    Xavier    | 将 weight 和 bias 以 [Xavier] 方式初始化                                                |
| `NormalInit`       |    Normal    | 将 weight 和 bias 以正态分布的方式初始化                                                |
| `TruncNormalInit`  | TruncNormal  | 将 weight 和 bias 以被截断的正态分布的方式初始化，参数 a 和 b 为正态分布的有效区域      |
| `UniformInit`      |   Uniform    | 将 weight 和 bias 以均匀分布的方式初始化，参数 a 和 b 为均匀分布的范围                  |
| `KaimingInit`      |   Kaiming    | 将 weight 和 bias 以 [Kaiming] 的方式初始化。                                           |
| `Caffe2XavierInit` | Caffe2Xavier | Caffe2 中 Xavier 初始化方式，在 Pytorch 中对应 `fan_in`, `normal` 模式的 Kaiming 初始化 |
| `PretrainedInit`   |  Pretrained  | 加载预训练权重                                                                          |

模块基类接受 `init_cfg` 参数，继承自模块基类的模型可以在 `init_cfg` 里指定初始化器，选择相应的初始化方式。

### 权重初始化

假设我们定义了模型 `ToyNet`，它继承自模块基类（`BaseModule`），并在 `__init__` 里调用了 `BaseModule` 的 `__init__`。此时我们可以在模型初始化阶段指定 `init_cfg` 来选择模型的初始化方式，然后在 `ToyNet` 实例化后调用 `init_weights` 方法，完成权重的初始化。

#### 加载预训练权重

`init_cfg` 是一个字典时，`type` 字段就表示一种初始化器，它需要被注册到 `WEIGHT_INITIALIZERS` [注册器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/registry.html)。我们可以通过指定 `init_cfg=dict(type='Pretrained', checkpoint='path/to/ckpt')` 来加载预训练权重，其中 `Pretrained` 为 `PretrainedInit` 初始化器的缩写，这个映射名由 `WEIGHT_INITIALIZERS` 维护；`checkpoint` 是 `PretrainedInit` 的初始化参数，用于指定权重的加载路径，它可以是本地磁盘路径，也可以是 URL。

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

# 08/16 20:51:24 - mmengine - INFO - load model from: ./pretrained.pth
# 08/16 20:51:24 - mmengine - INFO - local loads checkpoint from path: ./pretrained.pth
```

#### 常用的初始化方式

和使用 `PretrainedInit` 初始化器类似，如果我们想对卷积做 `Kaiming` 初始化，需要令 `init_cfg=dict(type='Kaiming', layer='Conv2d')`。这样模型初始化时，就会以 `Kaiming` 初始化的方式来初始化类型为 `Conv2d` 的模块。

有时候我们可能需要用不同的初始化方式去初始化不同类型的模块，例如对卷积使用 `Kaiming` 初始化，对线性层使用 `Xavier`
初始化。此时我们可以让 `init_cfg` 是一个列表，其中的每一个元素都表示对某些层使用特定的初始化方式。

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

# 08/16 20:50:36 - mmengine - INFO -
# linear.weight - torch.Size([1, 1]):
# XavierInit: gain=1, distribution=normal, bias=0
#
# 08/16 20:50:36 - mmengine - INFO -
# linear.bias - torch.Size([1]):
# XavierInit: gain=1, distribution=normal, bias=0
#
# 08/16 20:50:36 - mmengine - INFO -
# conv.weight - torch.Size([1, 1, 1, 1]):
# KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0
#
# 08/16 20:50:36 - mmengine - INFO -
# conv.bias - torch.Size([1]):
# KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0
```

类似地，`layer` 参数也可以是一个列表，表示列表中的多种不同的 `layer` 均使用 `type` 指定的初始化方式

```python
toy_net = ToyNet(
    init_cfg=[
        dict(type='Kaiming', layer=['Conv2d', 'Linear'])
    ],
)
toy_net.init_weights()

# 08/16 20:51:58 - mmengine - INFO -
# conv1.weight - torch.Size([1, 1]):
# KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0
#
# 08/16 20:51:58 - mmengine - INFO -
# conv1.bias - torch.Size([1]):
# KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0
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

toy_net = ToyNet(
    init_cfg=[
        dict(type='Kaiming', layer=['Conv2d'],
             override=dict(name='conv2', type='Xavier')),
    ],
)
toy_net.init_weights()

# 08/16 20:52:41 - mmengine - INFO -
# conv1.weight - torch.Size([1, 1, 1, 1]):
# KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0
#
# 08/16 20:52:41 - mmengine - INFO -
# conv1.bias - torch.Size([1]):
# KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0
#
# 08/16 20:52:41 - mmengine - INFO -
# conv2.weight - torch.Size([1, 1, 1, 1]):
# XavierInit: gain=1, distribution=normal, bias=0
#
# 08/16 20:52:41 - mmengine - INFO -
# conv2.bias - torch.Size([1]):
# KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0
```

`override` 可以理解成一个嵌套的 `init_cfg`， 他同样可以是 `list` 或者 `dict`，也需要通过 `type`
字段指定初始化方式。不同的是 `override` 必须制定 `name`，`name` 相当于 `override`
的作用域，如上例中，`override` 的作用域为 `toy_net.conv2`， 我们
我们会以 `Xavier` 初始化方式初始化 `toy_net.conv2` 下的所有参数，而不会影响作用域以外的模块。

### 自定义的初始化方式

尽管 `init_cfg` 能够控制各个模块的初始化方式，但是在不扩展 `WEIGHT_INITIALIZERS`
的情况下，我们是无法初始化一些自定义模块的，例如表格中提到的大多数初始化器，都需要对应的模块有 `weight` 和 `bias` 属性 。对于这种情况，我们建议让自定义模块实现 `init_weights` 方法。模型调用 `init_weights`
时，会链式的调用所有子模块的 `init_weights`。

假设我们定义了以下模块：

- 继承自 `nn.Module` 的 `ToyConv`，实现了 `init_weights` 方法，让 `custom_weight` 初始化为 1，`custom_bias` 初始化为 0
- 继承自模块基类的模型 `ToyNet`，且含有 `ToyConv` 子模块。

我们在调用 `ToyConv` 的 `init_weights` 方法时，会链式的调用的子模块 `ToyConv` 的 `init_weights` 方法，实现自定义模块的初始化。

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
            override=dict(name='conv2', type='Xavier')),
    ], )
toy_net.init_weights()

# 只显示了自定义初始化的日志
...
# 08/16 21:17:35 - mmengine - INFO -
# custom_conv.custom_weight - torch.Size([1, 1, 1, 1]):
# Initialized by user-defined `init_weights` in ToyConv
#
# 08/16 21:17:35 - mmengine - INFO -
# custom_conv.custom_bias - torch.Size([1]):
# The value is the same before and after calling `init_weights` of ToyNet
```

这里我们对 `init_cfg` 和 `init_weights` 两种初始化方式做一些总结：

**1. 配置 `init_cfg` 控制初始化**

- 通常用于初始化一些比较底层的模块，例如卷积、线性层等。如果想通过 `init_cfg` 配置自定义模块的初始化方式，需要将相应的初始化器注册到 `WEIGHT_INITIALIZERS` 里。
- 动态初始化特性，初始化方式随 `init_cfg` 的值改变。

**2. 实现 `init_weights` 方法**

- 通常用于初始化自定义模块。相比于 `init_cfg` 的自定义初始化，实现 `init_weights` 方法更加简单，无需注册，但是没有 `init_cfg` 那么灵活，可以动态的指定任意模块的初始化方式。

```{note}
init_weights 的优先级比 `init_cfg` 高，如果 `init_cfg` 中已经指定了某个模块的初始化方式
```

```{note}
执行器会在 train() 函数中调用 init_weights。
```

## 模型基类（BaseModel）

[执行器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html)接受的模型需要满足一定的接口规范，模型需要实现 `train_step`，`val_step` 和 `test_step` 方法。对于检测、识别、分割一类的深度学习任务，上述方法通常为标准的流程，例如在 `train_step` 里更新参数，返回损失；`val_step` 和 `test_step` 返回预测结果。因此 MMEngine 抽象出模型基类 `BaseModel`，实现了上述接口的标准流程。我们只需要让模型继承自模型基类，并按照一定的规范实现 `forward`，就能让模型在执行器中运行起来。

模型基类继承自模块基类，能够通过配置 `init_cfg` 灵活的选择初始化方式。

### 接口约定

[forward](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.model.BaseModel.forward): `forward` 的入参需要和 [DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) 的输出保持一致 (自定义[数据处理器](#数据处理器datapreprocessor)除外)，如果 `DataLoader` 返回元组类型的数据 `data`，`forward` 需要能够接受 `*data` 的解包后的参数；如果返回字典类型的数据 `data`，`forward` 需要能够接受 `**data` 解包后的参数。 `mode` 参数用于控制 forward 的返回结果：

- `mode='loss'`：`loss` 模式通常在训练阶段启用，并返回一个损失字典。损失字典的 key-value 分别为损失名和可微的 `torch.Tensor`。字典中记录的损失会被用于更新参数和记录日志。模型基类会在 `train_step` 方法中调用该模式的 `forward`。
- `mode='predict'`： `predict` 模式通常在验证、测试阶段启用，并返回列表/元组型式的预测结果，预测结果需要和 [process](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.evaluator.Evaluator.process) 接口的参数相匹配。OpenMMLab 系列算法对 `predict` 模式的输出有着更加严格的约定，需要输出列表型式的[数据元素](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/data_element.html)。模型基类会在 `val_step`，`test_step` 方法中调用该模式的 `forward`。
- `mode='tensor'`：`tensor` 和 `predict` 均用于返回模型的预测结果，区别在于 OpenMMLab 系列的算法库要求 `predict` 模式返回数据元素列表，而 `tensor` 模式则返回 `torch.Tensor` 类型的结果。`tensor` 模式为 `forward` 的默认模式，如果我们想获取一张或一个批次（batch）图片的推理结果，可以直接调用 `model(inputs)` 来获取预测结果。

[train_step](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.model.BaseModel.train_step): 调用 `loss` 模式的 `forward` 接口，得到损失字典。模型基类基于[优化器封装](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html) 实现了标准的梯度计算、参数更新、梯度清零流程。如果模型需要自定义的参数更新逻辑，可以重载 `train_step` 接口，具体例子见：[使用 MMEngine 训练生成对抗网络](TODO)

[val_step](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.model.BaseModel.val_step): 调用 `predict` 模式的 `forward`，返回预测结果，预测结果会被进一步传给[钩子（Hook）](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/hook.html)的 `after_train_iter` 和 `after_val_iter` 接口。

[test_step](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.model.BaseModel.test_step): 同 `val_step`，预测结果会被进一步传给 `after_test_iter` 接口。

基于上述接口约定，我们定义了继承自模型基类的 `NeuralNetwork`，配合执行器来训练 `FashionMNIST`：

```python
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from mmengine.model import BaseModel
from mmengine.evaluator import BaseMetric
from mmengine import Runner


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(dataset=training_data, batch_size=64)
test_dataloader = DataLoader(dataset=test_data, batch_size=64)


class NeuralNetwork(BaseModel):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, img, label, mode='tensor'):
        x = self.flatten(img)
        pred = self.linear_relu_stack(x)
        loss = self.loss(pred, label)
        if mode == 'loss':
            return dict(loss=loss)
        else:
            return pred.argmax(1), loss.item()


class FashionMnistMetric(BaseMetric):
    def process(self, data, preds) -> None:
        self.results.append(((data[1] == preds[0].cpu()).sum(), preds[1], len(preds[0])))

    def compute_metrics(self, results):
        correct, loss, batch_size = zip(*results)
        test_loss, correct = sum(loss) / len(self.results), sum(correct) / sum(batch_size)
        return dict(Accuracy=correct, Avg_loss=test_loss)


runner = Runner(
    model=NeuralNetwork(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=1e-3)),
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_cfg=dict(fp16=True),
    val_dataloader=test_dataloader,
    val_evaluator=dict(metrics=FashionMnistMetric()))
runner.train()
```

相比于 [Pytorch 官方示例](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#)，MMEngine 的代码更短，记录的日志也更加丰富。

在这个例子中，`NeuralNetwork.forward` 存在着以下跨模块的接口约定：

- 由于 `train_dataloader` 会返回一个 `(img, label)` 型式的元组，因此 `forward` 接口的前两个参数分别需要为 `img` 和 `label`。
- 由于 `forward` 在 `predict` 模式下会返回 `(pred, loss)` 型式的元组，因此 `process` 的 preds 参数应当同样为相同型式的元组。

### 数据处理器（DataPreprocessor）

如果你的电脑配有 GPU（或其他能够加速训练的硬件，如 mps、ipu 等），并运行了上节的代码示例。你会发现 Pytorch 的示例是在 CPU 上运行的，而 MMEngine 的示例是在 GPU 上运行的。`MMEngine` 是在何时把数据和模型从 CPU 搬运到 GPU 的呢？

事实上，执行器会在构造阶段将模型搬运到指定设备，而数据则会在 `train_step`、`val_step`、`test_step` 中，被[基础数据处理器（BaseDataPreprocessor）](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.model.BaseDataPreprocessor)搬运到指定设备，进一步将处理好的数据传给模型。数据处理器作为模型基类的一个属性，会在模型基类的构造过程中被实例化。

为了体现数据处理器起到的作用，我们仍然以[上一节](#模型基类basemodel)训练 FashionMNIST 为例, 实现了一个简易的数据处理器，用于搬运数据和归一化：

```python
from mmengine.model import BaseDataPreprocessor, BaseModel


class NeuralNetwork(BaseModel):
    def __init__(self):
        super(NeuralNetwork, self).__init__(
            data_preprocessor=NormalizeDataPreprocessor())
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, img, label, mode='tensor'):
        x = self.flatten(img)
        pred = self.linear_relu_stack(x)
        loss = self.loss(pred, label)
        if mode == 'loss':
            return dict(loss=loss)
        else:
            return pred.argmax(1), loss.item()

class NormalizeDataPreprocessor(BaseDataPreprocessor):
    def forward(self, data, training=False):
        img, label = [item.cuda() for item in data]
        img = (img - 127.5) / 127.5
        return img, label
```

此时 `NeuralNetwork.forward` 接受的 `img` 和 `label` 分别对应 `NormalizeDataPreprocessor.forward` 的返回值。

```{note}
上例中数据处理器的 training 参数用于区分训练、测试阶段不同的批增强策略，`train_step` 会传入 `training=True`，`test_step` 和 `val_step` 则会传入 `trainig=Fasle`。
```

```{note}
通常情况下，我们要求 DataLoader 的 `data` 数据解包后（字典类型的被 **data 解包，元组列表类型被 *data 解包）能够直接传给模型的 `forward`。但是如果数据处理器修改了 data 的数据类型，则要求数据处理器的 `forward` 的返回值与模型 `forward` 的入参相匹配。
```

[kaiming]: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
[xavier]: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
