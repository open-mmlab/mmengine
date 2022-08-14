# 模型

MMEngine 在 `nn.Module` 的基础上进一步的抽象出了基础模块（`BaseModule`） 和基础模型
（`BaseModel`），前者用于配置模型初始化策略，后者定义了模型训练、验证、测试、推理的基本流程。

## 基础模块（BaseModule）

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
| `UniformInit`      |   Uniform    | 将 weight 和 bias 以均匀分布的型式初始化，参数 a 和 b 为均匀分布的范围                  |
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

## 基础模型（BaseModel）

[执行器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html)要求模型需要实现 `train_step`，`val_step` 和 `test_step` 方法。因此 MMEngine 定义了基础模型 `BaseModel`，并在上述接口中实现了基本的训练流程，我们可以通过继承基础模型，实现符合执行器接口标准的模型。

### 接口定义

[forward](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.model.BaseModel.forward): `forward` 的入参需要和 [DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) 的输出保持一致 (自定义[数据处理器](#数据处理器datapreprocessor)除外)，如果 `DataLoader` 返回 元组类型的数据 `data`，`forward` 需要能够接受 `*data` 的入参；如果返回字典类型的数据 `data`，`forward` 需要能够接受 `**data` 的入参。 `mode` 参数用于控制 forward 的返回结果：

- `mode='loss'`：`forawrd` **必须**返回一个字典， key-value 分别为损失名和可微的 `torch.Tensor`。字典中记录的损失会被用于更新参数，多次迭代统计后输出到终端。该模式会被 `train_step` 调用。
- `mode='predict'`： `forward` 必须返回列表/元组型式的预测结果，预测结果需要和\[评测指标\]的(https://mmengine.readthedocs.io/zh_CN/latest/tutorials/metric_and_evaluator.html) `process` 接口的第一个参数相符合。该模式会被 `val_step`, `test_step` 接口调用。OpenMMLab 系列算法则有更加严格的约定，需要输出列表型式的[数据元素](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/data_element.html)
- `mode='tensor'`：`tensor` 和 `predict` 均用于返回模型的预测结果，区别在于 OpenMMLab 系列的算法库要求 `predict` 返回数据元素列表，而 `tensor` 则返回 `torch.Tensor` 类型的结果。

[train_step](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.model.BaseModel.train_step): 调用 `forward` 接口，得到损失字典，进行参数更新并返回整理后的损失字典。基础模型基于[优化器封装](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html) 实现了标准的梯度计算、参数更新、梯度清零流程。如果模型需要自定义的参数更新逻辑，可以重载 `train_step` 接口，具体例子见：[使用 MMEngine 训练生成对抗网络](TODO)

[val_step](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.model.BaseModel.val_step): 返回预测结果或损失，预测结果会被进一步传给[钩子（Hook）](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/hook.html)的 `after_train_iter`、`after_val_iter` 和 `after_test_iter` 接口。

[test_step](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.model.BaseModel.test_step): 同 `val_step`，但是只返回预测结果。

基于上述接口约定，我们定义了继承自基础模型的 `NeuralNetwork`，配合执行器来训练 FashionMNIST

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
        self.results.append(((data[1] == preds[0].cpu()).sum() \
             / len(preds[0]), preds[1]))

    def compute_metrics(self, results):
        correct, loss = zip(*results)
        test_loss, correct = sum(loss) / len(self.results), sum(correct) / len(self.results)
        return dict(Accuracy=correct, Avg_loss=test_loss)


runner = Runner(
    model=NeuralNetwork(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=1e-3)),
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_cfg=dict(fp16=True),
    val_dataloader=test_dataloader,
    val_evaluator=dict(type=FashionMnistMetric))
runner.train()
```

`NeuralNetwork.forward` 存在跨模块的接口约定：

- `train_dataloader` 返回一个 `(img, label)` 型式的元组，因此 `forward` 接口的前两个参数分别为 `img` 和 `label`。
- `forward` 在 `predict` 模式下返回 `(pred, loss)` 型式的元组，因此 `process` 的 preds 参数同样为相同型式的元组。

### 数据处理器（DataPreprocessor）

如果你的电脑配有 Nvidia 的 GPU，并且运行了上节的代码样例，不难发现 Pytorch 的样例是基于 CPU 运行的，而 MMEngine 的样例是基于 GPU 运行的。细心的你可能会奇怪，数据和模型从 CPU 搬运到 GPU 的过程在何时发生？

执行器会在构造阶段将模型搬运到指定设备，而数据则会在 `train_step`、`val_step`、`test_step` 中，被[基础数据处理器（BaseDataPreprocessor）](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.model.BaseDataPreprocessor)搬运到指定设备，进一步将处理好的数据传给模型。数据处理器是基础模型的属性，在基础模型的构造过程中被实例化。

为了体现数据处理器起到的作用，我们仍然以[上一节](#基础模型basemodel)训练 FashionMNIST 为例, 实现了一个简易的数据处理器，用于搬运数据和归一化：

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

```{node}
上例中数据处理器的 training 参数用于区分训练、测试阶段不同的批增强策略，`train_step` 会传入 `training=True`，`test_step` 和 `val_step` 则会传入 `trainig=Fasle`。
```

```{node}
通常情况下，我们要求 DataLoader 的 `data` 数据解包后（字典类型的被 **data 解包，元组列表类型被 *data 解包）能够直接传给模型的 `forward`。但是如果数据处理器修改了 data 的数据类型，则要求数据处理器的 `forward` 的返回值与模型 `forward` 的入参相匹配。
```

[kaiming]: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
[xavier]: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
