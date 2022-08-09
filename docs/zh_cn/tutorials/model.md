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

基于标准化的模型初始化流程，MMEngine 抽象出
`BaseModule`，让我们能够更加灵活的选择模型的初始化方式，同样的，基于标准化的模型训练流程，MMEngine
抽象出基础模型（`BaseModel`），让模型的核心代码更加集中，方便阅读和理解。尽管 Pytorch 对 `nn.Module`
的接口没有任何要求，但是我们认为一套标准的模型接口能够让我们更好的理解代码，因此 MMEngine 要求，只有符合基础模型接口约定的模型才能在[执行器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html)中训练起来。

### 定义基础模型

基础模型实现了 `train_step`、`val_step` 和 `test_step` 接口，并要求子类必须实现 `forward`
方法。要想理解为什么基础模型有这些接口约定，我们不妨先来看看 [Pytorch 官方提供的训练 FashionMNIST 的流程](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。

#### Pytorch 标准训练流程

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# 准备数据集
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

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 定义模型
class NeuralNetwork(nn.Module):
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

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

learning_rate = 1e-3
batch_size = 64
epochs = 5

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 定义训练流程
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# 定义测试流程
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

我们不难发现，Pytorch 官方 demo 中，模型（`NeuralNetwork`）只负责前向推理，而损失计算、参数更新、结果预测的逻辑分别在
`train_loop` 和 `test_loop`
里实现。这样的结构非常适合训练优化逻辑比较简单的任务，即前向推理一次，计算一次梯度，更新一次参数。

#### MMEngine 训练流程

MMEngine 作为通用的深度学习训练框架，需要应对不同任务、不同需求的参数更新逻辑，如果套用上例的训练流程，就会出现核心代码分散在不同模块的情况，这是不合理的。以训练生成对抗网络为例，我们需要分别优化生成器和判别器，我们就需要在 `train_loop` 和 `test_step` 中实现和算法相关的训练、预测逻辑。这样设计存在明显的缺点：

1. 模型的参数更新逻辑属于算法的一部分，不应该分散在各个模块内，既不利于阅读，也不利于维护。
2. `test_loop` 和 `train_loop` 这一层的抽象容易和算法绑定，一个新的训练流程的算法会需要同时派生出新的 `loop` 和新的 `model`

因此 MMEngine 重新划分了模型功能的边界，模型不仅需要负责前向推理，还需要负责参数更新、和结果预测，因此为基础模型抽象出
`train_step`、`val_step` 和 `test_step` 接口。此外基础模型的 `forward`
接口也需要承担更多功能，在训练阶段返回损失，验证、测试阶段返回预测结果。

- [forward](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.model.BaseModel.forward): `forward` 的入参需要和 [DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) 的输出保持一致，如果 `DataLoader` 返回 元组类型的数据 `data`，`forward` 需要能够接受 `*data` 的入参；如果返回字典类型的数据 `data`，`forward` 需要能够接受 `**data` 的入参。 `mode` 参数用于控制 forward 的返回结果：

  - `mode='loss'`：`forawrd` **必须**返回一个字典， key-value 分别为损失名和可微的 `torch.Tensor`。字典中记录的损失会被用于更新参数，多次迭代统计后输出到终端。该模式会被 `train_step` 调用。
  - `mode='predict'`： `forward` 必须返回列表/元组型式的预测结果，预测结果需要和\[评测指标\]的(https://mmengine.readthedocs.io/zh_CN/latest/tutorials/metric_and_evaluator.html) `process` 接口的第一个参数相符合。该模式会被 `val_step`, `test_step` 接口调用。OpenMMLab 系列算法则有更加严格的约定，需要输出列表型式的[数据元素](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/data_element.html)
  - `mode='tensor'`：`tensor` 和 `predict` 均用于返回模型的预测结果，区别在于 OpenMMLab 系列的算法库要求 `predict` 返回数据元素列表，而 `tensor` 则返回 `torch.Tensor` 类型的结果。

- [train_step](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.model.BaseModel.train_step): 调用 `forward` 接口，得到损失字典，进行参数更新并返回整理后的损失字典。基础模型基于[优化器封装](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html) 实现了标准的梯度计算、参数更新、梯度清零流程。如果模型需要自定义的参数更新逻辑，可以重载 `train_step` 接口，具体例子见：[使用 MMEngine 训练生成对抗网络](TODO)

- [val_step](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.model.BaseModel.val_step): 返回预测结果或损失（待开发），预测结果会被进一步传给[钩子（Hook）](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/hook.html)的 `after_train_iter`、`after_val_iter` 和 `after_test_iter` 接口。

- [test_step](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.model.BaseModel.test_step): 同 `val_step`，但是只返回预测结果。

基于上述接口约定，我们重新实现 `NeuralNetwork`，基于 MMEngine 来实现训练流程：

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
        self.results.append(((data[1] == preds[0].cpu()).sum() / len(preds[0]), preds[1]))

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

相比于 Pytorch 的官方示例，基于 MMEngine 的样例代码更加简洁、日志更加清晰、且支持了混合精度训练。
这里再次强调 MMEngine 实现的 `NeuralNetwork.forward` 存在跨模块的接口约定：

- `train_dataloader` 返回一个 `(img, label)` 型式的元组，因此 `forward` 接口的前两个参数分别为 `img` 和 `label`。
- `forward` 在 `predict` 模式下返回 `(pred, loss)` 型式的元组，因此 `process` 的 preds 参数同样为相同型式的元组。

### 数据处理器（DataPreprocessor）

如果你的电脑配有 Nvidia 的 GPU，并且运行了上节的代码样例，不难发现 Pytorch 的样例是基于 CPU 运行的，而 MMEngine 的样例是基于 GPU 运行的。细心的你可能会奇怪，数据和模型从 CPU 搬运到 GPU 的过程在何时发生？

执行器会在构造阶段将模型搬运到指定设备，而数据则会在 `train_step`、`val_step`、`test_step` 中，被[基础数据处理器（BaseDataPreprocessor）](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.model.BaseDataPreprocessor)搬运到指定设备，进一步将处理好的数据传给模型。

![data_preprocessor](https://user-images.githubusercontent.com/57566630/183727857-b77f1568-60e2-435f-bd97-700fd93733db.png)

MMEngine 还实现了 [图像数据处理器 （`ImgDataPreprocessor`）](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.model.BaseDataPreprocessor)除了数据搬运，还可以完成 BGR 和 RGB 格式互转，归一化等操作（图像数据处理器要求 DataLoader 返回一个字典）。

MMEngine 抽象出数据处理器的概念，并在其中实现数据搬运、归一化等操作主要有以下两个原因：

- 有些深度学习任务需要实现基于 batch 的数据增强策略，例如 [Mixup](https://arxiv.org/pdf/1710.09412.pdf)，这部分代码不应该和具体算法相绑定。抽象出图像处理器这个概念可以很好的解决这个问题，模型和图像处理器可以灵活地搭配使用。
- 如果图像在进入模型之前就做了归一化，就意味着需要将 `float` 型的数据从 CPU 搬运到 GPU。如果在进入模型之后再做归一化，我们可以先将 `uchar` 的数据搬运到到 GPU，再做归一化，IO 负载可以减少 4 倍。

我们来看一个数据处理器的简单示例，假设 DataLoader 返回 `（img，label）`型式的元组，`img` 和 `label` 类型均为 `torch.Tensor`，且 img 的形状为 `(n,c,h,w)`。我们希望数据处理器能够对 `img` 做均值标准差均为 127.5 的归一化，模型的 forward 接口直接接受归一化后的结果：

```python
from mmengine.model import BaseDataPreprocessor, BaseModel


class NormalizeDataPreprocessor(BaseDataPreprocessor):
    def forward(self, data, training=False):
        img, label = [item.cuda() for item in data]
        img = (img - 127.5) / 127.5
        return img, label

class CustomModel(BaseModel):
    def forward(self, img, label, mode):
        ...
```

此时 `CustomModel.forward` 接受的 `img` 和 `label` 分别对应 `NormalizeDataPreprocessor` 的返回值。

```{node}
上例中数据处理器的 training 参数用于区分训练、测试阶段不同的批增强策略，`train_step` 会传入 `training=True`，`test_step` 和 `val_step` 则会传入 `trainig=Fasle`。
```

[kaiming]: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
[xavier]: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
