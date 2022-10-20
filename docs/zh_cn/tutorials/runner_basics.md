# 执行器（Runner）基础

- [执行器（Runner）基础](#执行器runner基础)
  - [执行器示例与最佳实践](#执行器示例与最佳实践)
    - [面向初学者的示例代码（入门参考）](#面向初学者的示例代码入门参考)
    - [示例代码讲解](#示例代码讲解)
    - [使用配置文件（最佳实践）](#使用配置文件最佳实践)
  - [基本数据流](#基本数据流)
  - [为什么使用执行器（可选）](#为什么使用执行器可选)
  - [下一步的建议](#下一步的建议)

欢迎来到 MMEngine 用户界面的核心——执行器！

作为 MMEngine 中的“集大成者”，执行器涵盖了整个框架的方方面面，肩负着串联所有组件的重要责任；因此，其中的代码和实现逻辑需要兼顾各种情景，相对庞大复杂。但是**不用担心**！在这篇教程中，我们将隐去繁杂的细节，速览执行器常用的接口、功能、示例，为你呈现一个清晰易懂的用户界面。阅读完本篇教程，你将会：

- 掌握执行器的常见使用方式与最佳实践
- 了解执行器基本数据流
- 亲身感受使用执行器的优越性（也许）

## 执行器示例与最佳实践

使用执行器构建属于你自己的训练流程，通常有两种开始方式：

- 参考 [API 文档](mmengine.runner.Runner)，逐项确认和配置参数
- 在已有配置（如 [15 分钟上手](../get_started/15_minutes.md)或 [mmdet](https://github.com/open-mmlab/mmdetection) 等下游算法库）的基础上，进行定制化修改

两种方式各有利弊。使用前者，初学者很容易迷失在茫茫多的参数项中不知所措；而使用后者，一份过度精简或过度详细的参考配置都不利于初学者快速找到所需内容。

解决上述问题的关键在于，把执行器作为备忘录：掌握其中最常用的部分，并在有特殊需求时聚焦感兴趣的部分。下面我们将通过一个适合初学者参考的例子，说明其中最常用的参数，并为一些不常用参数给出进阶指引。

### 面向初学者的示例代码（入门参考）

```{hint}
这个例子省略了大量实现细节，并且与 15 分钟教程的示例有所不同（我们将在下面解释原因）。我们希望你在本教程中更多地关注整体结构，而非具体模块的实现。这种“自顶向下”的思考方式是我们所倡导的。别担心，之后你将有充足的机会和指引，聚焦于自己想要改进的模块
```

<details>
<summary>为了让下面的示例可以运行而添加的实现细节。在本教程中并不重要，无需关注</summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from mmengine.model import BaseModel
from mmengine.evaluator import BaseMetric
from mmengine.registry import MODELS, DATASETS, METRICS


@MODELS.register_module()
class MyAwesomeModel(BaseModel):
    def __init__(self, layers=4, activation='relu') -> None:
        super().__init__()
        if activation == 'relu':
            act_type = nn.ReLU
        elif activation == 'silu':
            act_type = nn.SiLU
        elif activation == 'none':
            act_type = nn.Identity
        else:
            raise NotImplementedError
        sequence = [nn.Linear(2, 64), act_type()]
        for _ in range(layers-1):
            sequence.extend([nn.Linear(64, 64), act_type()])
        self.mlp = nn.Sequential(*sequence)
        self.classifier = nn.Linear(64, 2)

    def forward(self, data, labels, mode):
        x = self.mlp(data)
        x = self.classifier(x)
        if mode == 'tensor':
            return x
        elif mode == 'predict':
            return F.softmax(x, dim=1), labels
        elif mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}


@DATASETS.register_module()
class MyDataset(Dataset):
    def __init__(self, is_train):
        self.is_train = is_train
        if self.is_train:
            torch.manual_seed(0)
            self.labels = torch.randint(0, 2, (10000,))
        else:
            torch.manual_seed(3407)
            self.labels = torch.randint(0, 2, (1000,))
        r = 3 * (self.labels+1) + torch.randn(self.labels.shape)
        theta = torch.rand(self.labels.shape) * 2 * torch.pi
        self.data = torch.vstack([r*torch.cos(theta), r*torch.sin(theta)]).T

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


@METRICS.register_module()
class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__()

    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(r['correct'] for r in results)
        total_size = sum(r['batch_size'] for r in results)
        return dict(accuracy=100*total_correct/total_size)
```

</details>

<details>
<summary>点击展开一段长长的示例代码。做好准备</summary>

<table>
<thead>
<tr>
<th> 示例1 </th>
<th> 示例2 </th>
</tr>
<tbody>
<tr>

<td valign="top">

```python
from torch.utils.data import (
    DataLoader, default_collate)
from torch.optim import Adam
from mmengine.runner import Runner


runner = Runner(
    # 你的模型
    model=MyAwesomeModel(
        layers=2,
        activation='relu'),
    # 模型检查点、日志等都将存储在工作路径中
    work_dir='exp/my_awesome_model',

    # 训练相关配置
    train_dataloader=DataLoader(
        dataset=MyDataset(
            is_train=True,
            size=10000),
        shuffle=True,
        collate_fn=default_collate,
        batch_size=64,
        pin_memory=True,
        num_workers=2),
    train_cfg=dict(
        by_epoch=True,
        max_epochs=10,
        val_interval=1),
    # 优化器封装，MMEngine 中的新概念，提供
    # 更丰富的优化选择。通常使用默认即可，可
    # 缺省。有特殊需求可查阅文档更换，如
    # 'AmpOptimWrapper' 开启混合精度训练
    optim_wrapper=dict(
        optimizer=dict(
            type=Adam,
            lr=0.001)),
    param_scheduler=dict(
        type='MultiStepLR',
        milestones=[1, 2]),

    # 验证所需配置
    val_dataloader=DataLoader(
        dataset=MyDataset(
            is_train=False,
            size=1000),
        shuffle=False,
        collate_fn=default_collate,
        batch_size=1000,
        pin_memory=True,
        num_workers=2),
    val_cfg=dict(),
    # 验证指标与验证器封装，MMEngine 中的
    # 新概念，可自由实现与配置，也可缺省
    val_evaluator=dict(type=Accuracy),

    # 其他进阶配置，无特殊需要时尽量缺省
    # 钩子属于进阶用法，如无特殊需要，尽量缺省
    default_hooks=dict(
        checkpoint=dict(
            type='CheckpointHook',
            interval=1)),
    # `luancher` 与 `env_cfg` 共同构成
    # 分布式训练环境配置
    launcher='none',
    env_cfg=dict(backend='nccl'),
    log_level='INFO'
)
runner.train()
```

</td>
<td valign="top">

```python
from mmengine.runner import Runner

runner = Runner(
    # 你的模型
    model=dict(type='MyAwesomeModel',
        layers=2,
        activation='relu'),
    # 模型检查点、日志等都将存储在工作路径中
    work_dir='exp/my_awesome_model',

    # 训练相关配置
    train_dataloader=dict(
        dataset=dict(type='MyDataset',
            is_train=True,
            size=10000),
        sampler=dict(
            type='DefaultSampler',
            shuffle=True),
        collate_fn=dict(type='default_collate'),
        batch_size=64,
        pin_memory=True,
        num_workers=2),
    train_cfg=dict(
        by_epoch=True,
        max_epochs=10,
        val_interval=1),
    # 优化器封装，MMEngine 中的新概念，提供
    # 更丰富的优化选择。通常使用默认即可，可
    # 缺省。有特殊需求可查阅文档更换，如
    # 'AmpOptimWrapper' 开启混合精度训练
    optim_wrapper=dict(
        optimizer=dict(
            type='Adam',
            lr=0.001)),
    param_scheduler=dict(
        type='MultiStepLR',
        milestones=[1, 2]),

    # 验证所需配置
    val_dataloader=dict(
        dataset=dict(type='MyDataset',
            is_train=False,
            size=1000),
        sampler=dict(
            type='DefaultSampler',
            shuffle=False),
        collate_fn=dict(type='default_collate'),
        batch_size=1000,
        pin_memory=True,
        num_workers=2),
    val_cfg=dict(),
    # 验证指标与验证器封装，MMEngine 中的
    # 新概念，可自由实现与配置，也可缺省
    val_evaluator=dict(type='Accuracy'),

    # 其他进阶配置，无特殊需要时尽量缺省
    # 钩子属于进阶用法，如无特殊需要，尽量缺省
    default_hooks=dict(
        checkpoint=dict(
            type='CheckpointHook',
            interval=1)),
    # `luancher` 与 `env_cfg` 共同构成
    # 分布式训练环境配置
    launcher='none',
    env_cfg=dict(backend='nccl'),
    log_level='INFO'
)
runner.train()
```

</td>
</tr>
</table>

</details>

### 示例代码讲解

真是两段长长的代码！但是如果你通读了上述样例，即使不了解实现细节，你也一定大体理解了这个训练流程，并感叹于执行器代码的紧凑与可读性（也许）。这也是 MMEngine 所期望的：结构化、模块化、标准化的训练流程，使得复现更加可靠、对比更加清晰。

细心的你可能会发现，两段示例代码是等效的，但风格有些微不同：左侧示例更接近于 [15分钟上手](../tutorials/../get_started/15_minutes.md)，而右侧显得较为陌生。我们将在下面讲解两种风格。但需要明确的是：两段代码仍非执行器的最佳实践，而是一个中间过渡。

上述例子可能会让你产生如下问题：

<details>
<summary>参数项实在是太多了！</summary>

不用担心，正如我们前面所说，**把执行器作为备忘录**。执行器涵盖了方方面面，防止你漏掉重要内容；但你不需要配置所有参数。如[15分钟上手](../tutorials/../get_started/15_minutes.md)中的极简例子（甚至，舍去 `val_evaluator`）也可以正常运行。所有的参数由你的需求驱动，不关注的内容往往缺省值也可以工作得很好。

</details>

<details>
<summary>右侧示例似乎和 15 分钟上手的写法截然不同？为什么传入参数是`dict`？</summary>

是的，这与 MMEngine 的风格相关。在 MMEngine 中我们提供了两种不同风格的执行器构建方式：a）基于手动构建的（左），以及 b）基于配置与注册机制的（右）。如果你感到迷惑，下面的例子将给出一个对比：

```python
from mmengine.model import BaseModel
from mmengine.runner import Runner
from mmengine.registry import MODELS # 模型根注册器，你可以暂时忽略
@MODELS.register_module() # 用于注册的装饰器，你可以暂时忽略
class MyAwesomeModel(BaseModel): # 你的自定义模型
    def __init__(self, layers=18, activation='silu'):
        ...

# 基于配置与注册机制的例子
runner = Runner(
    model=dict(
        type='MyAwesomeModel',
        layers=50,
        activation='relu'),
    ...
)

# 基于手动构建的例子
model = MyAwesomeModel(layers=18, activation='relu')
runner = Runner(
    model=model,
    ...
)
```

类似上述例子，执行器中的参数大多同时支持两种输入类型。以 `dict` 作为输入时，该模块会在执行器内部被构建。如果你对于注册机制并不了解，你可以暂且简单理解为，执行器会根据 `type` 寻找被特定装饰器修饰的类，调用其 `__init__` 方法，并将 `dict` 中其余参数传入（并不确切，但足以理解、处理大部分情况）。如果你在使用中发现问题，或者想要进一步了解完整用法，我们推荐阅读[注册器（Registry）](./registry.md)文档。

这一段内容涉及到 MMEngine 内部实现、注册器设计模式以及 `Python` 参数传递机制，因而对于初学者来说相对隐晦、不易理解。但理解这部分内容仍然十分重要：它为理解执行器真正的最佳实践带来曙光。

如果你作为初学者无法立刻理解，使用*手动构建的方式*依然不失为一种好选择，甚至在小规模使用、试错和调试时是一种更加推荐的方式，因为对于 IDE 更加友好。我们也常挣扎于是否要在教程中展示这些内容，但最终我们确信，配置文件——一种基于配置与注册机制的构建方式——是使用 MMEngine 的最佳实践，并且该方式已经在 OpenMMLab 的下游库中广泛使用、成为事实标准。我们将在接下来的章节中略微修改示例，从而展示该部分。

</details>

<details>
<summary>我来自 mmdet/mmcls...下游库，为什么例子写法与我接触的不同？</summary>

OpenMMLab 下游库广泛采用了配置文件的方式。我们将在下个章节，基于上述示例稍微变换，从而展示配置文件——MMEngine 中执行器的最佳实践——的用法。

</details>

### 使用配置文件（最佳实践）

上述例子中我们大量使用 `dict` 作为参数。如果你对 `Python` 的参数传递有所了解，你可能会意识到示例代码等价于如下代码段

```python
my_cfg = dict(
    model=dict(
        type='MyAwesomeModel',
        layers=2,
        activation='relu'),
    work_dir='exp/my_awesome_model',
    ... # 以下参数完全拷贝自原示例，此处省略
)
runner = Runner(**my_cfg)
```

这意味着 `my_cfg` 可以被导出到各种格式（如 `json` `yaml` 等）的文件中保存，并在需要时加载。真酷！如果你有过使用 `yaml` 或 `yacs` 等工具进行实验配置的经历，你一定会理解配置文件对于数据、模型、实验管理的重要意义。

MMEngine 提供了一套 `Python` 语法的、功能强大的配置文件工具。你可以从之前的示例代码中**近乎**（我们将在下面说明）无缝地转换到配置文件

<details>
<summary>点击展开全部代码</summary>

```python
# 以下代码存放在 example_config.py 文件中
# 基本拷贝自上面的示例，并将每项结尾的逗号删去
model=dict(type='MyAwesomeModel',
    layers=2,
    activation='relu')
work_dir='exp/my_awesome_model'

train_dataloader=dict(
    dataset=dict(type='MyDataset',
        is_train=True,
        size=10000),
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    collate_fn=dict(type='default_collate'),
    batch_size=64,
    pin_memory=True,
    num_workers=2)
train_cfg=dict(
    by_epoch=True,
    max_epochs=10,
    val_interval=1)
optim_wrapper=dict(
    optimizer=dict(
        type='Adam',
        lr=0.001))
param_scheduler=dict(
    type='MultiStepLR',
    milestones=[1, 2])

val_dataloader=dict(
    dataset=dict(type='MyDataset',
        is_train=False,
        size=1000),
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),
    collate_fn=dict(type='default_collate'),
    batch_size=1000,
    pin_memory=True,
    num_workers=2)
val_cfg=dict()
val_evaluator=dict(type='Accuracy')

default_hooks=dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1))
launcher='none'
env_cfg=dict(backend='nccl')
log_level='INFO'
```

</details>

此时，我们只需要在训练代码中加载配置，然后运行即可

```python
from mmengin.config import Config
from mmengine.runner import Runner
config = Config.fromfile('example_config.py')
runner = Runner.from_cfg(config)
runner.train()
```

```{note}
当使用配置文件写法时，你将不得不更多地了解[注册器](./registry.md)，因为此时你的自定义模块的实现代码通常存放在独立文件中，可能并未被正确注册，进而导致构建失败。
```

```{warning}
虽然与示例中的写法一致，但 `from_cfg` 与 `__init__` 的缺省值处理可能存在些微不同，例如`env_cfg`参数。为了可复现性，我们推荐执行器配置文件包含执行器的所有参数，以确保训练的每个部分清晰可见，避免缺省值带来的潜在歧义。因此，我们建议多付出一点点努力，在配置文件中将缺省值补全。
```

执行器配置文件已经在 OpenMMLab 的众多下游库（mmcls，mmdet...）中被广泛使用，并成为事实标准与最佳实践。配置文件的功能远不止如此，如果你对于继承、覆写等进阶功能感兴趣，请参考[配置（Config）](./config.md)文档。

## 基本数据流

```{hint}
在本章节中，我们将会介绍执行器内部各模块之间的数据传递流向与格式约定。如果你还没有基于 MMEngine 构建一个训练流程，本章节的部分内容可能会比较抽象、枯燥；你也可以暂时跳过，并在将来有需要时结合实践进行阅读。
```

接下来，我们将**稍微**深入执行器的内部，结合图示来理清其中数据的流向与格式约定。

![基本数据流](https://user-images.githubusercontent.com/112053249/191187150-58ac9e7e-7cf6-4b36-a0f8-39067f95e229.jpg)

上图是执行器的**基本**数据流，其中虚线边框、灰色填充的不同形状代表不同的数据格式，实线方框代表模块或方法。由于 MMEngine 强大的灵活性与可扩展性，你总可以继承某些关键基类并重载其中的方法，因此上图并不总是成立。只有当你的自定义模型没有重载 `train_step`、`val_step` 与 `test_step` 方法时上图才会成立（而这在检测、分割等任务上是常见的，参考[模型](./model.md)教程）。

<details>
<summary>可以确切地说明每个数据元素的具体类型吗？</summary>

很遗憾，这一点无法做到。虽然 MMEngine 做了大量类型注释，但 `Python` 是一门高度动态化的编程语言，同时以数据为核心的深度学习系统也需要足够的灵活性来处理纷繁复杂的数据源，你有充分的自由决定何时需要（有时是必须）打破类型约定。因此，在你自定义某一或某几个模块（如 `val_evaluator` ）时，你需要确保它的输入与上游（如 `model` 的输出）兼容，同时输出可以被下游解析。MMEngine 将处理数据的灵活性交给了用户，因而也需要用户保证数据流的兼容性——当然，实际上手后会发现，这一点并不十分困难。

数据一致性的考验一直存在于深度学习领域，MMEngine 也在尝试用自己的方式改进。如果你有兴趣，可以参考[数据集基类](../advanced_tutorials/basedataset.md)与[抽象数据接口](../advanced_tutorials/data_element.md)文档——但是请注意，它们主要面向进阶用户。

</details>

<details>
<summary>什么是 data preprocessor？我可以用它做裁减缩放等图像预处理吗？</summary>

虽然图中的 data preprocessor 与 model 是分离的，但在实际中前者是后者的一部分，因此可以在[模型](./model.md)文档中的数据处理器章节找到。

通常来说，数据处理器不需要额外关注和指定，默认的数据处理器只会自动将数据搬运到 GPU 中。但是，如果你的模型与数据加载器的数据格式不匹配，你也可以自定义一个数据处理器来进行格式转换。

裁减缩放等图像预处理更推荐在[数据变换](./data_transform.md)中进行，但如果是 batch 相关的数据处理（如 batch-resize 等），可以在这里实现。

</details>

<details>
<summary>为什么 model 产生了 3 个不同的输出？ loss、predict、tensor 是什么含义？</summary>

[15 分钟上手](../get_started/15_minutes.md)对此有一定的描述，你需要在自定义模型的 `forward` 函数中实现 3 条数据通路，适配训练、验证等不同需求。[模型](./model.md)文档中对此有详细解释。

</details>

<details>
<summary>我可以看出红线是训练流程，蓝线是验证/测试流程，但绿线是什么？</summary>

在目前的执行器流程中，`'tensor'` 模式的输出并未被使用，大多数情况下用户无需实现。但一些情况下输出中间结果可以方便地进行 debug

</details>

<details>
<summary>如果我重载了`train_step`等方法，上图会完全失效吗？</summary>

默认的 `train_step`、`val_step`、`test_step` 的行为，覆盖了从数据进入 `data preprocessor` 到 `model` 输出 `loss`、`predict` 结果的这一段流程，不影响其余部分。

</details>

## 为什么使用执行器（可选）

```{hint}
这一部分内容并不能教会你如何使用执行器乃至整个 MMEngine，如果你正在被雇主/教授/DDL催促着几个小时内拿出成果，那这部分可能无法帮助到你，请随意跳过。但我们仍强烈推荐抽出时间阅读本章节，这可以帮助你更好地理解并使用 MMEngine
```

<details>
<summary>放轻松，接下来是闲聊时间</summary>

恭喜你通关了执行器！这真是一篇长长的、但还算有趣（希望如此）的教程。无论如何，请相信这些都是为了让你更加**轻松**——不论是本篇教程、执行器，还是 MMEngine。

执行器是 MMEngine 中所有模块的“管理者”。所有的独立模块——不论是模型、数据集这些看得见摸的着的，还是日志记录、分布式训练、随机种子等相对隐晦的——都在执行器中被统一调度、产生关联。事物之间的关系是复杂的，但执行器为你处理了一切，并提供了一个清晰易懂的配置式接口。这样做的好处主要有：

1. 你可以轻易地在已搭建流程上修改/添加所需配置，而不会搅乱整个代码。也许你起初只有单卡训练，但你随时可以添加1、2行的分布式配置，切换到多卡甚至多机训练
2. 你可以享受 MMEngine 不断引入的新特性，而不必担心后向兼容性。混合精度训练、可视化、崭新的分布式训练方式、多种设备后端……我们会在保证后向兼容性的前提下不断吸收社区的优秀建议与前沿技术，并以简洁明了的方式提供给你
3. 你可以集中关注并实现自己的惊人想法，而不必受限于其他恼人的、不相关的细节。执行器的缺省值会为你处理绝大多数的情况

所以，MMEngine 与执行器会确实地让你更加轻松。只要花费一点点努力完成迁移，你的代码与实验会随着 MMEngine 的发展而与时俱进；如果再花费一点努力，MMEngine 的配置系统可以让你更加高效地管理数据、模型、实验。便利性与可靠性，这些正是我们努力的目标。

蓝色药丸，还是红色药丸——你愿意加入吗？

</details>

## 下一步的建议

如果你想要进一步地：

<details>
<summary>实现自己的模型结构</summary>

参考[模型（Model）](./model.md)

</details>

<details>
<summary>使用自己的数据集</summary>

MMEngine 使用和 pytorch 一致的 `dataloader` ，请参考 pytorch 相关文档进行构建

同时 MMEngine 提供了一个进阶的[数据集基类](../advanced_tutorials/basedataset.md)供下游库与用户使用，如有兴趣也可以阅读文档了解

</details>

<details>
<summary>更换模型评测/验证指标</summary>

参考[模型精度评测（Evaluation）](./evaluation.md)

</details>

<details>
<summary>调整优化器封装（如开启混合精度训练、梯度累积等）与更换优化器</summary>

参考[优化器封装（OptimWrapper）](./optim_wrapper.md)

</details>

<details>
<summary>动态调整学习率等参数（如 warmup ）</summary>

参考[优化器参数调整策略（Parameter Scheduler）](./param_scheduler.md)

</details>

<details>
<summary>数据增强与预处理</summary>

参考[数据变换（Data Transform）](./data_transform.md)

</details>

<details>
<summary>其他</summary>

- 左侧的“示例”中包含更多常用的与新特性的示例代码可供参考
- “进阶教程”中有更多面向资深开发者的内容，可以更加灵活地配置训练流程、日志、可视化等
- 如果以上所有内容都无法实现你的新想法，那么[钩子（Hook）](./hook.md)值得一试

</details>
