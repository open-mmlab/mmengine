# 执行器（Runner）基础

- [执行器（Runner）基础](#执行器runner基础)
  - [执行器示例与最佳实践](#执行器示例与最佳实践)
    - [面向初学者的示例代码（并非最佳实践）](#面向初学者的示例代码并非最佳实践)
    - [示例代码讲解](#示例代码讲解)
    - [使用配置文件（最佳实践）](#使用配置文件最佳实践)
  - [基本数据流](#基本数据流)
  - [执行器的设计理念（可选）](#执行器的设计理念可选)
  - [下一步的建议](#下一步的建议)

欢迎来到 MMEngine 用户界面的核心——执行器！

作为 MMEngine 中的“集大成者”，执行器涵盖了整个框架的方方面面，肩负着串联所有组件的重要责任；因此，其中的代码、内容也显得较为晦涩难懂。
但是**不用担心**！在这篇教程中，我们将隐去繁杂的细节，速览执行器常用的接口、功能、示例，为你呈现一个清晰易懂的用户界面。阅读完本篇教程，你将会：

- 掌握执行器的常见使用方式与最佳实践
- 了解执行器基本数据流
- 亲身感受使用执行器的优越性（也许）

## 执行器示例与最佳实践

使用执行器构建属于你自己的训练流程，通常有两种开始方式：

- 参考 [API 文档](mmengine.runner.Runner)，逐项确认和配置参数
- 在已有配置（如 [15 分钟上手](../get_started/15_minutes.md)或 mmdet 等下游算法库）的基础上，进行定制化修改

两种方式各有利弊。使用前者，初学者很容易迷失在茫茫多的参数项中不知所措；而使用后者，一份过度精简或过度详细的参考配置都不利于初学者快速找到所需内容。

解决上述问题的关键在于，把执行器作为备忘录：掌握其中最常用的部分，并在有特殊需求时聚焦感兴趣的部分。下面我们将通过一个适合初学者参考的例子，说明其中最常用的参数，并为一些不常用参数给出进阶指引。

### 面向初学者的示例代码（并非最佳实践）

> 提醒⚠️：这个例子省略了大量实现细节，并且与 15 分钟教程的示例有所不同（我们将在下面解释原因）。我们希望你在本教程中更多地关注整体结构，而非具体模块的实现。这种“自顶向下”的思考方式是我们所倡导的。别担心，之后你将有充足的机会和指引，聚焦于自己想要改进的模块

<details>
<summary>点击展开一段长长的示例代码。做好准备</summary>

```python
from mmengine.runner import Runner

runner = Runner(
    model=dict( # 你的训练模型
        type='MyAwesomeModel',
        layers=50,
        activation='relu'),
    work_dir='exp/my_awesome_model', # 工作路径；模型检查点、日志等都将存储在工作路径中

    # 训练相关配置
    train_dataloader=dict( # 训练所用数据加载器，概念与 `torch` 一致
        dataset=dict(
            type='MyDataset',
            is_train=True),
        sampler=dict(
            type='DefaultSampler',
            shuffle=True),
        batch_size=4,
        num_workers=8),
    train_cfg=dict( # 训练所用配置，控制循环（Loop）行为
        by_epoch=True,
        max_epochs=10,
        val_interval=1),
    optim_wrapper=dict( # 优化器封装，MMEngine 中的新概念，提供更丰富的优化选择
        type='OptimizerWrapper', # 通常使用默认即可，可缺省；有特殊需求可查阅文档更换，如 'AmpOptimWrapper' 开启混合精度训练
        optimizer=dict( # 等同于 `torch` 的优化器
            type='SGD',
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0001)),
    param_scheduler=dict( # 控制训练中的学习率、动量调节
        type='MultiStepLR',
        milestones=[1, 2]),

    # 验证所需配置
    val_dataloader=..., # 与训练类似，此处省略
    val_cfg=dict(),
    val_evaluator=dict(type='MyValMetric'), # 验证指标与验证器封装，MMEngine 中的新概念，可自由实现与配置，也可缺省

    # 测试所需配置，与验证配置类似
    test_dataloader=...,
    test_cfg=dict(),
    test_evaluator=dict(type='MyTestMetric'),

    # 其他环境相关、进阶配置；若无特殊需要，尽量缺省
    default_hooks=dict( # 钩子属于进阶用法，如无特殊需要，尽量缺省
        checkpoint=dict(
            type='CheckpointHook',
            interval=1))
    launcher='none', # 与 `env_cfg` 共同构成分布式训练环境配置
    env_cfg=...,
    log_level='INFO' # 日志等级
)
```

</details>

### 示例代码讲解

真是一个长长的例子！但是如果你通读了上述样例，即使不了解实现细节，你也一定大体理解了这个训练流程，并感叹于执行器代码的紧凑与可读性（也许）。这也是 MMEngine 所期望的：结构化、模块化、标准化的训练流程。但需要明确的是：上述例子仍非执行器的最佳实践，而是一个中间过渡。

上述例子可能会让你产生如下问题：

<details>
<summary>参数项实在是太多了！</summary>

不用担心，正如我们前面所说，**把执行器作为备忘录**。执行器涵盖了方方面面，防止你漏掉重要内容；但你不需要配置所有参数。如[15分钟上手](../tutorials/../get_started/15_minutes.md)中的极简例子（甚至，舍去 `val_evaluator`）也可以正常运行。所有的参数由你的需求驱动，不关注的内容往往缺省值也可以工作得很好。

</details>

<details>
<summary>似乎和 15 分钟上手的写法截然不同？为什么传入参数是`dict`？</summary>

是的，这与 MMEngine 的风格相关。在 MMEngine 中我们提供了两种不同风格的执行器构建方式：a）基于配置与注册机制的，以及 b）基于手动构建的。如果你感到迷惑，下面的例子将给出一个对比：

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

类似上述例子，执行器中的参数大多同时支持两种输入类型。以`dict`作为输入时，该模块会在执行器内部被构建。如果你对于注册机制并不了解，你可以暂且简单理解为，执行器会根据`type`寻找被装饰器修饰的类，调用其`__init__`方法，并将`dict`中其余参数传入（并不确切，但足以理解、处理大部分情况）。如果你在使用中发现问题，或者想要进一步了解完整用法，我们推荐阅读[注册器（Registry）](./registry.md)文档。

这一段内容涉及到 MMEngine 内部实现、注册器设计模式以及`Python`参数传递机制，因而对于初学者来说相对艰深、不易理解。而且，虽然紧凑、可读性强，这个例子对于 IDE 跳转和 debug 并不友好，因此并非最佳实践。但理解这部分内容仍然十分重要：它为理解执行器真正的最佳实践带来曙光。

如果你作为初学者无法立刻理解，使用*手动构建的方式*依然不失为一种好选择，甚至在小规模使用时是一种推荐方式。我们也常挣扎于是否要在教程中展示这些内容，但最终我们确信，配置文件——一种基于配置与注册机制的构建方式——是使用 MMEngine 的最佳实践，并且该方式已经在 OpenMMLab 的下游库中广泛使用、成为事实标准。我们将在接下来的章节中略微修改示例，从而展示该部分。

</details>

<details>
<summary>我来自 mmdet/mmcls...下游库，为什么例子写法与我接触的不同？</summary>

OpenMMLab 下游库广泛采用了配置文件的方式。我们将在下个章节，基于上述示例稍微变换，从而展示配置文件——MMEngine 中执行器的最佳实践——的用法。

</details>

### 使用配置文件（最佳实践）

上述例子中我们大量使用`dict`作为参数。如果你对`Python`的参数传递有所了解，你可能会意识到示例代码等价于如下代码段

```python
my_cfg = dict(
    model=dict(
        type='MyAwesomeModel',
        layers=50,
        activation='relu'),
    work_dir='exp/my_awesome_model',
    ... # 以下参数完全拷贝自原示例，此处省略
)
runner = Runner(**my_cfg)
```

这意味着`my_cfg`可以被导出到各种格式（如`json` `yaml`等）的文件中保存，并在需要时加载。真酷！如果你有过使用`yaml`或`yacd`等工具进行实验配置的经历，你一定会理解配置文件对于大规模实验的重要意义。

MMEngine 提供了一套`Python`语法的、功能强大的配置文件工具。你可以从之前的示例代码中**近乎**（我们将在下面说明）无缝地转换到配置文件

```python
# 以下代码存放在 example_config.py 文件中
# 拷贝自上面的示例，并将每项结尾的逗号删去
model=dict(
    type='MyAwesomeModel',
    layers=50,
    activation='relu')
work_dir='exp/my_awesome_model'
... # 以下省略
```

此时，我们只需要在训练代码中加载配置，然后运行即可

```python
from mmengin.config import Config
from mmengine.runner import Runner
config = Config.fromfile('example_config.py')
runner = Runner.from_cfg(config)
runner.train()
```

> 提醒⚠️：当使用配置文件写法时，你将不得不更多地了解[注册器](./registry.md)，因为此时你的自定义模块的实现代码通常存放在独立文件中，可能并未被正确注册，进而导致构建失败。

> 警告⚠️：虽然与示例中的写法一致，但`from_cfg`与`__init__`的缺省值处理可能存在些微不同，例如`env_cfg`参数。为了可复现性，我们推荐执行器配置文件包含执行器的所有参数，以确保训练的每个部分清晰可见，避免缺省值带来的潜在歧义。因此，我们建议多付出一点点努力，在配置文件中将缺省值补全。

执行器配置文件已经在 OpenMMLab 的众多下游库（mmcls，mmdet...）中被广泛使用，并成为事实标准与最佳实践。配置文件的功能远不止如此，如果你对于继承、覆写等进阶功能感兴趣，请参考[配置（Config）](./config.md)文档。

## 基本数据流

> 提醒⚠️：本章节将会介绍，执行器中各个模块之间的数据传递流向与格式约定。如果你还没有基于 MMEngine 构建一个训练流程，本章节的部分内容可能会比较抽象、枯燥；你也可以暂时跳过，并在将来有需要时结合实践进行阅读。

接下来，我们将**稍微**深入执行器的内部，结合图示来理清其中数据的流向与格式约定。

（此处应该有图，但还没完成）

## 执行器的设计理念（可选）

> 提醒⚠️：这一部分内容并不能教会你如何使用执行器乃至整个 MMEngine，如果你正在被雇主/教授/DDL催促着几个小时内拿出成果，那这部分可能无法帮助到你，请随意跳过。但我们仍强烈推荐抽出时间阅读本章节，这可以帮助你更好地理解并使用 MMEngine

<details>
<summary>放轻松，接下来是哲学时间</summary>

内容暂缺，大纲：

- 结构化的方式搭建训练流程，让使用者聚焦其关注的部分
- 模块化设计，易于替换组件，避免实验代码牵一发动全身
- 强大的配置文件，便于管理大规模实验
- 屏蔽随机数、分布式等恼人的工程细节

</details>

## 下一步的建议

如果你想要进一步地：

<details>
<summary>实现自己的模型结构</summary>

参考[模型（Model）](./model.md)

</details>

<details>
<summary>使用自己的数据集</summary>

MMEngine 使用和 pytorch 一致的`dataloader`，请参考 pytorch 相关文档进行构建

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
