# 使用 MMEngine 来训练模型

MMEngine 实现了 OpenMMLab 算法库的新一代训练架构，为算法模型的训练、测试、推理和可视化定义了一套基类与接口。
和 OpenMMLab 算法库的上一代训练架构相比，它具有如下三个特点：

- 统一：为不同方向算法模型的训练、测试、推理、和可视化过程进行了抽象并定义了一套统一的接口
- 清晰：封装的层次与逻辑清晰简单，抽象的定义与接口更加清晰，模块的拆分与边界更加清晰
- 灵活：在统一的基础框架内，模块可以灵活拓展和插拔，支持各类型算法和学习范式，包括少样本和零样本学习，自监督、半监督、和弱监督学习，和模型的蒸馏、剪枝、与量化。

## 样例

以在 CIFAR-10 数据集上训练一个 ResNet-50 模型为例，我们将使用 80 行以内的代码，利用 mmengine 构建一个完整的、
可配置的训练和验证流程。完整的代码如下所示，之后我们会详细介绍每一部分的代码：

```python
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD
from torch.utils.data import DataLoader

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner


class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels


class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        return dict(accuracy=100 * total_correct / total_size)


norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
train_dataloader = DataLoader(batch_size=32,
                              dataset=torchvision.datasets.CIFAR10(
                                  'data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(**norm_cfg)
                                  ])))

val_dataloader = DataLoader(batch_size=32,
                            dataset=torchvision.datasets.CIFAR10(
                                'data/cifar10',
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(**norm_cfg)
                                ])))

runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
)
runner.train()
```

在以上样例中，我们构建了一系列训练和验证所需要的核心组件，并利用这些组件构建了一个执行器（Runner），通过调用这
个执行器的 `train()` 接口启动训练。如下为构建该执行器所需的各核心组件：

```python
runner = Runner(
    # 用以训练和验证的模型，需要满足特定的接口需求
    model=MMResNet50(),
    # 工作路径，用以保存训练日志、权重文件信息
    work_dir='./work_dir',
    # 训练数据加载器，需要满足 PyTorch 数据加载器协议
    train_dataloader=train_dataloader,
    # 优化器包装，用于模型优化，并提供 AMP、梯度累积等附加功能
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    # 训练配置，用于指定训练周期、验证间隔等信息
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    # 验证数据加载器，需要满足 PyTorch 数据加载器协议
    val_dataloader=val_dataloader,
    # 验证配置，用于指定验证所需要的额外参数
    val_cfg=dict(),
    # 用于验证的评测器，这里使用默认评测器，并评测指标
    val_evaluator=dict(metrics=Accuracy()),
)
```

在样例中，可以看到 MMEngine 的执行器对模型和评测指标的接口有约定。

具体而言，对于**模型**，我们约定其 forward 方法除了接受来自数据集的若干参数外，还需要接受额外的参数 `mode`，对
于训练，我们需要 `mode` 接受字符串 "loss"，并返回一个包含 "loss" 字段的字典；对于验证，我们需要 `mode` 接受字符
串 "predict"，并返回同时包含预测信息和真实信息的结果。

```python
class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels
```

对于**评测指标**，我们约定其需要继承 `BaseMetric`，并实现 `process` 和 `compute_metrics` 方法。其中 `process`
方法接受数据集的输出和模型 `mode="predict"` 时的输出，此时的数据为一个批次的数据，对这一批次的数据进行处理后，
保存信息至 `self.results` 属性。而 `compute_metrics` 接受 `results` 参数，这一参数的输入为 `process` 中保存的
所有信息（包括分布式环境中各个进程中保存的信息），利用这些信息计算，并返回保存有评测指标结果的字典。

```python
class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        # 将一个批次的中间结果保存至 `self.results`
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        # 返回保存有评测指标结果的字典，其中键为指标名称
        return dict(accuracy=100 * total_correct / total_size)
```

## 组件

MMEngine 将算法模型训练、推理、测试和可视化过程中的各个组件进行了抽象，定义了如下几个组件和他们的相关接口，这些组件的关系如下图所示：

![runner_modules](https://user-images.githubusercontent.com/40779233/165018333-c54a3405-a566-4de6-a1d5-f2a3a836bd41.jpeg)

以下根据上图简述这些模块的功能与联系，用户可以通过各个组件的用户文档了解他们。

- [执行器（Runner）](./runner.md)：负责执行训练、测试和推理任务并管理这些过程中所需要的各个组件。
- [钩子（Hook）](./hook.md)：负责在训练、测试、推理任务执行过程中的特定位置执行自定义逻辑。
- [数据集（Dataset）](./basedataset.md)：负责在训练、测试、推理任务中构建数据集，并将数据送给模型。实际使用过程中会被数据加载器（DataLoader）封装一层，数据加载器会启动多个子进程来加载数据。
- [模型（Model）](./model.md)：在训练过程中接受数据、输出 loss，在测试、推理任务中接受数据，并进行预测。分布式训练等情况下会被模型的封装器（Model Wrapper，如 .`nn.DistributedDataParallel`）封装一层。
- [评测指标与评测器（Metrics & Evaluator）](./metric_and_evaluator.md)：评测器负责基于数据集对模型的预测进行评估。评测器内还有一层抽象是评测指标，负责计算具体的一个或多个评测指标（如召回率、正确率等）。
- [数据元素（Data Element）](./data_element.md)：评测器，模型和数据之间交流的接口使用数据元素进行封装。
- [参数调度器（Parameter Scheduler）](./param_scheduler.md)：训练过程中，对学习率、动量等参数进行动态调整。
- [优化器（Optimizer）](./optimizer_wrapper.md)：优化器负责在训练过程中执行反向传播优化模型。实际使用过程中会被优化器封装（OptimWrapper）封装一层，实现梯度累加、混合精度训练等功能。
- [日志管理（Logging Modules）](./logging.md)：负责管理 Runner 运行过程中产生的各种日志信息。其中消息枢纽 （MessageHub）负责实现组件与组件、执行器与执行器之间的数据共享，日志处理器（Log Processor）负责对日志信息进行处理，处理后的日志会分别发送给执行器的日志器（Logger）和可视化器（Visualizer）进行日志的管理与展示。
- [配置类（Config）](./config.md)：在 OpenMMLab 算法库中，用户可以通过编写 config 来配置训练、测试过程以及相关的组件。
- [注册器（Registry）](./registry.md)：负责管理算法库中具有相同功能的模块。MMEngine 根据对算法库模块的抽象，定义了一套根注册器，算法库中的注册器可以继承自这套根注册器，实现模块的跨算法库调用。
- [分布式通信原语（Distributed Communication Primitives）](./distributed.md)：负责在程序分布式运行过程中不同进程间的通信。这套接口屏蔽了分布式和非分布式环境的区别，同时也自动处理了数据的设备和通信后端。
- [其他工具（Utils）](./utils.md)：还有一些工具性的模块，如管理器混入（ManagerMixin），它实现了一种全局变量的创建和获取方式，Runner 内很多全局可见对象的基类就是 ManagerMixin。
