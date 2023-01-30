# 15 分钟上手 MMEngine

以在 CIFAR-10 数据集上训练一个 ResNet-50 模型为例，我们将使用 80 行以内的代码，利用 MMEngine 构建一个完整的、
可配置的训练和验证流程，整个流程包含如下步骤：

1. [构建模型](#构建模型)
2. [构建数据集和数据加载器](#构建数据集和数据加载器)
3. [构建评测指标](#构建评测指标)
4. [构建执行器并执行任务](#构建执行器并执行任务)

## 构建模型

首先，我们需要构建一个**模型**，在 MMEngine 中，我们约定这个模型应当继承 `BaseModel`，并且其 `forward` 方法除了接受来自数据集的若干参数外，还需要接受额外的参数 `mode`：对于训练，我们需要 `mode` 接受字符串 "loss"，并返回一个包含 "loss" 字段的字典；对于验证，我们需要 `mode` 接受字符串 "predict"，并返回同时包含预测信息和真实信息的结果。

```python
import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel


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

## 构建数据集和数据加载器

其次，我们需要构建训练和验证所需要的**数据集 (Dataset)**和**数据加载器 (DataLoader)**。
对于基础的训练和验证功能，我们可以直接使用符合 PyTorch 标准的数据加载器和数据集。

```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
train_dataloader = DataLoader(batch_size=32,
                              shuffle=True,
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
                            shuffle=False,
                            dataset=torchvision.datasets.CIFAR10(
                                'data/cifar10',
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(**norm_cfg)
                                ])))
```

## 构建评测指标

为了进行验证和测试，我们需要定义模型推理结果的**评测指标**。我们约定这一评测指标需要继承 `BaseMetric`，并实现 `process` 和 `compute_metrics` 方法。其中 `process` 方法接受数据集的输出和模型 `mode="predict"`
时的输出，此时的数据为一个批次的数据，对这一批次的数据进行处理后，保存信息至 `self.results` 属性。
而 `compute_metrics` 接受 `results` 参数，这一参数的输入为 `process` 中保存的所有信息
（如果是分布式环境，`results` 中为已收集的，包括各个进程 `process` 保存信息的结果），利用这些信息计算并返回保存有评测指标结果的字典。

```python
from mmengine.evaluator import BaseMetric

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

## 构建执行器并执行任务

最后，我们利用构建好的**模型**，**数据加载器**，**评测指标**构建一个**执行器 (Runner)**，同时在其中配置
**优化器**、**工作路径**、**训练与验证配置**等选项，即可通过调用 `train()` 接口启动训练：

```python
from torch.optim import SGD
from mmengine.runner import Runner

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
    val_evaluator=dict(type=Accuracy),
)

runner.train()
```

最后，让我们把以上部分汇总成为一个完整的，利用 MMEngine 执行器进行训练和验证的脚本：

<a href="https://colab.research.google.com/github/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/get_started.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"/></a>

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
                              shuffle=True,
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
                            shuffle=False,
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

输出的训练日志如下：

```
2022/08/22 15:51:53 - mmengine - INFO -
------------------------------------------------------------
System environment:
    sys.platform: linux
    Python: 3.8.12 (default, Oct 12 2021, 13:49:34) [GCC 7.5.0]
    CUDA available: True
    numpy_random_seed: 1513128759
    GPU 0: NVIDIA GeForce GTX 1660 SUPER
    CUDA_HOME: /usr/local/cuda
...

2022/08/22 15:51:54 - mmengine - INFO - Checkpoints will be saved to /home/mazerun/work_dir by HardDiskBackend.
2022/08/22 15:51:56 - mmengine - INFO - Epoch(train) [1][10/1563]  lr: 1.0000e-03  eta: 0:18:23  time: 0.1414  data_time: 0.0077  memory: 392  loss: 5.3465
2022/08/22 15:51:56 - mmengine - INFO - Epoch(train) [1][20/1563]  lr: 1.0000e-03  eta: 0:11:29  time: 0.0354  data_time: 0.0077  memory: 392  loss: 2.7734
2022/08/22 15:51:56 - mmengine - INFO - Epoch(train) [1][30/1563]  lr: 1.0000e-03  eta: 0:09:10  time: 0.0352  data_time: 0.0076  memory: 392  loss: 2.7789
2022/08/22 15:51:57 - mmengine - INFO - Epoch(train) [1][40/1563]  lr: 1.0000e-03  eta: 0:08:00  time: 0.0353  data_time: 0.0073  memory: 392  loss: 2.5725
2022/08/22 15:51:57 - mmengine - INFO - Epoch(train) [1][50/1563]  lr: 1.0000e-03  eta: 0:07:17  time: 0.0347  data_time: 0.0073  memory: 392  loss: 2.7382
2022/08/22 15:51:57 - mmengine - INFO - Epoch(train) [1][60/1563]  lr: 1.0000e-03  eta: 0:06:49  time: 0.0347  data_time: 0.0072  memory: 392  loss: 2.5956
2022/08/22 15:51:58 - mmengine - INFO - Epoch(train) [1][70/1563]  lr: 1.0000e-03  eta: 0:06:28  time: 0.0348  data_time: 0.0072  memory: 392  loss: 2.7351
...
2022/08/22 15:52:50 - mmengine - INFO - Saving checkpoint at 1 epochs
2022/08/22 15:52:51 - mmengine - INFO - Epoch(val) [1][10/313]    eta: 0:00:03  time: 0.0122  data_time: 0.0047  memory: 392
2022/08/22 15:52:51 - mmengine - INFO - Epoch(val) [1][20/313]    eta: 0:00:03  time: 0.0122  data_time: 0.0047  memory: 308
2022/08/22 15:52:51 - mmengine - INFO - Epoch(val) [1][30/313]    eta: 0:00:03  time: 0.0123  data_time: 0.0047  memory: 308
...
2022/08/22 15:52:54 - mmengine - INFO - Epoch(val) [1][313/313]  accuracy: 35.7000
```

基于 PyTorch 和基于 MMEngine 的训练流程对比如下：

![output](https://user-images.githubusercontent.com/57566630/203142869-cfe5f855-f391-4fd4-a80c-beecf1bd111f.gif)

除了以上基础组件，你还可以利用**执行器**轻松地组合配置各种训练技巧，如开启混合精度训练和梯度累积（见 [优化器封装（OptimWrapper）](../tutorials/optim_wrapper.md)）、配置学习率衰减曲线（见 [评测指标与评测器（Metrics & Evaluator）](../tutorials/evaluation.md)）等。
