# 节省显存

在深度学习训练推理过程中显存容量至关重要，其决定了模型是否能成功运行。常见的节省显存办法包括：

- 启用高效卷积BN评估功能（实验性）

  基于在[这篇论文](https://arxiv.org/abs/2305.11624)中讨论的概念，我们最近在MMCV中[引入](https://github.com/open-mmlab/mmcv/pull/2807)了一个实验性功能：高效卷积BN评估。这个功能的设计目标是在不损害性能的情况下减少网络训练过程中的显存占用。如果你的网络架构包含了一系列连续的Conv+BN模块，而且这些BN层在训练过程中保持在 `eval` 模式（在使用 [MMDetection](https://github.com/open-mmlab/mmdetection)训练对象检测器时很常见），这个功能可以将显存消耗减少超过 $20%$。要启用高效卷积BN评估功能，只需添加以下命令行参数：`--cfg-options efficient_conv_bn_eval="[backbone]"`。当你在输出日志中看到 `Enabling the "efficient_conv_bn_eval" feature for these modules ...`时，意味着功能已成功启用。由于这仍处于实验阶段，我们非常期待听到你对它的使用体验。请在[这个讨论线程](https://github.com/open-mmlab/mmengine/discussions/1252)分享你的使用报告、观察和建议。你的反馈对于进一步的开发和确定是否应将此功能集成到稳定版中至关重要。

- 梯度累加

  梯度累加是指在每计算一个批次的梯度后，不进行清零而是进行梯度累加，当累加到一定的次数之后，再更新网络参数和梯度清零。 通过这种参数延迟更新的手段，实现与采用大 batch 尺寸相近的效果，达到节省显存的目的。但是需要注意如果模型中包含 batch normalization 层，使用梯度累加会对性能有一定影响。

- 梯度检查点

  梯度检查点是一种以时间换空间的方法，通过减少保存的激活值来压缩模型占用空间，但是在计算梯度时必须重新计算没有存储的激活值。在 torch.utils.checkpoint 包中已经实现了对应功能。简要实现过程是：在前向阶段传递到 checkpoint 中的 forward 函数会以 `torch.no_grad` 模式运行，并且仅仅保存 forward 函数的输入和输出，然后在反向阶段重新计算中间层的激活值 （intermediate activations）。

- 大模型训练技术

  最近的研究表明大型模型训练将有利于提高模型质量，但是训练如此大的模型需要巨大的资源，单卡显存已经越来越难以满足存放整个模型，因此诞生了大模型训练技术，典型的如 [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/#zero-overview) 和 FairScale 的[完全分片数据并行](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)（Fully Sharded Data Parallel, FSDP）技术，其允许在数据并行进程之间分片模型的参数、梯度和优化器状态，并同时仍然保持数据并行的简单性。

MMEngine 目前支持`梯度累加`和`大模型训练 FSDP 技术 `。下面说明其用法。

## 梯度累加

配置写法如下所示：

```python
optim_wrapper_cfg = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9),
    # 累加 4 次参数更新一次
    accumulative_counts=4)
```

配合 Runner 使用的完整例子如下：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mmengine.runner import Runner
from mmengine.model import BaseModel

train_dataset = [(torch.ones(1, 1), torch.ones(1, 1))] * 50
train_dataloader = DataLoader(train_dataset, batch_size=2)


class ToyModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, img, label, mode):
        feat = self.linear(img)
        loss1 = (feat - label).pow(2)
        loss2 = (feat - label).abs()
        return dict(loss1=loss1, loss2=loss2)


runner = Runner(
    model=ToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=1),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01),
                       accumulative_counts=4)
)
runner.train()
```

## 梯度检查点

```{note}
MMEngine v0.9.0 开始支持梯度检查点的功能。关于性能的比较可点击 [#1319](https://github.com/open-mmlab/mmengine/pull/1319)。如果你在使用过程中遇到任何问题，欢迎在 [#1319](https://github.com/open-mmlab/mmengine/pull/1319) 反馈。
```

只需在 Runner 的 cfg 参数中配置 `activation_checkpointing` 即可开启梯度检查点。

以[15 分钟上手 MMEngine](../get_started/15_minutes.md) 为例：

```python
cfg = dict(
    activation_checkpointing=['resnet.layer1', 'resnet.layer2', 'resnet.layer3']
)
runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
    launcher=args.launcher,
    cfg=cfg,
)
runner.train()
```

## 大模型训练

```{warning}
如果你有训练大模型的需求，推荐阅读[大模型训练](./large_model_training.md)。
```

PyTorch 1.11 中已经原生支持了 FSDP 技术。配置写法如下所示：

```python
# 位于 cfg 配置文件中
model_wrapper_cfg=dict(type='MMFullyShardedDataParallel', cpu_offload=True)
```

配合 Runner 使用的完整例子如下：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mmengine.runner import Runner
from mmengine.model import BaseModel

train_dataset = [(torch.ones(1, 1), torch.ones(1, 1))] * 50
train_dataloader = DataLoader(train_dataset, batch_size=2)


class ToyModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, img, label, mode):
        feat = self.linear(img)
        loss1 = (feat - label).pow(2)
        loss2 = (feat - label).abs()
        return dict(loss1=loss1, loss2=loss2)


runner = Runner(
    model=ToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=1),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)),
    cfg=dict(model_wrapper_cfg=dict(type='MMFullyShardedDataParallel', cpu_offload=True))
)
runner.train()
```

注意必须在分布式训练环境中 FSDP 才能生效。
