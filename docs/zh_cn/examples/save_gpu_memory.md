# 节省显存

在深度学习训练推理过程中显存容量至关重要，其决定了模型是否能成功运行。常见的节省显存办法包括：

- 梯度累加

  梯度累加是指在每计算一个批次的梯度后，不进行清零而是进行梯度累加，当累加到一定的次数之后，再更新网络参数和梯度清零。 通过这种参数延迟更新的手段，实现与采用大 batch 尺寸相近的效果，达到节省显存的目的。但是需要注意如果模型中包含 batch normalization 层，使用梯度累加会对性能有一定影响。

- 梯度检查点

  梯度检查点是一种以时间换空间的方法，通过减少保存的激活值来压缩模型占用空间，但是在计算梯度时必须重新计算没有存储的激活值。在 torch.utils.checkpoint 包中已经实现了对应功能。简要实现过程是：在前向阶段传递到 checkpoint 中的 forward 函数会以 `torch.no_grad` 模式运行，并且仅仅保存输入参数和 forward 函数，在反向阶段重新计算其 forward 输出值。

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

## 大模型训练

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
    cfg=dict(model_wrapper_cfg=dict(
        type='MMFullyShardedDataParallel',
        # 启用 cpu_offload 可以减少内存占用，但会降低训练速度
        cpu_offload=True,
        # 合理选择 auto_wrap_policy 以降低显存占用，参考
        # https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html
        # 目前需要手动将函数注册到 FSDP_WRAP_POLICYS 注册器中
        # 之后的更新会提供更便利的方式
        fsdp_auto_wrap_policy='my_auto_wrap_policy'
        ))
)
runner.train()
```

注意必须在分布式训练环境中 FSDP 才能生效。

```{note}
尽管 PyTorch 1.11 中已经原生支持 FSDP，但并不完善。在 MMEngine 中我们仅在 PyTorch 1.12+ 版本支持 FSDP 技术
```

```{note}
PyTorch 1.12 引入的一些新特性，暂时还未在 MMEngine 中得到支持。以下特性将会在后续更新中逐渐支持：

- Local/Shareded 模式的模型权重保存/加载方式（目前仅使用 Full 模式）
- 支持在配置文件中配置 `sharding_strategy` 和 `mixed_precision`
```

```{warning}
目前为止，MMEngine 的 FSDP 暂时不支持保存完整的 `optimizer_state`，因此无法继续训练（resume training）。我们将在接下来的更新中支持这一功能
```
