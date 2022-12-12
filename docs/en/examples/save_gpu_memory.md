# Save Memory on GPU

Memory capacity is critical in deep learning training and inference and determines whether the model can run successfully. Common memory saving approaches include:

- Gradient Accumulation

  Gradient accumulation is the mechanism that runs at a configured number of steps accumulating the gradients instead of updating parameters, after which the network parameters are updated and the gradients are cleared. With this technique of delayed parameter update, the result is similar to those scenarios using a large batch size, while the memory of activation can be saved. However, it should be noted that if the model contains a batch normalization layer, using gradient accumulation will impact performance.

- Gradient Checkpointing

  Gradient checkpointing is a time-for-space method that compresses the model by reducing the number of saved activations, however, the unstored activations must be recomputed when calculating the gradient. The corresponding functionality has been implemented in the `torch.utils.checkpoint` package. The implementation can be briefly concluded as that, in the forward phase, the forward function passed to the checkpoint runs in `torch.no_grad` mode and saves only the input and the output of the forward function. Then recalculates its intermediate activations in the backward phase.

- Large Model Training Techniques

  Recent research has shown that training a large model would be helpful to improve performance, but training a model at such a scale requires huge resources, and it is hard to store the entire model in the memory of a single graphics card. Therefore large model training techniques, typically such as [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/#zero-overview) and the Fully Shared Data Parallel ([FSDP](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)) technique introduced in FairScale are introduced. These techniques allow slicing the parameters, gradients, and optimizer states among the parallel processes, while still maintaining the simplicity of the data parallelism.

MMEngine now supports gradient accumulation and large model training FSDP techniques, and the usages are described as follows.

## Gradient Accumulation

The configuration can be written in this way:

```python
optim_wrapper_cfg = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9),
    # update every four times
    accumulative_counts=4)
```

The full example working with `Runner` is as follows.

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

## Large Model Training

`FSDP` is officially supported from PyTorch 1.11. The config can be written in this way:

```python
# located in cfg file
model_wrapper_cfg=dict(type='MMFullyShardedDataParallel', cpu_offload=True)
```

The full example working with `Runner` is as follows.

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

Please be noted that `FSDP` works only in distributed training environments.
