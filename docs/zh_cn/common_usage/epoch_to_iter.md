# 从 EpochBased 切换至 IterBased

MMEngine 支持两种训练模式，基于轮次的 EpochBased 方式和基于迭代次数的 IterBased 方式，这两种方式在下游算法库均有使用，例如 [MMDetection](https://github.com/open-mmlab/mmdetection) 默认使用 EpochBased 方式，[MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 默认使用 IterBased 方式。

MMEngine 很多模块默认以 EpochBased 的模式执行，例如 `ParamScheduler`, `LoggerHook`, `CheckpointHook` 等，常见的 EpochBased 配置写法如下：

```python
param_scheduler = dict(
    type='MultiStepLR',
    milestones=[6, 8]
    by_epoch=True  # by_epoch 默认为 True，这边显式的写出来只是为了方便对比
)

default_hooks = dict(
    logger=dict(type='LoggerHook'),
    checkpoint=dict(type='CheckpointHook', interval=2),
)

train_cfg = dict(
    by_epoch=True,  # by_epoch 默认为 True，这边显式的写出来只是为了方便对比
    max_epochs=10,
    val_interval=2
)

log_processor = dict(
    by_epoch=True
)  # log_processor 的 by_epoch 默认为 True，这边显式的写出来只是为了方便对比， 实际上不需要设置

runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
    param_scheduler=param_scheduler
    default_hooks=default_hooks,
    log_processor=log_processor,
    train_cfg=train_cfg,
    resume=True,
)
```

如果想按照 iter 训练模型，需要做以下改动：

1. 将 `train_cfg` 中的 `by_epoch` 设置为 `False`，同时将 `max_iters` 设置为训练的总 iter 数，`val_iterval` 设置为验证间隔的 iter 数。

   ```python
   train_cfg = dict(
       by_epoch=False,
       max_iters=10000,
       val_interval=2000
   )
   ```

2. 将 `default_hooks` 中的 `logger` 的 `log_metric_by_epoch` 设置为 False， `checkpoint` 的 `by_epoch` 设置为 `False`。

   ```python
   default_hooks = dict(
       logger=dict(type='LoggerHook', log_metric_by_epoch=False),
       checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
   )
   ```

3. 将 `param_scheduler` 中的 `by_epoch` 设置为 `False`，并將 `epoch` 相关的参数换算成 `iter`

   ```python
   param_scheduler = dict(
       type='MultiStepLR',
       milestones=[6000, 8000],
       by_epoch=False,
   )
   ```

   除了这种方式，如果你能保证 IterBasedTraining 和 EpochBasedTraining 总 iter 数一致，直接设置 `convert_to_iter_based` 为 `True` 即可。

   ```python
   param_scheduler = dict(
       type='MultiStepLR',
       milestones=[6, 8]
       convert_to_iter_based=True
   )
   ```

4. 将 `log_processor` 的 `by_epoch` 设置为 `False`。

   ```python
   log_processor = dict(
       by_epoch=False
   )
   ```

以 [15 分钟教程训练 CIFAR10 数据集](../get_started/15_minutes.md)为例：

<table class="docutils">
<thead>
  <tr>
    <th>Step</th>
    <th>Training by epoch</th>
    <th>Training by iteration</th>
<tbody>
<tr>
  <td>Build model</td>
  <td colspan="2"><div>

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

</td>
  </div>
</tr>

<tr>
  <td>Build dataloader</td>

<td colspan="2">

```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
train_dataloader = DataLoader(
    batch_size=32,
    shuffle=True,
    dataset=torchvision.datasets.CIFAR10(
        'data/cifar10',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg)])))

val_dataloader = DataLoader(
    batch_size=32,
    shuffle=False,
    dataset=torchvision.datasets.CIFAR10(
        'data/cifar10',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg)])))
```

</td>
</tr>

<tr>
  <td>Prepare metric</td>
  <td colspan="2">

```python
from mmengine.evaluator import BaseMetric

class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        # save the middle result of a batch to `self.results`
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        # return the dict containing the eval results
        # the key is the name of the metric name
        return dict(accuracy=100 * total_correct / total_size)
```

</td>
  </tr>

<tr>
  <td>Configure default hooks</td>
  <td valign="top" class='two-column-table-wrapper' width="50%" colspan="1">
  <div style="overflow-x: auto">

```python
default_hooks = dict(
    logger=dict(type='LoggerHook', log_metric_by_epoch=True),
    checkpoint=dict(type='CheckpointHook', interval=2, by_epoch=True),
)
```

</div>
  </td>

<td valign="top" class='two-column-table-wrapper' width="50%" colspan="1">
  <div style="overflow-x: auto">

```python
default_hooks = dict(
    logger=dict(type='LoggerHook', log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
)
```

</div>
  </td>
</tr>

<tr>
  <td>Configure parameter scheduler</td>
  <td valign="top" class='two-column-table-wrapper' width="50%" colspan="1">
  <div style="overflow-x: auto">

```python
param_scheduler = dict(
    type='MultiStepLR',
    milestones=[6, 8],
    by_epoch=True,
)
```

</div>
  </td>

<td valign="top" class='two-column-table-wrapper' width="50%" colspan="1">
  <div style="overflow-x: auto">

```python
param_scheduler = dict(
    type='MultiStepLR',
    milestones=[6000, 8000],
    by_epoch=False,
)
```

</div>
  </td>
</tr>

<tr>
  <td>Configure log_processor</td>
  <td valign="top" class='two-column-table-wrapper' width="50%" colspan="1">
  <div style="overflow-x: auto">

```python
# The default configuration of log_processor is used for epoch based training.
# Defining it here additionally is for building runner with the same way.
log_processor = dict(by_epoch=True)
```

</div>
  </td>

<td valign="top" class='two-column-table-wrapper' width="50%" colspan="1">
  <div style="overflow-x: auto">

```python
log_processor = dict(by_epoch=False)
```

</div>
  </td>
</tr>

<tr>
  <td>Configure train_cfg</td>
  <td valign="top" class='two-column-table-wrapper' width="50%" colspan="1">
  <div style="overflow-x: auto">

```python
train_cfg = dict(
    by_epoch=True,
    max_epochs=10,
    val_interval=2
)
```

</div>
  </td>

<td valign="top" class='two-column-table-wrapper' width="50%" colspan="1">
  <div style="overflow-x: auto">

```python
train_cfg = dict(
    by_epoch=False,
    max_iters=10000,
    val_interval=2000
)
```

</div>
  </td>
</tr>

<tr>
  <td>Build Runner</td>
  <td colspan="2">

```python
from torch.optim import SGD
from mmengine.runner import Runner
runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=train_cfg,
    log_processor=log_processor,
    default_hooks=default_hooks,
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
)
runner.train()
```

</td>
</tr>

</thead>
</table>

```{note}
如果基础配置文件为 train_dataloader 配置了基于 iteration/epoch 采样的 sampler，则需要在当前配置文件中将其更改为指定类型的 sampler，或将其设置为 None。当 dataloader 中的 sampler 为 None，MMEngine 或根据 train_cfg 中的 by_epoch 参数选择 `InfiniteSampler`（False） 或 `DefaultSampler`（True）。
```

```{note}
如果基础配置文件在 train_cfg 中指定了 type，那么必须在当前配置文件中将 type 覆盖为（IterBasedTrainLoop 或 EpochBasedTrainLoop），而不能简单的指定 by_epoch 参数。
```
