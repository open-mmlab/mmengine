# EpochBasedTraining to IterBasedTraining

Epoch-based training and iteration-based training are two commonly used training way in MMEngine. For example, downstream repositories like [MMDetection](https://github.com/open-mmlab/mmdetection) choose to train the model by epoch and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) choose to train the model by iteration.

Many modules in MMEngine default to training models by epoch, such as `ParamScheduler`, `LoggerHook`, `CheckPointHook`, etc. Therefore, you need to adjust the configuration of these modules if you want to train by iteration. For example, a commonly used epoch based configuration is as follows:

```python
param_scheduler = dict(
    type='MultiStepLR',
    milestones=[6, 8]
    by_epoch=True  # by_epoch is True by default
)

default_hooks = dict(
    logger=dict(type='LoggerHook', log_metric_by_epoch=True),  # log_metric_by_epoch is True by default
    checkpoint=dict(type='CheckpointHook', interval=2, by_epoch=True),  # by_epoch is True by default
)

train_cfg = dict(
    by_epoch=True,  # set by_epoch=True or type='EpochBasedTrainLoop'
    max_epochs=10,
    val_interval=2
)

log_processor = dict(
    by_epoch=True
)  # This is the default configuration, and just set it here for comparison.

runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    # Assuming train_dataloader is configured with an epoch-based sampler
    train_dataloader=train_dataloader_cfg,
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
    param_scheduler=param_scheduler
    default_hooks=default_hooks,
    log_processor=log_processor,
    train_cfg=train_cfg,
    resume=True,
)
```

There are four steps to convert the above configuration to iteration based training:

1. Set `by_epoch` in `train_cfg` to False, and set `max_iters` to the total number of training iterations and `val_interval` to the interval between validation iterations.

   ```python
   train_cfg = dict(
       by_epoch=False,
       max_iters=10000,
       val_interval=2000
     )
   ```

2. Set `log_metric_by_epoch` to `False` in logger and `by_epoch` to `False` in checkpoint.

   ```python
   default_hooks = dict(
       logger=dict(type='LoggerHook', log_metric_by_epoch=False),
       checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
   )
   ```

3. Set `by_epoch` in param_scheduler to `False` and convert any epoch-related parameters to iteration.

   ```python
   param_scheduler = dict(
       type='MultiStepLR',
       milestones=[6000, 8000],
       by_epoch=False,
   )
   ```

   Alternatively, if you can ensure that the total number of iterations for IterBasedTraining and EpochBasedTraining is the same, simply set `convert_to_iter_based` to True.

   ```python
   param_scheduler = dict(
       type='MultiStepLR',
       milestones=[6, 8]
       convert_to_iter_based=True
   )
   ```

4. Set by_epoch in log_processor to False.

   ```python
   log_processor = dict(
       by_epoch=False
   )
   ```

Take [training CIFAR10](../get_started/15_minutes.md) as an example:

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
    milestones=[6, 8]
    by_epoch=True
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
If the base configuration file has configured a epoch/iteration based sampler for the train_dataloader, then it is necessary to change it to a specified type of sampler in the current configuration file, or set it to None. When the sampler in the dataloader is set to None, MMEngine will choose either the InfiniteSampler (when by_epoch is False) or the DefaultSampler (when by_epoch is True) according to the train_cfg parameter.
```

```{note}
If `type` is configured for the `train_cfg` in the base configuration, you must overwrite the type to target type (EpochBasedTrainLoop or IterBasedTrainLoop) rather than simply set `by_epoch` to True/False.
```
