# EpochBasedTraining to IterBasedTraining

Many modules in MMEngine default to training models by epoch, such as `ParamScheduler`, `LoggerHook`, `CheckPointHook`, etc. Therefore, you need to adjust the configuration of these modules when training by iteration. For example, a commonly epoch based configuration is as follows:

```python
param_scheduler = dict(
    type='MultiStepLR',
    milestones=[6, 8]
)

default_hooks = dict(
    logger=dict(type='LoggerHook'),
    checkpoint=dict(type='CheckpointHook', interval=2),
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=10,
    val_interval=2
)

runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
    param_scheduler=param_scheduler
    default_hooks=default_hooks,
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
