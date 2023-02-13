# 从 EpochBased 切换至 IterBased

MMEngine 很多模块的默认配置都会按照 Epoch 训练模型，例如 ParamScheduler, LoggerHook, CheckPointHook 等，因此按照 epoch 训练时，不需要为这些模块设置 `by_epoch` 参数：

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

如果需要按照 iter 训练模型，则需要做以下改动：

1. 将 `train_cfg` 中的 `by_epoch` 设置为 `False`，同时设置 `max_iters` 为训练的总 iter 数量，val_iterval 为验证的间隔 iter 数量。

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
