# 设置日志、权重保存、验证的频率

MMEngine 支持两种训练模式，基于轮次的 `EpochBased` 方式和基于迭代次数的 `IterBased` 方式，这两种方式在下游算法库均有使用，例如 [MMDetection](https://github.com/open-mmlab/mmdetection) 默认使用 EpochBased 方式，[MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 默认使用 IterBased 方式。

在不同的训练模式下，MMEngine 间隔（interval）的语义会有区别，`EpochBased` 的间隔以 `Epoch` 为单位，`IterBased` 以 `Iteration` 为单位。

## 设置训练和验证的间隔

设置 [Runner](mmengine.runner.Runner) 初始化参数 `train_cfg` 中的 `val_interval` 值即可定制训练和验证的间隔。

- EpochBased

在 `EpochBased` 模式下，`val_interval` 的默认值为 1，表示训练一个 Epoch，验证一次。

```python
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

- IterBased

在 `IterBased` 模式下，`val_interval` 的默认值为 1000，表示训练迭代 1000 次，验证一次。

```python
runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=False, max_iters=10000, val_interval=2000),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
)
runner.train()
```

## 设置保存权重的间隔

设置 [CheckpointHook](mmengine.hooks.CheckpointHook) 的 `interval` 值即可定制保存权重的间隔。

- EpochBased

在 `EpochBased` 模式下，`interval` 的默认值为 1，表示训练一个 Epoch，保存一次权重。

```python
# 将 interval 设置为 2，表示每 2 个 epoch 保存一次权重
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2))
runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
    default_hooks=default_hooks,
)
runner.train()
```

- IterBased

默认以 Epoch 为单位保存权重，如果希望以 Iteration 为单位，需设置 `by_epoch=False`。

```python
# 设置 by_epoch=False 以及 interval = 500，表示每 500 个 iteration 保存一次权重
default_hooks = dict(checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500))
runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=False, max_iters=10000, val_interval=1000),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
    default_hooks=default_hooks,
)
runner.train()
```

`CheckpointHook` 的更多用法可查看 [CheckpointHook 教程](../tutorials/hook.md#checkpointhook)。

## 设置打印日志的间隔

默认情况下，每迭代 10 次往终端打印 1 次日志，可以通过设置 [LoggerHook](mmengine.hooks.LoggerHook) 的 `interval` 参数进行设置。

```python
# 设置每 20 次打印一次
default_hooks = dict(logger=dict(type='LoggerHook', interval=20))
runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
    default_hooks=default_hooks,
)
runner.train()
```

`LoggerHook` 的更多用法可查看 [LoggerHook 教程](../tutorials/hook.md#loggerhook)。
