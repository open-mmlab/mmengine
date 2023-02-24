# Setting the Frequency of Logging, Weight Saving, and Validation

MMEngine supports two training modes, `EpochBased` based on epochs and `IterBased` based on the number of iterations. Both of these modes are used in downstream algorithm libraries such as [MMDetection](https://github.com/open-mmlab/mmdetection), which uses the `EpochBased` mode by default, and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), which uses the `IterBased` mode by default.

Under different training modes, the semantics of the interval in MMEngine will be different. The interval in `EpochBased` mode is based on epochs, while that in `IterBased` mode is based on iterations.

## Setting the Interval for Training and Validation

To customize the interval for training and validation, set the `val_interval` parameter in the initialization parameter `train_cfg` of [Runner](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py).

- EpochBased

In `EpochBased` mode, the default value of `val_interval` is 1, which means to validate once after training an epoch.

```
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

In `IterBased` mode, the default value of `val_interval` is 1000, which means to validate once after training 1000 iterations.

```
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

## Setting the Interval for Saving Weights

To customize the interval for saving weights, set the `interval` parameter of [CheckpointHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py).

- EpochBased

In `EpochBased` mode, the default value of `interval` is 1, which means saving weights once after training for one epoch.

```
# Set interval to 2, which means to save weights once every 2 epochs
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

By default, weights are saved in units of epochs. If you want to save weights in units of iterations, you need to set `by_epoch=False`.

```
# Set by_epoch=False and interval=500, which means to save weights once every 500 iterations
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

For more information on how to use `CheckpointHook`, please refer to the [CheckpointHook tutorial](../tutorials/hook.md#checkpointhook).

## Setting the interval for printing logs

By default, logs are printed to the terminal once every 10 iterations. You can set the interval using the `interval` parameter of the [LoggerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/logger_hook.py).

```
pythonCopy code# Print logs every 20 iterations
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

For more information on how to use `LoggerHook`, please refer to the [LoggerHook tutorial](../tutorials/hook.md#loggerhook).

