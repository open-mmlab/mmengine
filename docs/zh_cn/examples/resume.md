## 恢复训练

恢复训练是指从上次训练被中断的状态接着训练，这里的状态包括模型的权重、优化器和优化器参数调整策略的状态。

下面是实例化一个 Runner 的示例

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
)
runner.train()
```

假设训练在第 3 个 epoch 被中断，可以设置 `Runner` 的 `resume` 参数开启恢复训练功能。

### 自动恢复训练

设置 `Runner` 的 `resume` 等于 `True`，`Runner` 会从 `work_dir` 加载最新的 checkpoint。

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
    resume=True,
)
runner.train()
```

### 指定 checkpoint 路径

如果希望指定恢复训练的路径，除了设置 `resume=True`，还需要设置 `load_from` 参数

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
    load_from='./work_dir/epoch_2.pth'
    resume=True,
)
runner.train()
```
