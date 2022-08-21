# Resume training

Resuming training means continuing training from the state saved from some previous training, where the state includes the model's weights, the state of the optimizer and the state of parameter scheduler.

The following examples will be based on the Runner object initialized as follows

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

## Automatically resume training

Suppose the training is interrupted at epoch 3, you can set the `resume` parameter of `Runner` to enable resume training.
Set `resume` of `Runner` equal to `True` and `Runner` will load the latest checkpoint from `work_dir`.

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

## Specify the checkpoint path

If you want to specify the path to resume training, in addition to setting `resume=True`, you also need to set the `load_from` parameter.

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
