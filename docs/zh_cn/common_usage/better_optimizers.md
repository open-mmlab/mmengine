# 性能更优的优化器

本文档提供了一些 MMEngine 支持的第三方优化器，它们可能会带来更快的收敛速度或者更高的性能。

## D-Adaptation

[D-Adaptation](https://github.com/facebookresearch/dadaptation) 提供了 `DAdaptAdaGrad`、`DAdaptAdam` 和 `DAdaptSGD` 优化器。

```{note}
如使用 D-Adaptation 提供的优化器，需将 mmengine 升级至 `0.6.0`。
```

- 安装

```bash
pip install dadaptation
```

- 使用

以使用 `DAdaptAdaGrad` 为例。

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    # 如需查看 DAdaptAdaGrad 的输入参数，可查看
    # https://github.com/facebookresearch/dadaptation/blob/main/dadaptation/dadapt_adagrad.py
    optim_wrapper=dict(optimizer=dict(type='DAdaptAdaGrad', lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
)
runner.train()
```

## Lion

[lion-pytorch](https://github.com/lucidrains/lion-pytorch) 提供了 `Lion` 优化器。

```{note}
如使用 Lion 提供的优化器，需将 mmengine 升级至 `0.6.0`。
```

- 安装

```bash
pip install lion-pytorch
```

- 使用

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    # 如需查看 Lion 的输入参数，可查看
    # https://github.com/lucidrains/lion-pytorch/blob/main/lion_pytorch/lion_pytorch.py
    optim_wrapper=dict(optimizer=dict(type='Lion', lr=1e-4, weight_decay=1e-2)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
)
runner.train()
```

## Sophia

[Sophia](https://github.com/kyegomez/Sophia) 提供了 `Sophia`、`SophiaG`、`DecoupledSophia` 和 `Sophia2` 优化器。

```{note}
如使用 Sophia 提供的优化器，需将 mmengine 升级至 `0.7.4`。
```

- 安装

```bash
pip install Sophia-Optimizer
```

- 使用

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    # 如需查看 SophiaG 的输入参数，可查看
    # https://github.com/kyegomez/Sophia/blob/main/Sophia/Sophia.py
    optim_wrapper=dict(optimizer=dict(type='SophiaG', lr=2e-4, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
)
runner.train()
```

## bitsandbytes

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes) 提供了 `AdamW8bit`、`Adam8bit`、`Adagrad8bit`、`PagedAdam8bit`、`PagedAdamW8bit`、`LAMB8bit`、 `LARS8bit`、`RMSprop8bit`、`Lion8bit`、`PagedLion8bit` 和 `SGD8bit` 优化器。

```{note}
如使用 D-Adaptation 提供的优化器，需将 mmengine 升级至 `0.9.0`。
```

- 安装

```bash
pip install bitsandbytes
```

- 使用

以 `AdamW8bit` 为例。

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    # 如需查看 AdamW8bit 的输入参数，可查看
    # https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/optim/adamw.py
    optim_wrapper=dict(optimizer=dict(type='AdamW8bit', lr=1e-4, weight_decay=1e-2)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
)
runner.train()
```
