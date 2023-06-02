# Better performance optimizers

This document provides some third-party optimizers supported by MMEngine, which may bring faster convergence speed or higher performance.

## D-Adaptation

[D-Adaptation](https://github.com/facebookresearch/dadaptation) provides `DAdaptAdaGrad`, `DAdaptAdam` and `DAdaptSGD` optimziers。

```{note}
If you use the optimizer provided by D-Adaptation, you need to upgrade mmengine to `0.6.0`.
```

- Installation

```bash
pip install dadaptation
```

- Usage

Take the `DAdaptAdaGrad` as an example.

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    # To view the input parameters for DAdaptAdaGrad, you can refer to
    # https://github.com/facebookresearch/dadaptation/blob/main/dadaptation/dadapt_adagrad.py
    optim_wrapper=dict(optimizer=dict(type='DAdaptAdaGrad', lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
)
runner.train()
```

## Lion-Pytorch

[lion-pytorch](https://github.com/lucidrains/lion-pytorch) provides the `Lion` optimizer。

```{note}
If you use the optimizer provided by Lion-Pytorch, you need to upgrade mmengine to `0.6.0`.
```

- Installation

```bash
pip install lion-pytorch
```

- Usage

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    # To view the input parameters for Lion, you can refer to
    # https://github.com/lucidrains/lion-pytorch/blob/main/lion_pytorch/lion_pytorch.py
    optim_wrapper=dict(optimizer=dict(type='Lion', lr=1e-4, weight_decay=1e-2)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
)
runner.train()
```

## Sophia

[Sophia](https://github.com/kyegomez/Sophia) provides `Sophia`, `SophiaG`, `DecoupledSophia` and `Sophia2` optimizers.

```{note}
If you use the optimizer provided by Sophia, you need to upgrade mmengine to `0.7.4`.
```

- Installation

```bash
pip install Sophia-Optimizer
```

- Usage

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    # To view the input parameters for SophiaG, you can refer to
    # https://github.com/kyegomez/Sophia/blob/main/Sophia/Sophia.py
    optim_wrapper=dict(optimizer=dict(type='SophiaG', lr=2e-4, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
)
runner.train()
```
