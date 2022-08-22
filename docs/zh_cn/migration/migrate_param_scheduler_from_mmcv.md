# 迁移 MMCV 参数调度器到 MMEngine

MMCV 1.x 版本使用 LrUpdaterHook 和 MomentumUpdaterHook 来调整学习率和动量。
但随着深度学习算法训练方式的不断发展，使用 Hook 修改学习率的方式已无法满足需求，因此 MMEngine 提供了参数调度器（ParamScheduler）。
一方面，参数调度器的接口与 PyTroch 的学习率调度器（LRScheduler）对齐，另一方面，参数调度器提供了更丰富的功能，详细请参考[参数调度器使用指南](../tutorials/param_scheduler.md)

## 迁移学习率调度器（LrUpdater）

MMEngine 中使用 LRScheduler 替代 LrUpdaterHook，配置文件中的字段从原本的 `lr_config` 修改为 `param_scheduler`。
MMCV 中的学习率配置与 MMEngine 中的参数调度器配置对应关系如下：

### 迁移学习率预热（Warmup）

MMCV 中有 3 种学习率预热方式，分别是 'constant', 'linear', 'exp'，在 MMEngine 中对应的配置应修改为:

- 'constant' 预热

  MMCV:

  ```python
  lr_config = dict(warmup='constant', warmup_ratio=0.1, warmup_iters=500, warmup_by_epoch=False)
  ```

  MMEngine:

  ```python
  param_scheduler = [
      dict(type='ConstantLR', factor=0.1, begin=0, end=500, by_epoch=False)
  ]
  ```

- 'linear' 预热

  MMCV:

  ```python
  lr_config = dict(warmup='linear', warmup_ratio=0.1, warmup_iters=500, warmup_by_epoch=False)
  ```

  MMEngine:

  ```python
  param_scheduler = [
      dict(type='LinearLR', start_factor=0.1, begin=0, end=500, by_epoch=False)
  ]
  ```

- 'exp' 预热

  MMCV:

  ```python
  lr_config = dict(warmup='exp', warmup_ratio=0.1, warmup_iters=500, warmup_by_epoch=False)
  ```

  MMEngine:

  ```python
  param_scheduler = [
      dict(type='ExponentialLR', gamma=0.1, begin=0, end=500, by_epoch=False)
  ]
  ```

### 迁移 'fixed' 学习率（FixedLrUpdaterHook）

MMCV:

```python
lr_config = dict(policy='fixed')
```

MMEngine:

```python
param_scheduler = [
    dict(type='ConstantLR', factor=1)
]
```

### 迁移 'step' 学习率（StepLrUpdaterHook）

MMCV:

```python
lr_config = dict(policy='step', step=[8, 11], gamma=0.1, by_epoch=True)
```

MMEngine:

```python
param_scheduler = [
    dict(type='MultiStepLR', milestone=[8, 11], gamma=0.1, by_epoch=True)
]
```

### 迁移 'poly' 学习率（PolyLrUpdaterHook）

MMCV:

```python
lr_config = dict(policy='poly', power=0.7, min_lr=0.001, by_epoch=True)
```

MMEngine:

```python
param_scheduler = [
    dict(type='PolyLR', power=0.7, eta_min=0.001, begin=0, end=num_epochs, by_epoch=True)
]
```

### 迁移 'exp' 学习率（ExpLrUpdaterHook）

MMCV:

```python
lr_config = dict(policy='exp', power=0.5, by_epoch=True)
```

MMEngine:

```python
param_scheduler = [
    dict(type='ExponentialLR', gamma=0.5, begin=0, end=num_epochs, by_epoch=True)
]
```

### 迁移 'CosineAnnealing' 学习率（CosineAnnealingLrUpdaterHook）

MMCV:

```python
lr_config = dict(policy='CosineAnnealing', min_lr=0.5, by_epoch=True)
```

MMEngine:

```python
param_scheduler = [
    dict(type='CosineAnnealingLR', eta_min=0.5, T_max=num_epochs, begin=0, end=num_epochs, by_epoch=True)
]
```

如果在使用 epoch-based 训练循环且配置文件中依旧按 epoch 设置周期时希望学习率按 iteration 更新，在 MMCV 中需要将 `by_epoch` 设置为 False, 而在 MMEngine 中需要注意， 配置中的 `by_epoch` 仍需设置为 True，通过在配置中添加 `convert_to_iter_based=True` 来构建按 iteration 更新的参数调度器：

MMCV:

```python
lr_config = dict(policy='CosineAnnealing', min_lr=0.5, by_epoch=False)
```

MMEngine:

```python
param_scheduler = [
    dict(type='CosineAnnealingLR',
         eta_min=0.5,
         T_max=num_epochs,
         by_epoch=True,
         convert_to_iter_based=True)
]
```

### 迁移 'FlatCosineAnnealing' 学习率（FlatCosineAnnealingLrUpdaterHook）

像 FlatCosineAnnealing 这种由多个学习率策略拼接而成的学习率，原本需要重写 Hook 来实现，而在 MMEngine 中只需将两个参数调度器组合即可

MMCV:

```python
lr_config = dict(policy='FlatCosineAnnealing', start_percent=0.5, min_lr=0.005, by_epoch=True)
```

MMEngine:

```python
param_scheduler = [
    dict(type='ConstantLR', factor=1, begin=0, end=num_epochs * 0.75)
    dict(type='CosineAnnealingLR',
         eta_min=0.005,
         begin=num_epochs * 0.75,
         end=num_epochs,
         T_max=num_epochs * 0.25,
         by_epoch=True)
]
```

### 迁移 'CosineRestart' 学习率（CosineRestartLrUpdaterHook）

MMCV:

```python
lr_config = dict(policy='CosineRestart',
                 periods=[5, 10, 15],
                 restart_weights=[1, 0.7, 0.3],
                 min_lr=0.001,
                 by_epoch=True)
```

MMEngine:

```python
param_scheduler = [
    dict(type='CosineRestartLR',
         periods=[5, 10, 15],
         restart_weights=[1, 0.7, 0.3],
         eta_min=0.001,
         by_epoch=True)
]
```

### 迁移 'OneCycle' 学习率（OneCycleLrUpdaterHook）

MMCV:

```python
lr_config = dict(policy='OneCycle',
                 eta_max=0.02,
                 total_steps=90000,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 div_factor=25,
                 final_div_factor=1e4,
                 three_phase=True,
                 by_epoch=False)
```

MMEngine:

```python
param_scheduler = [
    dict(type='OneCycleLR',
         eta_max=0.02,
         total_steps=90000,
         pct_start=0.3,
         anneal_strategy='cos',
         div_factor=25,
         final_div_factor=1e4,
         three_phase=True,
         by_epoch=False)
]
```

### 迁移 'LinearAnnealing' 学习率（LinearAnnealingLrUpdaterHook）

MMCV:

```python
lr_config = dict(policy='LinearAnnealing', min_lr_ratio=0.01, by_epoch=True)
```

MMEngine:

```python
param_scheduler = [
    dict(type='LinearLR', start_factor=1, end_factor=0.01, begin=0, end=num_epochs, by_epoch=True)
]
```

## 迁移动量调度器（MomentumUpdater）

MMCV 使用 `momentum_config` 字段和 MomentumUpdateHook 调整动量。 MMEngine 中动量同样由参数调度器控制。用户可以简单将学习率调度器后的`LR`修改为`Momentum`，即可使用同样的策略来调整动量。动量调度器只需要和学习率调度器一样添加进`param_scheduler`列表中即可。举一个简单的例子：

MMCV:

```python
lr_config = dict(...)
momentum_config = dict(policy='CosineAnnealing', min_momentum=0.1, by_epoch=True)
```

MMEngine:

```python
param_scheduler = [
    # 学习率调度器配置
    dict(...),
    # 动量调度器配置
    dict(type='CosineAnnealingMomentum', eta_min=0.1, T_max=num_epochs, begin=0, end=num_epochs, by_epoch=True)
]
```

你可能还想阅读[参数调度器的教程](../tutorials/param_scheduler.md)或者[参数调度器的 API 文档](mmengine.optim.scheduler)。
