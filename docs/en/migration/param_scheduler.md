# Migrate parameter scheduler from MMCV to MMEngine

MMCV 1.x version uses [LrUpdaterHook](https://mmcv.readthedocs.io/en/v1.6.0/api.html#mmcv.runner.LrUpdaterHook) and [MomentumUpdaterHook](https://mmcv.readthedocs.io/en/v1.6.0/api.html#mmcv.runner.MomentumUpdaterHook) to adjust the learning rate and momentum.
However, the design of LrUpdaterHook has been difficult to meet more abundant customization requirements due to the development of the training strategies. Hence, MMEngine proposes parameter schedulers (ParamScheduler).

The interface of the parameter scheduler is consistent with PyTroch's learning rate scheduler (LRScheduler). In addition, the parameter scheduler provides stronger functions. For details, please refer to [Parameter Scheduler User Guide](../tutorials/param_scheduler.md).

## Learning rate scheduler (LrUpdater) migration

MMEngine uses LRScheduler instead of LrUpdaterHook. The field in the config file is changed from the original `lr_config` to `param_scheduler`.
The learning rate config in MMCV corresponds to the parameter scheduler config in MMEngine as follows:

### Learning rate warm-up migration

The learning rate warm-up can be achieved through the combination of schedulers by specifying the effective range `begin` and `end`. There are 3 learning rate warm-up methods in MMCV, namely `'constant'`, `'linear'`, `'exp'`. The corresponding config in MMEngine should be modified as follows:

#### Constant warm-up

<table class="docutils">
  <thead>
  <tr>
      <th>MMCV-1.x</th>
      <th>MMEngine</th>
  <tbody>
  <tr>
  <td>

```python
lr_config = dict(
    warmup='constant',
    warmup_ratio=0.1,
    warmup_iters=500,
    warmup_by_epoch=False
)
```

</td>
  <td>

```python
param_scheduler = [
    dict(type='ConstantLR',
         factor=0.1,
         begin=0,
         end=500,
         by_epoch=False),
    dict(...) # the main learning rate scheduler
]
```

</td>
  </tr>
  </thead>
  </table>

#### Linear warm-up

<table class="docutils">
  <thead>
  <tr>
      <th>MMCV-1.x</th>
      <th>MMEngine</th>
  <tbody>
  <tr>
  <td>

```python
lr_config = dict(
    warmup='linear',
    warmup_ratio=0.1,
    warmup_iters=500,
    warmup_by_epoch=False
)
```

</td>
  <td>

```python
param_scheduler = [
    dict(type='LinearLR',
         start_factor=0.1,
         begin=0,
         end=500,
         by_epoch=False),
    dict(...) # the main learning rate scheduler
]
```

</td>
  </tr>
  </thead>
  </table>

#### Exponential warm-up

<table class="docutils">
  <thead>
  <tr>
      <th>MMCV-1.x</th>
      <th>MMEngine</th>
  <tbody>
  <tr>
  <td>

```python
lr_config = dict(
    warmup='exp',
    warmup_ratio=0.1,
    warmup_iters=500,
    warmup_by_epoch=False
)
```

</td>
  <td>

```python
param_scheduler = [
    dict(type='ExponentialLR',
         gamma=0.1,
         begin=0,
         end=500,
         by_epoch=False),
    dict(...) # the main learning rate scheduler
]
```

</td>
  </tr>
  </thead>
  </table>

### Fixed learning rate (FixedLrUpdaterHook) migration

<table class="docutils">
<thead>
<tr>
    <th>MMCV-1.x</th>
    <th>MMEngine</th>
<tbody>
<tr>
<td>

```python
lr_config = dict(policy='fixed')
```

</td>
<td>

```python
param_scheduler = [
    dict(type='ConstantLR', factor=1)
]
```

</td>
</tr>
</thead>
</table>

### Step learning rate (StepLrUpdaterHook) migration

<table class="docutils">
<thead>
<tr>
    <th>MMCV-1.x</th>
    <th>MMEngine</th>
<tbody>
<tr>
<td>

```python
lr_config = dict(
    policy='step',
    step=[8, 11],
    gamma=0.1,
    by_epoch=True
)
```

</td>
<td>

```python
param_scheduler = [
    dict(type='MultiStepLR',
         milestones=[8, 11],
         gamma=0.1,
         by_epoch=True)
]
```

</td>
</tr>
</thead>
</table>

### Poly learning rate (PolyLrUpdaterHook) migration

<table class="docutils">
<thead>
<tr>
    <th>MMCV-1.x</th>
    <th>MMEngine</th>
<tbody>
<tr>
<td>

```python
lr_config = dict(
    policy='poly',
    power=0.7,
    min_lr=0.001,
    by_epoch=True
)
```

</td>
<td>

```python
param_scheduler = [
    dict(type='PolyLR',
         power=0.7,
         eta_min=0.001,
         begin=0,
         end=num_epochs,
         by_epoch=True)
]
```

</td>
</tr>
</thead>
</table>

### Exponential learning rate (ExpLrUpdaterHook) migration

<table class="docutils">
<thead>
<tr>
    <th>MMCV-1.x</th>
    <th>MMEngine</th>
<tbody>
<tr>
<td>

```python
lr_config = dict(
    policy='exp',
    power=0.5,
    by_epoch=True
)
```

</td>
<td>

```python
param_scheduler = [
    dict(type='ExponentialLR',
         gamma=0.5,
         begin=0,
         end=num_epochs,
         by_epoch=True)
]
```

</td>
</tr>
</thead>
</table>

### Cosine annealing learning rate (CosineAnnealingLrUpdaterHook) migration

<table class="docutils">
<thead>
<tr>
    <th>MMCV-1.x</th>
    <th>MMEngine</th>
<tbody>
<tr>
<td>

```python
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.5,
    by_epoch=True
)
```

</td>
<td>

```python
param_scheduler = [
    dict(type='CosineAnnealingLR',
         eta_min=0.5,
         T_max=num_epochs,
         begin=0,
         end=num_epochs,
         by_epoch=True)
]
```

</td>
</tr>
</thead>
</table>

### FlatCosineAnnealingLrUpdaterHook migration

The learning rate strategy combined by multiple phases like FlatCosineAnnealing originally needs to be achieved by rewriting a Hook. But in MMEngine, it can be achieved with combining two parameter scheduler configs:

<table class="docutils">
<thead>
<tr>
    <th>MMCV-1.x</th>
    <th>MMEngine</th>
<tbody>
<tr>
<td>

```python
lr_config = dict(
    policy='FlatCosineAnnealing',
    start_percent=0.5,
    min_lr=0.005,
    by_epoch=True
)
```

</td>
<td>

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

</td>
</tr>
</thead>
</table>

### CosineRestartLrUpdaterHook migration

<table class="docutils">
<thead>
<tr>
    <th>MMCV-1.x</th>
    <th>MMEngine</th>
<tbody>
<tr>
<td>

```python
lr_config = dict(policy='CosineRestart',
                 periods=[5, 10, 15],
                 restart_weights=[1, 0.7, 0.3],
                 min_lr=0.001,
                 by_epoch=True)
```

</td>
<td>

```python
param_scheduler = [
    dict(type='CosineRestartLR',
         periods=[5, 10, 15],
         restart_weights=[1, 0.7, 0.3],
         eta_min=0.001,
         by_epoch=True)
]
```

</td>
</tr>
</thead>
</table>

### OneCycleLrUpdaterHook migration

<table class="docutils">
<thead>
<tr>
    <th>MMCV-1.x</th>
    <th>MMEngine</th>
<tbody>
<tr>
<td>

```python
lr_config = dict(policy='OneCycle',
                 max_lr=0.02,
                 total_steps=90000,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 div_factor=25,
                 final_div_factor=1e4,
                 three_phase=True,
                 by_epoch=False)
```

</td>
<td>

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

</td>
</tr>
</thead>
</table>

Notice:  `by_epoch` defaults to `False` in MMCV. It now defaults to `True` in MMEngine.

### LinearAnnealingLrUpdaterHook migration

<table class="docutils">
<thead>
<tr>
    <th>MMCV-1.x</th>
    <th>MMEngine</th>
<tbody>
<tr>
<td>

```python
lr_config = dict(
    policy='LinearAnnealing',
    min_lr_ratio=0.01,
    by_epoch=True
)
```

</td>
<td>

```python
param_scheduler = [
    dict(type='LinearLR',
         start_factor=1,
         end_factor=0.01,
         begin=0,
         end=num_epochs,
         by_epoch=True)
]
```

</td>
</tr>
</thead>
</table>

## MomentumUpdater migration

MMCV uses `momentum_config` field and MomentumUpdateHook to adjust momentum. The momentum in MMEngine is also controlled by the parameter scheduler. Users can simply change the `LR` of the learning rate scheduler to `Momentum` to use the same strategy to adjust the momentum. The momentum scheduler shares the same `param_scheduler` field in the config with the learning rate scheduler:

<table class="docutils">
<thead>
<tr>
    <th>MMCV-1.x</th>
    <th>MMEngine</th>
<tbody>
<tr>
<td>

```python
lr_config = dict(...)
momentum_config = dict(
    policy='CosineAnnealing',
    min_momentum=0.1,
    by_epoch=True
)
```

</td>
<td>

```python
param_scheduler = [
    # config of learning rate schedulers
    dict(...),
    # config of momentum schedulers
    dict(type='CosineAnnealingMomentum',
         eta_min=0.1,
         T_max=num_epochs,
         begin=0,
         end=num_epochs,
         by_epoch=True)
]
```

</td>
</tr>
</thead>
</table>

## Migrate parameter update frequency related config

If you want to update the parameter rate based on iteration while using the epoch-based training loop and setting the effective range (`begin`, `end`) or period (`T_max`) and other variables according to epoch in MMCV, you need to set `by_epoch` to False.

However, in MMEngine, the `by_epoch` in the config still needs to be set to True. Instead, you need to add `convert_to_iter_based=True` in the config to build a parameter scheduler which updates by iteration, see [Parameter Scheduler Tutorial](../tutorials/param_scheduler.md) for more details.

Take the migration of CosineAnnealing as an example:

<table class="docutils">
<thead>
<tr>
    <th>MMCV-1.x</th>
    <th>MMEngine</th>
<tbody>
<tr>
<td>

```python
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.5,
    by_epoch=False
)
```

</td>
<td>

```python
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0.5,
        T_max=num_epochs,
        by_epoch=True,  # Notice, by_epoch need to be set to True
        convert_to_iter_based=True  # convert to an iter-based scheduler
    )
]
```

</td>
</tr>
</thead>
</table>

You may also want to read [parameter scheduler tutorial](../tutorials/param_scheduler.md) or [parameter scheduler API documentations](../api/optim).
