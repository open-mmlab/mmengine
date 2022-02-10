# Optimization

## How to adjust learning rate

Like the `torch.optim.lr_scheduler` in PyTorch, MMEngine provides a variety of parameter schedulers to adjust the learning rate in the optimizer's parameter groups. Through the schedulers in MMEngine, you can adjust the learning rate according to epoch or iteration, you can also combine multiple schedulers, or customize your own schedulers.

### Use a single learning rate scheduler

To use a single learning rate scheduler in MMEngine, you can set the `scheduler` field in the configuration file. For example:

```python
scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1)
```

### Combine multiple learning rate schedulers (such as warmup)

MMEngine supports stacking multiple schedulers, just set the `scheduler` field in the configuration file to a list of schedulers. Take the most common strategy, warmup, as an example:

```python
scheduler = [
    # linear lr warm-up
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=False,  # update lr by iteration
         end=500),  # warm-up 500 iterations

    # the main lr scheduler
    dict(type='MultiStepLR',
         by_epoch=True,  # update lr by epoch
         milestones=[8, 11],
         gamma=0.1)
]
```

Just like the lr schedulers in PyTorch, most schedulers can be called back-to-back. The result is that each scheduler is applied one after the other on the learning rate obtained by the one preceding it.

### Update frequency

MMEngine supports two different frequencies of learning rate updates: epoch-based updates and iteration-based updates. It can be switched by the `by_epoch` parameter in the scheduler.

Update by epoch:

```python
scheduler = dict(type='MultiStepLR',
                 by_epoch=True,  # update lr by epoch
                 milestones=[8, 11],  # decay at 8 and 11 epoch
                 gamma=0.1)
```

Update by iteration:

```python
scheduler = dict(type='MultiStepLR',
                 by_epoch=False,  # update lr by iteration
                 milestones=[60000, 80000],  # decay at 60000 and 80000 iteration
                 gamma=0.1)
```

### Effective interval

You can set the effective interval of the schedulers in MMEngine. Through the two parameters of `begin` and `end`, you can control the interval in which the scheduler is applied.

Example:

```python
scheduler = [
    # Use linear lr at [0, 1000) iterations.
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=1000),
    # Use cosine annealing lr at [1000, 9000) iterations.
    dict(type='CosineAnnealingLR',
         T_max=8000,
         by_epoch=False,
         begin=1000,
         end=9000)
]
```

## How to adjust momentum

Like the learning rate, momentum is also a set of parameters in the optimizer that can be scheduled. The momentum scheduler is used in the same way as the learning rate scheduler. You only need to add the corresponding configuration under the `scheduler` field in the configuration file.

Example:

```python
scheduler = [
    dict(type='LinearMomentum',
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=1000)
]
```

## How to adjust custom parameters

MMEngine provides a set of general parameter schedulers for scheduling different parameter in optimizer's `param_groups`. You can set the `param_name` parameter of the scheduler to control what parameters you want to schedule.

Example:

```python
scheduler = [
    dict(type='LinearScheduler',
         param_name='lr'  # schedule the `lr` parameter in `optimizer.param_groups`
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=1000)
]
```
