# Migrate Hook from MMCV to MMEngine

## Introduction

Due to the upgrade of our architecture design and the continuous increase of user demands, existing hook mount points in MMCV can no longer meet the requirements. Hence, we redesigned the mount points in MMEngine, and the functions of hooks were adjusted accordingly. It will help a lot to read the tutorial [Hook Design](../design/hook.md) before your migration.

This tutorial compares the difference in function, mount point, usage and implementation between [MMCV v1.6.0](https://github.com/open-mmlab/mmcv/tree/v1.6.0) and [MMEngine v0.5.0](https://github.com/open-mmlab/mmengine/tree/v0.5.0).

## Function Comparison

<table class="docutils">
<thead>
  <tr>
    <th></th>
    <th>MMCV</th>
    <th>MMEngine</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">Backpropagation and gradient update</td>
    <td>OptimizerHook</td>
    <td rowspan="2">Unify the backpropagation and gradient update operations into <a href="../tutorials/optim_wrapper.html">OptimWrapper</a> rather than hooks</td>
  </tr>
  <tr>
    <td>GradientCumulativeOptimizerHook</td>
  </tr>
  <tr>
    <td>Learning rate adjustment</td>
    <td>LrUpdaterHook</td>
    <td rowspan="2">Use ParamSchdulerHook and subclasses of <a href="../tutorials/param_scheduler.html">_ParamScheduler</a> to complete the adjustment of optimizer hyperparameters</td>
  </tr>
  <tr>
    <td>Momentum adjustment</td>
    <td>MomentumUpdaterHook</td>
  </tr>
  <tr>
    <td>Saving model weights at specified interval</td>
    <td>CheckpointHook</td>
    <td rowspan="2">The CheckpointHook is responsible for not only saving weights but also saving the optimal weights. Meanwhile, the model evaluation function of EvalHook is delegated to ValLoop or TestLoop.</td>
  </tr>
  <tr>
    <td>Model evaluation and optimal weights saving</td>
    <td>EvalHook</td>
  </tr>
  <tr>
    <td>Log printing</td>
    <td rowspan="3">LoggerHook and its subclasses can print logs, save logs and visualize data</td>
    <td>LoggerHook</td>
  </tr>
  <tr>
    <td>Visualization</td>
    <td>NaiveVisualizationHook</td>
  </tr>
  <tr>
    <td>Adding runtime information</td>
    <td>RuntimeInfoHook</td>
  </tr>
  <tr>
    <td>Model weights exponential moving average (EMA)</td>
    <td>EMAHook</td>
    <td>EMAHook</td>
  </tr>
  <tr>
    <td>Ensuring that the shuffle functionality of the distributed Sampler takes effect</td>
    <td>DistSamplerSeedHook</td>
    <td>DistSamplerSeedHook</td>
  </tr>
  <tr>
    <td>Synchronizing model buffer</td>
    <td>SyncBufferHook</td>
    <td>SyncBufferHook</td>
  </tr>
  <tr>
    <td>Empty PyTorch CUDA cache</td>
    <td>EmptyCacheHook</td>
    <td>EmptyCacheHook</td>
  </tr>
  <tr>
    <td>Calculating iteration time-consuming</td>
    <td>IterTimerHook</td>
    <td>IterTimerHook</td>
  </tr>
  <tr>
    <td>Analyzing bottlenecks of training time</td>
    <td>ProfilerHook</td>
    <td>Not yet available</td>
  </tr>
  <tr>
    <td>Provide the most concise function registration</td>
    <td>ClosureHook</td>
    <td>Not yet available</td>
  </tr>
</tbody>
</table>

## Mount Point Comparison

<table class="docutils">
<thead>
  <tr>
    <th colspan="2"></th>
    <th class="tg-uzvj">MMCV</th>
    <th class="tg-uzvj">MMEngine</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">Global mount points</td>
    <td>before run</td>
    <td>before_run</td>
    <td>before_run</td>
  </tr>
  <tr>
    <td>after run</td>
    <td>after_run</td>
    <td>after_run</td>
  </tr>
  <tr>
    <td rowspan="2">Checkpoint related</td>
    <td>after loading checkpoints</td>
    <td>None</td>
    <td>after_load_checkpoint</td>
  </tr>
  <tr>
    <td>before saving checkpoints</td>
    <td>None</td>
    <td>before_save_checkpoint</td>
  </tr>
  <tr>
    <td rowspan="6">Training related</td>
    <td>triggered before training</td>
    <td>None</td>
    <td>before_train</td>
  </tr>
  <tr>
    <td>triggered after training</td>
    <td>None</td>
    <td>after_train</td>
  </tr>
  <tr>
    <td>before each epoch</td>
    <td>before_train_epoch</td>
    <td>before_train_epoch</td>
  </tr>
  <tr>
    <td>after each epoch</td>
    <td>after_train_epoch</td>
    <td>after_train_epoch</td>
  </tr>
  <tr>
    <td>before each iteration</td>
    <td>before_train_iter</td>
    <td>before_train_iter, with additional args: batch_idx and data_batch</td>
  </tr>
  <tr>
    <td>after each iteration</td>
    <td>after_train_iter</td>
    <td>after_train_iter, with additional args: batch_idx, data_batch, and outputs</td>
  </tr>
  <tr>
    <td rowspan="6">Validation related</td>
    <td>before validation</td>
    <td>None</td>
    <td>before_val</td>
  </tr>
  <tr>
    <td>after validation</td>
    <td>None</td>
    <td>after_val</td>
  </tr>
  <tr>
    <td>before each epoch</td>
    <td>before_val_epoch</td>
    <td>before_val_epoch</td>
  </tr>
  <tr>
    <td>after each epoch</td>
    <td>after_val_epoch</td>
    <td>after_val_epoch</td>
  </tr>
  <tr>
    <td>before each iteration</td>
    <td>before_val_iter</td>
    <td>before_val_iter, with additional args: batch_idx and data_batch</td>
  </tr>
  <tr>
    <td>after each iteration</td>
    <td>after_val_iter</td>
    <td>after_val_iter, with additional args: batch_idx, data_batch and outputs</td>
  </tr>
  <tr>
    <td rowspan="6">Test related</td>
    <td>before test</td>
    <td>None</td>
    <td>before_test</td>
  </tr>
  <tr>
    <td>after test</td>
    <td>None</td>
    <td>after_test</td>
  </tr>
  <tr>
    <td>before each epoch</td>
    <td>None</td>
    <td>before_test_epoch</td>
  </tr>
  <tr>
    <td>after each epoch</td>
    <td>None</td>
    <td>after_test_epoch</td>
  </tr>
  <tr>
    <td>before each iteration</td>
    <td>None</td>
    <td>before_test_iter, with additional args: batch_idx and data_batch</td>
  </tr>
  <tr>
    <td>after each iteration</td>
    <td>None</td>
    <td>after_test_iter, with additional args: batch_idx, data_batch and outputs</td>
  </tr>
</tbody>
</table>

## Usage Comparison

In MMCV, to register hooks to the runner, you need to call the Runner's `register_training_hooks` method to register hooks to the Runner. In MMEngine, you can register hooks by passing them as parameters to the Runner's initialization method.

- MMCV

```python
model = ResNet18()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
lr_config = dict(policy='step', step=[2, 3])
optimizer_config = dict(grad_clip=None)
checkpoint_config = dict(interval=5)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
runner = EpochBasedRunner(
    model=model,
    optimizer=optimizer,
    work_dir='./work_dir',
    max_epochs=3,
    xxx,
)
runner.register_training_hooks(
    lr_config=lr_config,
    optimizer_config=optimizer_config,
    checkpoint_config=checkpoint_config,
    log_config=log_config,
    custom_hooks_config=custom_hooks,
)
runner.run([trainloader], [('train', 1)])
```

- MMEngine

```python
model=ResNet18()
optim_wrapper=dict(
    type='OptimizerWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9))
param_scheduler = dict(type='MultiStepLR', milestones=[2, 3]),
default_hooks = dict(
    logger=dict(type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5),
)
custom_hooks = [dict(type='NumClassCheckHook')]
runner = Runner(
    model=model,
    work_dir='./work_dir',
    optim_wrapper=optim_wrapper,
    param_scheduler=param_scheduler,
    train_cfg=dict(by_epoch=True, max_epochs=3),
    default_hooks=default_hooks,
    custom_hooks=custom_hooks,
    xxx,
)
runner.train()
```

For more details of MMEngine hooks, please refer to [Usage of Hooks](../tutorials/hook.md).

## Implementation Comparison

Taking `CheckpointHook` as an example, compared with [CheckpointHook](https://github.com/open-mmlab/mmcv/blob/v1.6.0/mmcv/runner/hooks/checkpoint.py) in MMCV, [CheckpointHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py) of MMEngine needs to implement the `after_val_epoch` method, since new `CheckpointHook` supports saving the optimal weights, while in MMCV, the function is achieved by EvalHook.

- MMCV

```python
class CheckpointHook(Hook):
    def before_run(self, runner):
        """Initialize out_dir and file_client"""

    def after_train_epoch(self, runner):
        """Synchronize buffer and save model weights, for tasks trained in epochs"""

    def after_train_iter(self, runner):
        """Synchronize buffers and save model weights for tasks trained in iterations"""
```

- MMEngine

```python
class CheckpointHook(Hook):
    def before_run(self, runner):
        """Initialize out_dir and file_client"""

    def after_train_epoch(self, runner):
        """Synchronize buffer and save model weights, for tasks trained in epochs"""

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """Synchronize buffers and save model weights for tasks trained in iterations"""

    def after_val_epoch(self, runner, metrics):
        """Save optimal weights according to metrics"""
```
