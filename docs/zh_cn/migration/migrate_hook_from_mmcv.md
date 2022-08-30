# 迁移 MMCV 钩子到 MMEngine

## 简介

由于架构设计的更新和用户需求的不断增加，MMCV 的钩子（Hook）点位已经满足不了需求，因此在 MMEngine 中对钩子点位进行了重新设计以及对钩子的功能做了调整。在开始迁移前，阅读[钩子的设计](../design/hook.md)会很有帮助。

本文对比 [MMCV v1.6.0](https://github.com/open-mmlab/mmcv/tree/v1.6.0) 和 [MMEngine v0.5.0](https://github.com/open-mmlab/mmengine/tree/v0.5.0) 的钩子在功能、点位、用法和实现上的差异。

## 功能差异

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
    <td rowspan="2">反向传播以及梯度更新</td>
    <td>OptimizerHook</td>
    <td rowspan="2">将反向传播以及梯度更新的操作抽象成 <a href="../tutorials/optim_wrapper.html">OptimWrapper</a> 而不是钩子</td>
  </tr>
  <tr>
    <td>GradientCumulativeOptimizerHook</td>
  </tr>
  <tr>
    <td>学习率调整</td>
    <td>LrUpdaterHook</td>
    <td rowspan="2">ParamSchdulerHook 以及 <a href="../tutorials/param_scheduler.html">_ParamScheduler</a> 的子类完成优化器超参的调整</td>
  </tr>
  <tr>
    <td>动量调整</td>
    <td>MomentumUpdaterHook</td>
  </tr>
  <tr>
    <td>按指定间隔保存权重</td>
    <td>CheckpointHook</td>
    <td rowspan="2">CheckpointHook 除了保存权重，还有保存最优权重的功能，而 EvalHook 的模型评估功能则交由 ValLoop 或 TestLoop 完成</td>
  </tr>
  <tr>
    <td>模型评估并保存最优模型</td>
    <td>EvalHook</td>
  </tr>
  <tr>
    <td>打印日志</td>
    <td rowspan="3">LoggerHook 及其子类实现打印日志、保存日志以及可视化功能</td>
    <td>LoggerHook</td>
  </tr>
  <tr>
    <td>可视化</td>
    <td>NaiveVisualizationHook</td>
  </tr>
  <tr>
    <td>添加运行时信息</td>
    <td>RuntimeInfoHook</td>
  </tr>
  <tr>
    <td>模型参数指数滑动平均</td>
    <td>EMAHook</td>
    <td>EMAHook</td>
  </tr>
  <tr>
    <td>确保分布式 Sampler 的 shuffle 生效</td>
    <td>DistSamplerSeedHook</td>
    <td>DistSamplerSeedHook</td>
  </tr>
  <tr>
    <td>同步模型的 buffer</td>
    <td>SyncBufferHook</td>
    <td>SyncBufferHook</td>
  </tr>
  <tr>
    <td>PyTorch CUDA 缓存清理</td>
    <td>EmptyCacheHook</td>
    <td>EmptyCacheHook</td>
  </tr>
  <tr>
    <td>统计迭代耗时</td>
    <td>IterTimerHook</td>
    <td>IterTimerHook</td>
  </tr>
  <tr>
    <td>分析训练时间的瓶颈</td>
    <td>ProfilerHook</td>
    <td>暂未提供</td>
  </tr>
  <tr>
    <td>提供注册方法给钩子点位的功能</td>
    <td>ClosureHook</td>
    <td>暂未提供</td>
  </tr>
</tbody>
</table>

## 点位差异

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
    <td rowspan="2">全局位点</td>
    <td>执行前</td>
    <td>before_run</td>
    <td>before_run</td>
  </tr>
  <tr>
    <td>执行后</td>
    <td>after_run</td>
    <td>after_run</td>
  </tr>
  <tr>
    <td rowspan="2">Checkpoint 相关</td>
    <td>加载 checkpoint 后</td>
    <td>无</td>
    <td>after_load_checkpoint</td>
  </tr>
  <tr>
    <td>保存 checkpoint 前</td>
    <td>无</td>
    <td>before_save_checkpoint</td>
  </tr>
  <tr>
    <td rowspan="6">训练相关</td>
    <td>训练前触发</td>
    <td>无</td>
    <td>before_train</td>
  </tr>
  <tr>
    <td>训练后触发</td>
    <td>无</td>
    <td>after_train</td>
  </tr>
  <tr>
    <td>每个 epoch 前</td>
    <td>before_train_epoch</td>
    <td>before_train_epoch</td>
  </tr>
  <tr>
    <td>每个 epoch 后</td>
    <td>after_train_epoch</td>
    <td>after_train_epoch</td>
  </tr>
  <tr>
    <td>每次迭代前</td>
    <td>before_train_iter</td>
    <td>before_train_iter，新增 batch_idx 和 data_batch 参数</td>
  </tr>
  <tr>
    <td>每次迭代后</td>
    <td>after_train_iter</td>
    <td>after_train_iter，新增 batch_idx、data_batch 和 outputs 参数</td>
  </tr>
  <tr>
    <td rowspan="6">验证相关</td>
    <td>验证前触发</td>
    <td>无</td>
    <td>before_val</td>
  </tr>
  <tr>
    <td>验证后触发</td>
    <td>无</td>
    <td>after_val</td>
  </tr>
  <tr>
    <td>每个 epoch 前</td>
    <td>before_val_epoch</td>
    <td>before_val_epoch</td>
  </tr>
  <tr>
    <td>每个 epoch 后</td>
    <td>after_val_epoch</td>
    <td>after_val_epoch</td>
  </tr>
  <tr>
    <td>每次迭代前</td>
    <td>before_val_iter</td>
    <td>before_val_iter，新增 batch_idx 和 data_batch 参数</td>
  </tr>
  <tr>
    <td>每次迭代后</td>
    <td>after_val_iter</td>
    <td>after_val_iter，新增 batch_idx、data_batch 和 outputs 参数</td>
  </tr>
  <tr>
    <td rowspan="6">测试相关</td>
    <td>测试前触发</td>
    <td>无</td>
    <td>before_test</td>
  </tr>
  <tr>
    <td>测试后触发</td>
    <td>无</td>
    <td>after_test</td>
  </tr>
  <tr>
    <td>每个 epoch 前</td>
    <td>无</td>
    <td>before_test_epoch</td>
  </tr>
  <tr>
    <td>每个 epoch 后</td>
    <td>无</td>
    <td>after_test_epoch</td>
  </tr>
  <tr>
    <td>每次迭代前</td>
    <td>无</td>
    <td>before_test_iter，新增 batch_idx 和 data_batch 参数</td>
  </tr>
  <tr>
    <td>每次迭代后</td>
    <td>无</td>
    <td>after_test_iter，新增 batch_idx、data_batch 和 outputs 参数</td>
  </tr>
</tbody>
</table>

## 用法差异

在 MMCV 中，将钩子注册到执行器（Runner），需调用执行器的 `register_training_hooks` 方法往执行器注册钩子，而在 MMEngine 中，可以通过参数传递给执行器的初始化方法进行注册。

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

MMEngine 钩子的更多用法请参考[钩子的用法](../tutorials/hook.md)。

## 实现差异

以 `CheckpointHook` 为例，MMEngine 的 [CheckpointHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py) 相比 MMCV 的 [CheckpointHook](https://github.com/open-mmlab/mmcv/blob/v1.6.0/mmcv/runner/hooks/checkpoint.py)（新增保存最优权重的功能，在 MMCV 中，保存最优权重的功能由 EvalHook 提供），因此，它需要实现 `after_val_epoch` 点位。

- MMCV

```python
class CheckpointHook(Hook):
    def before_run(self, runner):
        """初始化 out_dir 和 file_client 属性"""

    def after_train_epoch(self, runner):
        """同步 buffer 和保存权重，用于以 epoch 为单位训练的任务"""

    def after_train_iter(self, runner):
        """同步 buffer 和保存权重，用于以 iteration 为单位训练的任务"""
```

- MMEngine

```python
class CheckpointHook(Hook):
    def before_run(self, runner):
        """初始化 out_dir 和 file_client 属性"""

    def after_train_epoch(self, runner):
        """同步 buffer 和保存权重，用于以 epoch 为单位训练的任务"""

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """同步 buffer 和保存权重，用于以 iteration 为单位训练的任务"""

    def after_val_epoch(self, runner, metrics):
        """根据 metrics 保存最优权重"""
```
