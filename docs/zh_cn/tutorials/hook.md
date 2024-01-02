# 钩子（Hook）

钩子编程是一种编程模式，是指在程序的一个或者多个位置设置位点（挂载点），当程序运行至某个位点时，会自动调用运行时注册到位点的所有方法。钩子编程可以提高程序的灵活性和拓展性，用户将自定义的方法注册到位点便可被调用而无需修改程序中的代码。

## 内置钩子

MMEngine 提供了很多内置的钩子，将钩子分为两类，分别是默认钩子以及自定义钩子，前者表示会默认往执行器注册，后者表示需要用户自己注册。

每个钩子都有对应的优先级，在同一位点，钩子的优先级越高，越早被执行器调用，如果优先级一样，被调用的顺序和钩子注册的顺序一致。优先级列表如下：

- HIGHEST (0)
- VERY_HIGH (10)
- HIGH (30)
- ABOVE_NORMAL (40)
- NORMAL (50)
- BELOW_NORMAL (60)
- LOW (70)
- VERY_LOW (90)
- LOWEST (100)

**默认钩子**

|                    名称                     |                用途                |      优先级       |
| :-----------------------------------------: | :--------------------------------: | :---------------: |
|     [RuntimeInfoHook](#runtimeinfohook)     |   往 message hub 更新运行时信息    |  VERY_HIGH (10)   |
|       [IterTimerHook](#itertimerhook)       |            统计迭代耗时            |    NORMAL (50)    |
| [DistSamplerSeedHook](#distsamplerseedhook) | 确保分布式 Sampler 的 shuffle 生效 |    NORMAL (50)    |
|          [LoggerHook](#loggerhook)          |              打印日志              | BELOW_NORMAL (60) |
|  [ParamSchedulerHook](#paramschedulerhook)  |  调用 ParamScheduler 的 step 方法  |     LOW (70)      |
|      [CheckpointHook](#checkpointhook)      |         按指定间隔保存权重         |   VERY_LOW (90)   |

**自定义钩子**

|                名称                 |                用途                |    优先级     |
| :---------------------------------: | :--------------------------------: | :-----------: |
|         [EMAHook](#emahook)         |        模型参数指数滑动平均        |  NORMAL (50)  |
|  [EmptyCacheHook](#emptycachehook)  |       PyTorch CUDA 缓存清理        |  NORMAL (50)  |
| [SyncBuffersHook](#syncbuffershook) |         同步模型的 buffer          |  NORMAL (50)  |
|    [ProfilerHook](#profilerhook)    | 分析算子的执行时间以及显存占用情况 | VERY_LOW (90) |

```{note}
不建议修改默认钩子的优先级，因为优先级低的钩子可能会依赖优先级高的钩子。例如 CheckpointHook 的优先级需要比 ParamSchedulerHook 低，这样保存的优化器状态才是正确的状态。另外，自定义钩子的优先级默认为 `NORMAL (50)`。
```

两种钩子在执行器中的设置不同，默认钩子的配置传给执行器的 `default_hooks` 参数，自定义钩子的配置传给 `custom_hooks` 参数，如下所示：

```python
from mmengine.runner import Runner

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    logger=dict(type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
)

custom_hooks = [dict(type='EmptyCacheHook')]

runner = Runner(default_hooks=default_hooks, custom_hooks=custom_hooks, ...)
runner.train()
```

下面逐一介绍 MMEngine 中内置钩子的用法。

### CheckpointHook

[CheckpointHook](mmengine.hooks.CheckpointHook) 按照给定间隔保存模型的权重，如果是分布式多卡训练，则只有主（master）进程会保存权重。`CheckpointHook` 的主要功能如下：

- 按照间隔保存权重，支持按 epoch 数或者 iteration 数保存权重
- 保存最新的多个权重
- 保存最优权重
- 指定保存权重的路径
- 制作发布用的权重
- 设置开始保存权重的 epoch 数或者 iteration 数

如需了解其他功能，请阅读 [CheckpointHook API 文档](mmengine.hooks.CheckpointHook)。

下面介绍上面提到的 6 个功能。

- 按照间隔保存权重，支持按 epoch 数或者 iteration 数保存权重

  假设我们一共训练 20 个 epoch 并希望每隔 5 个 epoch 保存一次权重，下面的配置即可帮我们实现该需求。

  ```python
  # by_epoch 的默认值为 True
  default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5, by_epoch=True))
  ```

  如果想以迭代次数作为保存间隔，则可以将 `by_epoch` 设为 False，`interval=5` 则表示每迭代 5 次保存一次权重。

  ```python
  default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5, by_epoch=False))
  ```

- 保存最新的多个权重

  如果只想保存一定数量的权重，可以通过设置 `max_keep_ckpts` 参数实现最多保存 `max_keep_ckpts` 个权重，当保存的权重数超过 `max_keep_ckpts` 时，前面的权重会被删除。

  ```python
  default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=2))
  ```

  上述例子表示，假如一共训练 20 个 epoch，那么会在第 5, 10, 15, 20 个 epoch 保存模型，但是在第 15 个 epoch 的时候会删除第 5 个 epoch 保存的权重，在第 20 个 epoch 的时候会删除第 10 个 epoch 的权重，最终只有第 15 和第 20 个 epoch 的权重才会被保存。

- 保存最优权重

  如果想要保存训练过程验证集的最优权重，可以设置 `save_best` 参数，如果设置为 `'auto'`，则会根据验证集的第一个评价指标（验证集返回的评价指标是一个有序字典）判断当前权重是否最优。

  ```python
  default_hooks = dict(checkpoint=dict(type='CheckpointHook', save_best='auto'))
  ```

  也可以直接指定 `save_best` 的值为评价指标，例如在分类任务中，可以指定为 `save_best='top-1'`，则会根据 `'top-1'` 的值判断当前权重是否最优。

  除了 `save_best` 参数，和保存最优权重相关的参数还有 `rule`，`greater_keys` 和 `less_keys`，这三者用来判断 `save_best` 的值是越大越好还是越小越好。例如指定了 `save_best='top-1'`，可以指定 `rule='greater'`，则表示该值越大表示权重越好。

- 指定保存权重的路径

  权重默认保存在工作目录（work_dir），但可以通过设置 `out_dir` 改变保存路径。

  ```python
  default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5, out_dir='/path/of/directory'))
  ```

- 制作发布用的权重

  如果你想在训练结束后自动生成可发布的权重（删除不需要的权重，例如优化器状态），你可以设置 `published_keys` 参数，选择需要保留的信息。注意：需要相应设置 `save_best` 或者 `save_last` 参数，这样才会生成可发布的权重，其中设置 `save_best` 会生成最优权重的可发布权重，设置 `save_last` 会生成最后一个权重的可发布权重，这两个参数也可同时设置。

  ```python
  default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, save_best='accuracy', rule='less', published_keys=['meta', 'state_dict']))
  ```

- 设置开始保存权重的 epoch 数或者 iteration 数

  如果想要设置控制开始保存权重的 epoch 数或者 iteration 数，可以设置 `save_begin` 参数，默认为 0，表示从训练开始就保存权重。例如，如果总共训练 10 个 epoch，并且 `save_begin` 设置为 5，则将保存第 5、6、7、8、9 和 10 个 epoch 的权重。如果 `interval=2`，则仅保存第 5、7 和 9 个 epoch 的权重。

  ```python
  default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2, save_begin=5))
  ```

### LoggerHook

[LoggerHook](mmengine.hooks.LoggerHook) 负责收集日志并把日志输出到终端或者输出到文件、TensorBoard 等后端。

如果我们希望每迭代 20 次就输出（或保存）一次日志，我们可以设置 `interval` 参数，配置如下：

```python
default_hooks = dict(logger=dict(type='LoggerHook', interval=20))
```

如果你对日志的管理感兴趣，可以阅读[记录日志（logging）](../advanced_tutorials/logging.md)。

### ParamSchedulerHook

[ParamSchedulerHook](mmengine.hooks.ParamSchedulerHook) 遍历执行器的所有优化器参数调整策略（Parameter Scheduler）并逐个调用 step 方法更新优化器的参数。如需了解优化器参数调整策略的用法请阅读[文档](../tutorials/param_scheduler.md)。`ParamSchedulerHook` 默认注册到执行器并且没有可配置的参数，所以无需对其做任何配置。

### IterTimerHook

[IterTimerHook](mmengine.hooks.IterTimerHook) 用于记录加载数据的时间以及迭代一次耗费的时间。`IterTimerHook` 默认注册到执行器并且没有可配置的参数，所以无需对其做任何配置。

### DistSamplerSeedHook

[DistSamplerSeedHook](mmengine.hooks.DistSamplerSeedHook) 在分布式训练时调用 Sampler 的 step 方法以确保 shuffle 参数生效。`DistSamplerSeedHook` 默认注册到执行器并且没有可配置的参数，所以无需对其做任何配置。

### RuntimeInfoHook

[RuntimeInfoHook](mmengine.hooks.RuntimeInfoHook) 会在执行器的不同钩子位点将当前的运行时信息（如 epoch、iter、max_epochs、max_iters、lr、metrics等）更新至 message hub 中，以便其他无法访问执行器的模块能够获取到这些信息。`RuntimeInfoHook` 默认注册到执行器并且没有可配置的参数，所以无需对其做任何配置。

### EMAHook

[EMAHook](mmengine.hooks.EMAHook) 在训练过程中对模型执行指数滑动平均操作，目的是提高模型的鲁棒性。注意：指数滑动平均生成的模型只用于验证和测试，不影响训练。

```python
custom_hooks = [dict(type='EMAHook')]
runner = Runner(custom_hooks=custom_hooks, ...)
runner.train()
```

`EMAHook` 默认使用 `ExponentialMovingAverage`，可选值还有 `StochasticWeightAverage` 和 `MomentumAnnealingEMA`。可以通过设置 `ema_type` 使用其他的平均策略。

```python
custom_hooks = [dict(type='EMAHook', ema_type='StochasticWeightAverage')]
```

更多用法请阅读 [EMAHook API 文档](mmengine.hooks.EMAHook)。

### EmptyCacheHook

[EmptyCacheHook](mmengine.hooks.EmptyCacheHook) 调用 `torch.cuda.empty_cache()` 释放未被使用的显存。可以通过设置 `before_epoch`, `after_iter` 以及 `after_epoch` 参数控制释显存的时机，第一个参数表示在每个 epoch 开始之前，第二参数表示在每次迭代之后，第三个参数表示在每个 epoch 之后。

```python
# 每一个 epoch 结束都会执行释放操作
custom_hooks = [dict(type='EmptyCacheHook', after_epoch=True)]
runner = Runner(custom_hooks=custom_hooks, ...)
runner.train()
```

### SyncBuffersHook

[SyncBuffersHook](mmengine.hooks.SyncBuffersHook) 在分布式训练每一轮（epoch）结束时同步模型的 buffer，例如 BN 层的 `running_mean` 以及 `running_var`。

```python
custom_hooks = [dict(type='SyncBuffersHook')]
runner = Runner(custom_hooks=custom_hooks, ...)
runner.train()
```

### ProfilerHook

[ProfilerHook](mmengine.hooks.ProfilerHook) 用于分析模型算子的执行时间以及显存占用情况。

```python
custom_hooks = [dict(type='ProfilerHook', on_trace_ready=dict(type='tb_trace'))]
runner = Runner(custom_hooks=custom_hooks, ...)
runner.train()
```

profile 的结果会保存在 `work_dirs/{timestamp}` 下的 `tf_tracing_logs` 目录，通过 `tensorboard --logdir work_dirs/{timestamp}tf_tracing_logs`。

更多关于 ProfilerHook 的用法请阅读 [ProfilerHook](mmengine.hooks.ProfilerHook) 文档。

## 自定义钩子

如果 MMEngine 提供的默认钩子不能满足需求，用户可以自定义钩子，只需继承钩子基类并重写相应的位点方法。

例如，如果希望在训练的过程中判断损失值是否有效，如果值为无穷大则无效，我们可以在每次迭代后判断损失值是否无穷大，因此只需重写 `after_train_iter` 位点。

```python
import torch

from mmengine.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class CheckInvalidLossHook(Hook):
    """Check invalid loss hook.

    This hook will regularly check whether the loss is valid
    during training.

    Args:
        interval (int): Checking interval (every k iterations).
            Defaults to 50.
    """

    def __init__(self, interval=50):
        self.interval = interval

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """All subclasses should override this method, if they need any
        operations after each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        if self.every_n_train_iters(runner, self.interval):
            assert torch.isfinite(outputs['loss']),\
                runner.logger.info('loss become infinite or NaN!')
```

我们只需将钩子的配置传给执行器的 `custom_hooks` 的参数，执行器初始化的时候会注册钩子，

```python
from mmengine.runner import Runner

custom_hooks = [
    dict(type='CheckInvalidLossHook', interval=50)
]
runner = Runner(custom_hooks=custom_hooks, ...)  # 实例化执行器，主要完成环境的初始化以及各种模块的构建
runner.train()  # 执行器开始训练
```

便会在每次模型前向计算后检查损失值。

注意，自定义钩子的优先级默认为 `NORMAL (50)`，如果想改变钩子的优先级，则可以在配置中设置 priority 字段。

```python
custom_hooks = [
    dict(type='CheckInvalidLossHook', interval=50, priority='ABOVE_NORMAL')
]
```

也可以在定义类时给定优先级

```python
@HOOKS.register_module()
class CheckInvalidLossHook(Hook):

    priority = 'ABOVE_NORMAL'
```

你可能还想阅读[钩子的设计](../design/hook.md)或者[钩子的 API 文档](../api/hooks)。
