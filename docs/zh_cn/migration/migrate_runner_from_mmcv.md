# 迁移 MMCV 执行器到 MMEngine

## 简介

随着支持的深度学习任务越来越多，用户的需求不断增加，MMCV 的执行器（Runner）逐渐难以满足需求。MMEngine 在此基础上，扩大了执行器的作用域，让执行器承担更多的功能；抽象出[训练循环控制器（EpochBasedTrainLoop/IterBasedTrainLoop）](mmengine.runner.EpochBasedLoop)、[验证循环控制器（ValLoop）](mmengine.runner.ValLoop)和[测试循环控制器（TestLoop）](mmengine.runner.TestLoop)以满足更加复杂的训练流程。在开始迁移前，建议先阅读[执行器教程](../tutorials/runner.md)

## 迁移执行器（Runner）

正如[执行器教程](../tutorials/runner.md#手动构建模块来使用执行器)一节所述，使用执行器的某一项功能之前需要准备好对应功能所依赖的模块。本节主要介绍 MMCV 执行器和 MMEngine 执行器在准备训练、验证、测试时，流程上的区别。

### 准备日志器（logger）

**MMCV 准备日志器**
MMCV 需要在训练脚本里调用 `get_logger` 接口获得日志器，用它打印一些训练信息。日志器还是执行器的构造参数，

```python
logger = get_logger(name='custom', log_file=log_file, log_level=cfg.log_level)
env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
...
runner = Runner(
    ...
    logger=logger
    ...)
```

**MMEngine 准备日志器**

MMEngine 只需在执行器构造时传入日志器的日志等级。

```python
log_level = 'INFO'
```

### 设置随机种子

**MMCV 设计随机种子**

在训练脚本中手动的设置随机种子：

```python
...
seed = init_random_seed(args.seed, device=cfg.device)
seed = seed + dist.get_rank() if args.diff_seed else seed
logger.info(f'Set random seed to {seed}, '
            f'deterministic: {args.deterministic}')
set_random_seed(seed, deterministic=args.deterministic)
...
```

**MMEngine 设计随机种子**

配置执行器的 `randomness` 参数即可，配置规则详见[执行器 api 文档](https://mmengine.readthedocs.io/zh_CN/latest/api.html#mmengine.runner.Runner.set_randomness)

```python
randomness = dict(seed=5)
```

### 配置训练环境

MMCV 需要在训练脚本中实现配置环境变量，例如多进程启动方式、多进程通信后端等相关变量。而 MMEngine 只需要为执行器配置 `env_cfg`, 其默认值为 `dict(dist_cfg=dict(backend='nccl'))`，配置方式详见[执行器 api 文档](https://mmengine.readthedocs.io/zh_CN/latest/api.html#mmengine.runner.Runner.setup_env)，其默认值为：

```python
env_cfg = dict(dist_cfg=dict(backend='nccl'))
```

### 准备数据

MMCV 和 MMEngine 的执行器均可接受 DataLoader 类型的数据。因此准备数据的流程没有差异

```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CIFAR10(
    root='data', train=True, download=True, transform=transform)
train_dataloader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=2)

val_dataset = CIFAR10(
    root='data', train=False, download=True, transform=transform)
val_dataloader = DataLoader(
    val_dataset, batch_size=128, shuffle=False, num_workers=2)
```

```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CIFAR10(
    root='data', train=True, download=True, transform=transform)
train_dataloader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=2)

val_dataset = CIFAR10(
    root='data', train=False, download=True, transform=transform)
val_dataloader = DataLoader(
    val_dataset, batch_size=128, shuffle=False, num_workers=2)
```

### 准备模型

详见[迁移 MMCV 模型至 MMEngine](./migrate_model_from_mmcv.md)

```python
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel


class Model(BaseModel):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, img, label, mode):
        feat = self.pool(F.relu(self.conv1(img)))
        feat = self.pool(F.relu(self.conv2(feat)))
        feat = feat.view(-1, 16 * 5 * 5)
        feat = F.relu(self.fc1(feat))
        feat = F.relu(self.fc2(feat))
        feat = self.fc3(feat)
        if mode == 'loss':
            loss = self.loss_fn(feat, label)
            return dict(loss=loss)
        else:
            return [feat.argmax(1)]

model = Model()
```

需要注意的是，分布式训练时，MMCV 的执行器需要接受分布式封装后的模型，而 `MMEngine` 直接接受模型，在执行器实例化阶对其段进行分布式封装。

### 分布式初始化

**MMCV 分布式初始化**

MMCV 需要在执行器构建之前初始化分布式环境，并对模型进行分布式封装：

```python
...
init_dist(cfg.launcher, **cfg.dist_params)
model = MMDistributedDataParallel(
    model,
    device_ids=[int(os.environ['LOCAL_RANK'])],
    broadcast_buffers=False,
    find_unused_parameters=find_unused_parameters)
```

**MMEngine 分布式初始化**

执行器构造函数有 `launcher` 参数，如果其值不为 `None`，执行器构建阶段会自动执行分布式初始化，模型分布式封装的流程。换句话说，使用 `MMEngine` 的执行器时，我们无需在执行器外做分布式相关的操作，只需要配置 launcher 参数，选择训练的启动方式即可。

### 准备优化器

对于简单配置的优化，MMCV 和 MMEngine 的准备流程相同

```python
from torch.optim import SGD

optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
```

对于复杂配置的优化器，MMCV 需要基于优化器构造器构造器：

```python

optimizer_cfg = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    paramwise_cfg=dict(norm_decay_mult=0))

def build_optimizer_constructor(cfg):
    constructor_type = cfg.get('type')
    if constructor_type in OPTIMIZER_BUILDERS:
        return build_from_cfg(cfg, OPTIMIZER_BUILDERS)
    elif constructor_type in MMCV_OPTIMIZER_BUILDERS:
        return build_from_cfg(cfg, MMCV_OPTIMIZER_BUILDERS)
    else:
        raise KeyError(f'{constructor_type} is not registered '
                       'in the optimizer builder registry.')


def build_optimizer(model, cfg):
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimizerConstructor')
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_constructor = build_optimizer_constructor(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))
    optimizer = optim_constructor(model)
    return optimizer

optimizer = build_optimizer(model, optimizer_cfg)
```

MMEngine 将上述流程封装在执行器中，因此无需定义 `build_optimizer`，在执行器实例化时传入 `optim_wrapper` 参数即可:

```python
optim_wrapper = build_optimizer(model, optimizer_cfg)
runner = Runner(
    ...
    optim_wrapper=optim_wrapper,
    ...
)

optim_wrapper 的配置详见[优化器封装教程](../tutorials/optim_wrapper.md)
```

### 准备训练钩子

**MMCV 准备训练钩子：**

MMCV 常用训练钩子的配置如下：

```python
# learning rate scheduler config
lr_config = dict(policy='step', step=[2, 3])
# configuration of optimizer
optimizer_config = dict(grad_clip=None)
# configuration of saving checkpoints periodically
checkpoint_config = dict(interval=1)
# save log periodically and multiple hooks can be used simultaneously
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
# register hooks to runner and those hooks will be invoked automatically
runner.register_training_hooks(
    lr_config=lr_config,
    optimizer_config=optimizer_config,
    checkpoint_config=checkpoint_config,
    log_config=log_config)
```

其中：

- `lr_config` -> `LrUpdaterHook`
- `optimizer_config` -> `OptimizerHook`
- `checkpoint_config` -> `CheckPointHook`
- `log_config` -> `LoggerHook`

除了上面提到的 4 个 Hook，MMCV 执行器还默认配有 `IterTimerHook`。MMCV 需要先实例化执行器，再注册训练钩子，我们在后面的章节中对比二者实例化的区别。

**MMEngine 准备训练钩子**

MMEngine 执行器自带一些默认钩子：

- RuntimeInfoHook
- IterTimerHook
- DistSamplerSeedHook
- LoggerHook
- CheckpointHook

如果想对齐上述 MMCV 的执行器配置，我们只需要额外配置[优化器参数调整策略](../tutorials/param_scheduler.md)：

```python
from math import gamma

param_scheduler = dict(type='MultiStepLR', milestones=[2, 3], gamma=0.1)
```

```{note}
MMEngine 移除了 `OptimizerHook`，优化步骤在 model 中执行。
```

### 准备验证模块

MMCV 借助 `EvalHook` 实现验证流程，受限于篇幅，这里不做进一步展开。MMEngine 通过[验证循环（ValLoop）](../tutorials/runner.md#自定义执行流程) 和[评测器（Evaluator）](../tutorials/metric_and_evaluator.md)实现执行流程，如果我们想基于自定义的评价指标完成验证流程，则需要定义一个 `Metric`，并将其注册至 `METRICS` 注册器：

```python
import torch
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

@METRICS.register_module(force=True)
class ToyAccuracyMetric(BaseMetric):

    def process(self, label, pred) -> None:
        self.results.append((label[1], pred, len(label[1])))

    def compute_metrics(self, results: list) -> dict:
        num_sample = 0
        acc = 0
        for label, pred, batch_size in results:
            acc += (label == torch.stack(pred)).sum()
            num_sample += batch_size
        return dict(Accuracy=acc / num_sample)
```

```python
val_evaluator = dict(type='ToyAccuracyMetric')
```

此外，我们还需要配置[验证循环（ValLoop）](../tutorials/runner.md#自定义执行流程) 和

```python
val_cfg = dict(type='ValLoop')
```

### 构建执行器

**MMCV 构建执行器**

```python
runner = EpochBasedRunner(
    model=model,
    optimizer=optimizer,
    work_dir=work_dir,
    logger=logger,
    max_epochs=4
)
```

**MMEngine 构建执行器**

`MMEngine` 执行器的作用域比 MMCV 更广，把很多配置流程参数化，例如设置随机种子、启动分布式训练，因此需要接受更多的参数。除了前几节提到的参数，EpochBasedRunner，IterBasedRunner，max_epochs，max_iters，val_iterval 等配置（之前由 EvalHook 决定） 在 train_cfg 里体现：

- by_epoch: `True` 时相当于 MMCV 的 `EpochBasedRunner`，False 时相当于 `IterBasedRunner`。
- max_epoch/max_iters: 同 MMCV 执行器的配置
- val_iterval: 同 `EvalHook` 的 `interval` 参数

```python
from mmengine import Runner

runner = Runner(
    model=model,
    work_dir='./work_dir',
    randomness=randomness,
    env_cfg=env_cfg,
    launcher='none',  # 不开启分布式训练
    optim_wrapper=dict(optimizer=optimizer),
    param_scheduler=param_scheduler,
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=4, val_interval=1),
    val_dataloader=val_dataloader,
    val_evaluator=val_evaluator,
    val_cfg=val_cfg)
```

### 执行器训练流程

**MMCV 执行器训练流程**：

需要在训练之前执行加载权重、恢复训练的流程。然后再执行 runner.run()，传入需要训练的数据。

```python
if cfg.resume_from:
    runner.resume(cfg.resume_from)
elif cfg.load_from:
    runner.load_checkpoint(cfg.load_from)
runner.run(data_loaders, cfg.workflow)
```

**MMEngine** 执行器训练流程

如果需要加载权重或恢复训练，可以在执行器构建时配置

```python
runner = Runner(
    ...
    load_from='/path/to/checkpoint',
    resume=True
)
```

由于 MMEngine 的执行器在构造阶段就传入了训练数据，因此在调用 runner.train() 无需传入参数。

```python
runner.train()
```

### 执行器测试流程

MMCV 的执行器没有测试功能，因此需要自行实现测试脚本。MMEngine 的执行器只需要在建时配置 `test_dataloader`、`test_cfg` 和 `test_evaluator`，然后再调用 `runner.test()` 就执行测试流程。

**如果 `work_dir` 和训练时一致，无需手动加载 checkpoint:**

```python
runner = Runner(
    model=model,
    work_dir='./work_dir',
    randomness=randomness,
    env_cfg=env_cfg,
    launcher='none',  # 不开启分布式训练
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_evaluator=val_evaluator,
    val_cfg=val_cfg,
    test_dataloader=val_dataloader,  # 假设测试和验证使用相同的数据和评测器
    test_evaluator=val_evaluator,
    test_cfg=dict(type='TestLoop'),
    )
runner.test()
```

**如果 `work_dir` 和训练时不一致，需要额外指定 load_from:**

```python
runner = Runner(
    model=model,
    work_dir='./test_work_dir',
    load_from='./work_dir/epoch_5.pth',  # work_dir 不一致，指定 load_from，以加载指定的模型
    randomness=randomness,
    env_cfg=env_cfg,
    launcher='none',
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_evaluator=val_evaluator,
    val_cfg=val_cfg,
    test_dataloader=val_dataloader,
    test_evaluator=val_evaluator,
    test_cfg=dict(type='TestLoop'),
)
runner.test()
```

## 迁移自定义执行流程

在使用 MMCV 执行器时，我们可能会重载 `runner.train()/runner.val()` 或者 `runner.run_iter` 实现自定义的训练、测试流程。以重载 `runner.train` 为例，假设我们想对每张图片训练两遍：

```python
class CustomRunner(EpochBasedRunner):
    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            for _ in range(2)
                self.call_hook('before_train_iter')
                self.run_iter(data_batch, train_mode=True, **kwargs)
                self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1
```

在 MMEngine 中，要实现上述功能，我们需要重载一个新的循环控制器

```python
import imp
from mmengine.registry import LOOPS
from mmengine.runner import EpochBasedTrainLoop


@LOOPS.register_module()
class CustomEpochBasedTrainLoop(EpochBasedTrainLoop):
    def run_iter(self, idx, data_batch) -> None:
        for _ in range(2):
            super().run_iter(idx, data_batch)
```

然后在构建执行器时，指定 `train_cfg` 的 `type` 为 `CustomEpochBasedTrainLoop`。需要注意的是，`by_epoch` 和 `type` 不能同时配置，当配置 `by_epoch` 时，会推断训练循环控制器（xxxBasedTrainLoop）的类型为 `EpochBasedTrainLoop`。

```python
runner = Runner(
    model=model,
    work_dir='./test_work_dir',
    randomness=randomness,
    env_cfg=env_cfg,
    launcher='none',
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
    train_dataloader=train_dataloader,
    train_cfg=dict(
        type='CustomEpochBasedTrainLoop',
        max_epochs=5,
        val_interval=1),
    val_dataloader=val_dataloader,
    val_evaluator=val_evaluator,
    val_cfg=val_cfg,
    test_dataloader=val_dataloader,
    test_evaluator=val_evaluator,
    test_cfg=dict(type='TestLoop'),
)
runner.train()
```
