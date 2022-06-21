# 钩子（Hook）

钩子编程是一种编程模式，是指在程序的一个或者多个位置设置位点（挂载点），当程序运行至某个位点时，会自动调用运行时注册到位点的所有方法。钩子编程可以提高程序的灵活性和拓展性，用户将自定义的方法注册到位点便可被调用而无需修改程序中的代码。

## 钩子示例

下面是钩子的简单示例。

```python
pre_hooks = [(print, 'hello')]
post_hooks = [(print, 'goodbye')]

def main():
    for func, arg in pre_hooks:
        func(arg)
    print('do something here')
    for func, arg in post_hooks:
        func(arg)

main()
```

下面是程序的输出：

```
hello
do something here
goodbye
```

可以看到，`main` 函数在两个位置调用钩子中的函数而无需做任何改动。

在 PyTorch 中，钩子的应用也随处可见，例如神经网络模块（nn.Module）中的钩子可以获得模块的前向输入输出以及反向的输入输出。以 [`register_forward_hook`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) 方法为例，该方法往模块注册一个前向钩子，钩子可以获得模块的前向输入和输出。

下面是 `register_forward_hook` 用法的简单示例：

```python
import torch
import torch.nn as nn

def forward_hook_fn(
    module,  # 被注册钩子的对象
    input,  # module 前向计算的输入
    output,  # module 前向计算的输出
):
    print(f'"forward_hook_fn" is invoked by {module.name}')
    print('weight:', module.weight.data)
    print('bias:', module.bias.data)
    print('input:', input)
    print('output:', output)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        y = self.fc(x)
        return y

model = Model()
# 将 forward_hook_fn 注册到 model 每个子模块
for module in model.children():
    module.register_forward_hook(forward_hook_fn)

x = torch.Tensor([[0.0, 1.0, 2.0]])
y = model(x)
```

下面是程序的输出：

```python
"forward_hook_fn" is invoked by Linear(in_features=3, out_features=1, bias=True)
weight: tensor([[-0.4077,  0.0119, -0.3606]])
bias: tensor([-0.2943])
input: (tensor([[0., 1., 2.]]),)
output: tensor([[-1.0036]], grad_fn=<AddmmBackward>)
```

可以看到注册到 Linear 模块的 `forward_hook_fn` 钩子被调用，在该钩子中打印了 Linear 模块的权重、偏置、模块的输入以及输出。更多关于 PyTorch 钩子的用法可以阅读 [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.htm)。

## MMEngine 中钩子的设计

在介绍 MMEngine 中钩子的设计之前，先简单介绍使用 PyTorch 实现模型训练的基本步骤（示例代码来自 [PyTorch Tutorials](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    pass

class Net(nn.Module):
    pass

def main():
    transform = transforms.ToTensor()
    train_dataset = CustomDataset(transform=transform, ...)
    val_dataset = CustomDataset(transform=transform, ...)
    test_dataset = CustomDataset(transform=transform, ...)
    train_dataloader = DataLoader(train_dataset, ...)
    val_dataloader = DataLoader(val_dataset, ...)
    test_dataloader = DataLoader(test_dataset, ...)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for i in range(max_epochs):
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                outputs = net(inputs)
                loss = criterion(outputs, labels)

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = net(inputs)
            accuracy = ...
```

上面的伪代码是训练模型的基本步骤。如果要在上面的代码中加入定制化的逻辑，我们需要不断修改和拓展 `main` 函数。为了提高 `main` 函数的灵活性和拓展性，我们可以在 `main` 方法中插入位点，并在对应位点实现调用 hook 的抽象逻辑。此时只需在这些位点插入 hook 来实现定制化逻辑，即可添加定制化功能，例如加载模型权重、更新模型参数等。

```python
def main():
    ...
    call_hooks('before_run', hooks)  # 任务开始前执行的逻辑
    call_hooks('after_load_checkpoint', hooks)  # 加载权重后执行的逻辑
    call_hooks('before_train', hooks)  # 训练开始前执行的逻辑
    for i in range(max_epochs):
        call_hooks('before_train_epoch', hooks)  # 遍历训练数据集前执行的逻辑
        for inputs, labels in train_dataloader:
            call_hooks('before_train_iter', hooks)  # 模型前向计算前执行的逻辑
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            call_hooks('after_train_iter', hooks)  # 模型前向计算后执行的逻辑
            loss.backward()
            optimizer.step()
        call_hooks('after_train_epoch', hooks)  # 遍历完训练数据集后执行的逻辑

        call_hooks('before_val_epoch', hooks)  # 遍历验证数据集前执行的逻辑
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                call_hooks('before_val_iter', hooks)  # 模型前向计算前执行
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                call_hooks('after_val_iter', hooks)  # 模型前向计算后执行
        call_hooks('after_val_epoch', hooks)  # 遍历完验证数据集前执行

        call_hooks('before_save_checkpoint', hooks)  # 保存权重前执行的逻辑
    call_hooks('after_train', hooks)  # 训练结束后执行的逻辑

    call_hooks('before_test_epoch', hooks)  # 遍历测试数据集前执行的逻辑
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            call_hooks('before_test_iter', hooks)  # 模型前向计算后执行的逻辑
            outputs = net(inputs)
            accuracy = ...
            call_hooks('after_test_iter', hooks)  # 遍历完成测试数据集后执行的逻辑
    call_hooks('after_test_epoch', hooks)  # 遍历完测试数据集后执行

    call_hooks('after_run', hooks)  # 任务结束后执行的逻辑
```

在 MMEngine 中，我们将训练过程抽象成执行器（Runner），执行器除了完成环境的初始化，另一个功能是在特定的位点调用钩子完成定制化逻辑。更多关于执行器的介绍请阅读[执行器文档](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html)。

为了方便管理，MMEngine 将位点定义为方法并集成到[钩子基类（Hook）](https://mmengine.readthedocs.io/zh/latest/api.html#hook)中，我们只需继承钩子基类并根据需求在特定位点实现定制化逻辑，再将钩子注册到执行器中，便可自动调用钩子中相应位点的方法。

钩子中一共有 22 个位点：

- before_run
- after_run
- before_train
- after_train
- before_train_epoch
- after_train_epoch
- before_train_iter
- after_train_iter
- before_val
- after_val
- before_test_epoch
- after_test_epoch
- before_val_iter
- after_val_iter
- before_test
- after_test
- before_test_epoch
- after_test_epoch
- before_test_iter
- after_test_iter
- before_save_checkpoint
- after_load_checkpoint

## 内置钩子

MMEngine 提供了很多内置的钩子，将钩子分为两类，分别是默认钩子以及自定义钩子。

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

|                     名称                      |             用途              |        优先级        |
| :-----------------------------------------: | :-------------------------: | :---------------: |
|     [RuntimeInfoHook](#runtimeinfohook)     |    往 message hub 更新运行时信息    |  VERY_HIGH (10)   |
|       [IterTimerHook](#itertimerhook)       |           统计迭代耗时            |    NORMAL (50)    |
| [DistSamplerSeedHook](#distsamplerseedhook) | 确保分布式 Sampler 的 shuffle 生效  |    NORMAL (50)    |
|          [LoggerHook](#loggerhook)          |            打印日志             | BELOW_NORMAL (60) |
|  [ParamSchedulerHook](#paramschedulerhook)  | 调用 ParamScheduler 的 step 方法 |     LOW (70)      |
|      [CheckpointHook](#checkpointhook)      |          按指定间隔保存权重          |   VERY_LOW (90)   |

**自定义钩子**

|                 名称                  |        用途         |     优先级      |
| :---------------------------------: | :---------------: | :----------: |
|         [EMAHook](#emahook)         |    模型参数指数滑动平均     | NORMAL (50)  |
|  [EmptyCacheHook](#emptycachehook)  | PyTorch CUDA 缓存清理 | NORMAL (50)  |
| [SyncBuffersHook](#syncbuffershook) |   同步模型的 buffer    | NORMAL (50)  |
|       NaiveVisualizationHook        |        可视化        | LOWEST (100) |

```{note}
不建议修改默认钩子的优先级，因为优先级低的钩子可能会依赖优先级高的钩子。例如 CheckpointHook 的优先级需要比 ParamSchedulerHook 低，这样保存的优化器状态才是正确的状态。另外，自定义钩子的优先级默认为 `NORMAL (50)`。
```

两种钩子在执行器中的设置不同，默认钩子的配置传给执行器的 `default_hooks` 参数，自定义钩子的配置传给 `custom_hooks` 参数，如下所示：

```python
from mmengine import Runner

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    logger=dict(type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
)

custom_hooks = [
    dict(type='NaiveVisualizationHook', priority='LOWEST'),
]

runner = Runner(default_hooks=default_hooks, custom_hooks=custom_hooks, ...)
runner.run()
```

下面逐一介绍 MMEngine 中内置钩子的用法。

### CheckpointHook

`CheckpointHook` 按照给定间隔保存模型的权重，如果是分布式多卡训练，则只有主（master）进程会保存权重。

假设我们一共训练 21 个 epoch 并希望每隔 5 个 epoch 保存一次权重，下面的配置即可帮我们实现该需求。

```python
from mmengine import HOOKS

# by_epoch 的默认值为 True
checkpoint_config = dict(type='CheckpointHook', internal=5, by_epoch=True)
HOOKS.build(checkpoint_config)
```

上面的配置会保存第 5, 10, 15, 20 个 epoch 的权重。但是不会保存最后一个 epoch（第 21 个 epoch）的权重，因为 `interval=5` 表示每隔 5 个 epoch 才保存一次权重，而第 21 个 epoch 还没有隔 5 个 epoch，不过可以通过设置 `save_last=True` 保存最后一个 epoch 的权重。

```python
checkpoint_config = dict(type='CheckpointHook', internal=5, by_epoch=True, save_last=True)
```

如果想以迭代次数作为保存间隔，则可以将 `by_epoch` 设为 False，`internal=5` 则表示每迭代 5 次保存一次权重。

```python
checkpoint_config = dict(type='CheckpointHook', internal=5, by_epoch=False)
```

权重默认保存在工作目录（work_dir），但可以通过设置 `out_dir` 改变保存路径。

```python
checkpoint_config = dict(type='CheckpointHook', internal=5, out_dir='/path/of/directory')
```

如果只想保存一定数量的权重，可以通过设置 `max_keep_ckpts` 参数实现最多保存 `max_keep_ckpts` 个权重，当保存的权重数超过 `max_keep_ckpts` 时，前面的权重会被删除。

```python
checkpoint_config = dict(type='CheckpointHook', internal=5, max_keep_ckpts=2)
```

上述例子表示，假如一共训练 20 个 epoch，那么会在第 5, 10, 15, 20 个 epoch 保存模型，但是在第 15 个 epoch 的时候会删除第 5 个 epoch 保存的权重，在第 20 个 epoch 的时候会删除第 10 个 epoch 的权重，最终只有第 15 和第 20 个 epoch 的权重才会被保存。

### LoggerHook

`LoggerHook` 负责收集日志并把日志输出到终端或者输出到文件、TensorBoard 等后端。

如果我们希望每迭代 20 次就输出（或保存）一次日志，我们可以设置 interval 参数，配置如下：

```python
config = dict(type='LoggerHook', interval=20)
```

如果我们希望训练结束后把指定后缀的文件转存到其他路径，例如 Ceph。我们可以设置 out_dir、out_suffix 和 keep_loal 三个参数。第一个参数表示将文件转存到指定的路径；第二个参数表示需要转存以哪些后缀结尾的文件，默认是 .json、.log、.py 和 yaml；第三个参数表示当我们把文件转存到其他路径后是否删除被转存的文件。

```python
config = dict(type='LoggerHook', out_dir='s3://save_log/', out_suffix=('.json', '.py'), keep_local=True)
```

### ParamSchedulerHook

`ParamSchedulerHook` 遍历执行器的所有优化器参数调整策略（Parameter Scheduler）并逐个调用 step 方法更新优化器的参数。如需了解优化器参数调整策略的用法请阅读[文档](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/param_scheduler.html)。

```python
from mmengine import Runner

scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1)

default_hooks = dict(scheduler_hook=dict(type='ParamSchedulerHook'))
runner = Runner(scheduler=scheduler, default_hooks=default_hooks, ...)
runner.run()
```

### IterTimerHook

`IterTimerHook` 用于记录加载数据的时间以及迭代一次耗费的时间。

```python
config = dict(type='IterTimerHook')
```

### DistSamplerSeedHook

`DistSamplerSeedHook` 在分布式训练时调用 Sampler 的 step 方法以确保 shuffle 参数生效。

```python
config = dict(type='DistSamplerSeedHook')
```

### EMAHook

`EMAHook` 在训练过程中对模型执行指数滑动平均操作，目的是提高模型的鲁棒性。注意：指数滑动平均生成的模型只用于验证和测试，不影响训练。

```python
config = dict(type='EMAHook')
```

### EmptyCacheHook

`EmptyCacheHook` 调用 `torch.cuda.empty_cache()` 释放未被使用的显存。`EmptyCacheHook` 会在 3 个位点调用 `torch.cuda.empty_cache()`，分别是 `before_epoch`, `after_iter` 以及 `after_epoch`，用户可以通过参数控制是否调用。

```python
config = dict(type='EmptyCacheHook', before_epoch=False, after_epoch=True, after_iter=False)
```

### SyncBuffersHook

`SyncBuffersHook` 在分布式训练每一轮（epoch）结束时同步模型的 buffer，例如 BN 层的 `running_mean` 以及 `running_var`。

```python
config = dict(type='SyncBuffersHook')
```

### RuntimeInfoHook

`RuntimeInfoHook` 会在执行器的不同钩子位点将当前的运行时信息（如 epoch、iter、max_epochs、max_iters、lr、metrics等）更新至 message hub 中，
以便其他无法访问执行器的模块能够获取到这些信息。

```python
config = dict(type='RuntimeInfoHook')
```

## 添加自定义钩子

如果 MMEngine 提供的默认钩子不能满足需求，用户可以自定义钩子，只需继承钩子基类并重写相应的位点方法。

例如，如果希望在训练的过程中判断损失值是否有效，如果值为无穷大则无效，我们可以在每次迭代后判断损失值是否无穷大，因此只需重写 `after_train_iter` 位点。

```python
import torch

from mmengine import HOOKS
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
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model.
                Defaults to None.
        """
        if self.every_n_train_iters(runner, self.interval):
            assert torch.isfinite(runner.outputs['loss']),\
                runner.logger.info('loss become infinite or NaN!')
```

我们只需将钩子的配置传给执行器的 custom_hooks 的参数，执行器初始化的时候会注册钩子，

```python
from mmengine import Runner

custom_hooks = dict(
    dict(type='CheckInvalidLossHook', interval=50)
)
runner = Runner(custom_hooks=custom_hooks, ...)  # 实例化执行器，主要完成环境的初始化以及各种模块的构建
runner.run()  # 执行器开始训练
```

便会在每次模型前向计算后检查损失值。

注意，自定义钩子的优先级默认为 `NORMAL (50)`，如果想改变钩子的优先级，则可以在配置中设置 priority 字段。

```python
custom_hooks = dict(
    dict(type='CheckInvalidLossHook', interval=50, priority='ABOVE_NORMAL')
)
```
