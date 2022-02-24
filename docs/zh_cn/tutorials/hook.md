# 钩子（Hook）

钩子编程是一种编程模式，是指在程序的一个或者多个位置设置挂载点（位点），当程序运行至某个挂载点时，会自动调用运行时注册到挂载点的所有方法。钩子编程可以提高程序的灵活性和拓展性，用户将自定义的方法注册到挂载点便可被调用执行而无需修改程序中的代码。下面是钩子的简单示例。

```python
pre_hooks = [(print, 'hello')]
post_hooks = [(print, 'good bye')]

def main():
    for func, arg in pre_hooks:
        func(arg)
    print('do something here')
    for func, arg in post_hooks:
        func(arg)

main()
"""
hello
do something here
good bye
"""
```

可以看到，`main` 函数会在两个位置调用钩子中的函数而无需做任何改动。

## 钩子的设计

在介绍 MMEngine 钩子的设计之前，我们先简单介绍如何使用 PyTorch 编写一个简单的[训练脚本](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)：

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
    # define datasets and dataloader
    transform = transforms.ToTensor()
    train_dataset = CustomDataset(transform=transform, ...)
    val_dataset = CustomDataset(transform=transform, ...)
    test_dataset = CustomDataset(transform=transform, ...)
    train_dataloader = DataLoader(train_dataset, ...)
    val_dataloader = DataLoader(val_dataset, ...)
    test_dataloader = DataLoader(test_dataset, ...)

    # define a neural network
    net = Net()
    # define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for i in range(max_epochs):
        for inputs, labels in train_dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
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

上面的伪代码是训练一个模型的基本过程，为了实现无侵入定制训练过程，我们将训练过程划分为数个位点，只需要在这些位点插入各种逻辑即可达到目的，例如加载模型权重、更新模型参数。
因此，MMEngine 中钩子的作用是在不改变训练代码的前提下，灵活地在不同位点插入定制化的功能。根据需要，我们在训练过程中设置了16个位点，下面根据位点被调用的先后顺序列出这 16 个位点：

- before_run: 训练开始前执行
- after_load_checkpoint: 加载权重后执行
- before_train_epoch: 遍历训练数据集前执行
- before_train_iter: 模型前向计算前执行
- after_train_iter: 模型前向计算后执行
- after_train_epoch: 遍历完成训练数据集后执行
- before_val_epoch: 遍历验证数据集前执行
- before_val_iter: 模型前向计算前执行
- after_val_iter: 模型前向计算后执行
- after_val_epoch: 遍历完成验证数据集前执行
- before_save_checkpoint: 保存权重前执行
- after_train_epoch: 遍历完成训练数据集后执行
- before_test_epoch: 遍历测试数据集前执行
- before_test_iter: 模型前向计算前执行
- after_test_iter: 模型前向计算后执行
- after_test_epoch: 遍历完成测试数据集后执行
- after_run: 训练结束后执行

而控制整个训练过程的抽象在 MMEngine 中被设计为 Runner，它的行为之一是调用钩子完成训练过程。MMEngine 提供了两种类型的 Runner，一种是以 epoch 为单位迭代的 [EpochBasedRunner](https://github.com/open-mmlab/mmengine/blob/main/trainier/epoch_based_runner.py)，另一种是以 iteration 为单位迭代的 [IterBasedRunner](https://github.com/open-mmlab/mmengine/blob/main/trainier/iter_based_runner.py)。下面给出 EpochBasedRunner 调用钩子的伪代码。

```python
class EpochBasedRunner(BaseRunner):

    def run(self):
        self.call_hook('before_run')
        self.call_hook('after_load_checkpoint', checkpoint)
        # train + val
        for i in range(self.max_epochs):
            self.call_hook('before_train_epoch')
            for img, data_sample in self.train_dataloader:
                self.call_hook('before_train_iter', data_sample)
                outputs = model(img, data_sample)
                self.call_hook('after_train_iter', data_sample, outputs)
            self.call_hook('after_train_epoch')

            self.call_hook('before_val_epoch')
            if self._should_validate(i):
                for img, data_sample in self.val_dataloader:
                    self.call_hook('before_val_iter', data_sample)
                    outputs = model(img, data_sample)
                    self.call_hook('after_val_iter', data_sample, outputs)
            self.call_hook('after_val_epoch')

            self.call_hook('before_save_checkpoint', checkpoint)

        # test
        self.call_hook('before_test_epoch')
        if self._should_test():
            for img, data_sample in self.test_dataloader:
                self.call_hook('before_test_iter', data_sample)
                outputs = model(img, data_sample)
                self.call_hook('after_test_iter', data_sample, outputs)
        self.call_hook('after_test_epoch')

        self.call_hook('after_run')
```

## MMEngine 内置的钩子

MMEngine 提供了很多内置的钩子，每个钩子都有对应的优先级。在 Runner 训练过程中，同一位点，钩子的优先级越高，越早被调用，如果优先级一样，被调用的顺序和钩子注册的顺序一致。

| 名称      |      用途      |  优先级 |
|:----------:|:-------------:|:------:|
| CheckpointHook | 按指定间隔保存权重 | NORMAL (50) |
| OptimizerHook | 反向传播以及参数更新 | ABOVE_NORMAL (40) |
| ParamSchedulerHook | 调用 ParamScheduler 的 step 方法 | VERY_HIGH (10) |
| IterTimerHook | 统计迭代耗时 | LOW (70) |
| DistSamplerSeedHook | 确保分布式 Sampler 的 shuffle 生效 | NORMAL (50) |
| EmptyCacheHook | PyTorch CUDA 缓存清理 | NORMAL (50) |
| SyncBuffersHook | 同步模型的 buffer | NORMAL (50) |

### CheckpointHook

`CheckpointHook` 按照给定间隔保存模型的权重，如果是分布式多卡训练，则只有主（master）进程会保存权重。

假设我们一共训练 21 个 epoch 并希望每间隔 5 个 epoch 保存一次权重，下面的配置即可帮我们实现该需求。

```python
from mmengine import HOOKS

# by_epoch 的默认值为 True
checkpoint_config = dict(type='CheckpointHook', internal=5, by_epoch=True)
HOOKS.build(checkpoint_config)
```

上面的配置会保存第 5, 10, 15, 20 个 epoch 的权重。但是不会保存最后一个 epoch（第 21 个 epoch）的权重，因为 `interval=5` 表示每间隔 5 个 epoch 才会保存一次权重，而第 21 个 epoch 还没有间隔 5 个 epoch，不过可以通过设置 `save_last=True` 保存最后一个 epoch 的权重。

```python
checkpoint_config = dict(type='CheckpointHook', internal=5, by_epoch=True, save_last=True)
```

如果使用 `IterBasedTrainer`，则可以将 `by_epoch` 设为 False，`internal=5` 则表示每迭代 5 次保存一次权重。

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

假如一共训练 20 个 epoch，那么会在第 5, 10, 15, 20 个 epoch 保存模型，但是在第 15 个 epoch 的时候会删除第 5 个 epoch 保存的权重，在第 20 个 epoch 的时候会删除第 10 个 epoch 的权重，最终只有第 15 和第 20 个 epoch 的权重才会被保存。

考虑到分布式训练过程，如果有必要（例如分布式训练中没有使用同步 BN，而是普通 BN），则可以设置参数 `sync_buffer=True`，在保存权重前，会对模型 buffers（典型的例如 BN 的全局均值和方差参数）进行跨卡同步，让每张卡的 buffers 参数都相同，此时在主进程保存的权重和 buffer 才是符合期望的行为。

```python
checkpoint_config = dict(type='CheckpointHook', internal=5, sync_buffer=True)
```

### OptimizerHook

`OptimizerHook` 包含一些 optimizer 相关的操作：

- 梯度清零 runner.optimizer.zero_grad()
- 反向传播 runner.output['loss'].backward()
- 梯度截断 clip_grads（可选）
- 参数更新 runner.optimizer.step()

```python
from mmengine import HOOKS

optimizer_config = dict(type='OptimizerHook')
HOOKS.build(optimizer_config)
```

使用以上配置即可实现在 Trainer 中完成梯度清零、反向传播以及参数更新。

如果我们想对梯度进行截断，避免梯度爆炸，则可以设置 grad_clip 参数，该参数的设置可参考 [clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)

```python
optimizer_config=dict(type='OptimizerHook', grad_clip=dict(max_norm=35, norm_type=2))
```

模型中可能存在不参与计算图的模型参数，有两种可能，一种是该参数没有参与前向计算，另一种参与了前向计算但没有参与 loss 的计算。而如果模型中存在这种参数，会导致 PyTorch 抛出错误 `RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one`。我们可以通过设置 `detect_anomalous_params=True` 来检测并找出这种参数。

```python
optimizer_config=dict(type='OptimizerHook', detect_anomalous_params=True))
```

```{note}
`detect_anomalous_params=True` 会降低训练速度，推荐只用于调试。
```

除了 `OptimizerHook`，MMEngine 还提供了 `Fp16OptimizerHook` 和 `GradientCumulativeOptimizerHook`，前者用于混合精度训练，后者用于梯度累计。

`Fp16OptimizerHook` 是混合精度训练在 MMEngine 中的实现，主要逻辑如下：

- 维护一个 FP32 数值精度模型的副本
- 在每个 iteration
  - 拷贝并且转换成 FP16 模型
  - 前向传播（FP16 的模型参数)，此时 weights, activations 都是 FP16
  - loss 乘缩放参数 s，避免非 0 梯度溢出
  - 反向传播（FP16 的模型参数和参数梯度)， 此时 gradients 也是 FP16
  - 参数梯度乘 1/s
  - 利用 FP16 的梯度更新 FP32 的模型参数

![Fp16OptimizerHook](https://user-images.githubusercontent.com/58739961/154833936-abd7de05-ab67-4176-afef-bb647363736c.png)

关于 `Fp16OptimizerHook` 的使用请阅读[如何节省显存消耗](TODO)。

`GradientCumulativeOptimizerHook` 用于节省显存，即通过指定梯度累积的次数，实现反向传播多次才更新参数，常常用于显存不足但希望用较大的 batch size 训练模型。

```python
# cumulative_iters=4 表示累加参数梯度 4 次才更新一次参数
optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)
```

### ParamSchedulerHook

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

如果是非分布式训练可以设置 `distributed=False` 便使该钩子失效，即不同步模型的 buffer。

```python
config = dict(type='SyncBuffersHook', distributed=False)
```

## 添加自定义钩子

如果 MMEngine 提供的默认钩子不能满足需求，用户可以自定义钩子，只需继承钩子基类并重写相应的位点方法。

例如，如果希望在训练的过程中判断损失值是否有效，如果值为无穷大则无效，我们可以在每次迭代后判断损失值是否无穷大，因此只需重写  `after_train_iter` 位点。

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
            Default: 50.
    """

    def __init__(self, interval=50):
        self.interval = interval

    def after_train_iter(self, runner, data_batch):
        if self.every_n_iters(runner, self.interval):
            assert torch.isfinite(runner.outputs['loss']), \
                runner.logger.info('loss become infinite or NaN!')
```
