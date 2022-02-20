# 钩子（Hook）

钩子编程是一种编程模式，是指在程序的一个或者多个位置设置挂载点（位点），当程序运行至某个挂载点时，会自动调用所有运行时注册到挂载点的方法。钩子编程的优点之一是提高程序的灵活性，用户将自定义的方法注册到挂载点便可被调用执行而无需修改程序中的代码。下面是钩子的简单示例。

```python
pre_hooks = []
post_hooks = []

def main():
    for func, arg in pre_hooks:
        func(arg)
    # do something here
    for func, arg in post_hooks:
        func(arg)

pre_hooks.append((print, 'hello'))
post_hooks.append((print, 'good bye'))

main()
```

## 钩子设计

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
因此，MMEngine 中钩子的作用是在不改变训练代码的前提下，灵活地在不同位点插入定制化的功能。根据需要，我们将训练过程划分成 16 个位点，下面根据位点被调用的先后顺序列出这 16 个位点：

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

而控制整个训练过程的抽象在 MMEngine 中被设计为 Trainer，它的行为之一是调用钩子完成训练过程。MMEngine 提供了两种类型的 Trainer，一种是以 epoch 为单位迭代的 [EpochBasedTrainer](https://github.com/open-mmlab/mmengine/blob/main/trainier/epoch_based_runner.py)，另一种是以 iteration 为单位迭代的 [IterBasedTrainer](https://github.com/open-mmlab/mmengine/blob/main/trainier/iter_based_runner.py)。下面给出 EpochBasedTrainer 调用钩子的伪代码。

```python
class EpochBasedTrainer(BaseTrainer):

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

MMEngine 提供数个常用钩子，下面一一介绍这些钩子的用法。

## 钩子用法

### CheckpointHook

CheckpointHook 按照给定间隔保存模型的权重，如果是分布式多卡训练，则只有主（master）进程会保存权重。

假设我们一共训练 21 个 epoch 并希望每间隔 5 个 epoch 保存一次权重，下面的配置即可帮我们实现该需求。

```python
from mmengine import HOOKS

# by_epoch 的默认值为 True
ckpt_cfg = dict(type='CheckpointHook', internal=5, by_epoch=True)
HOOKS.build(ckpt_cfg)
```

上面的配置会保存第 5, 10, 15, 20 个 epoch 的权重。但是不会保存最后一个 epoch（第 21 个 epoch）的权重，因为 `interval=5` 表示每间隔 5 个 epoch 才会保存一次权重，而第 21 个 epoch 还没有间隔 5 个 epoch，不过可以通过设置 `save_last=True` 保存最后一个 epoch 的权重。

```python
ckpt_cfg = dict(type='CheckpointHook', internal=5, by_epoch=True, save_last=True)
```

如果 IterBasedRunner，则可以将 `by_epoch` 设为 False，`internal=5` 表示每迭代 5 次保存一次权重。

```python
ckpt_cfg = dict(type='CheckpointHook', internal=5, by_epoch=False)
```

权重默认保存在工作目录（work_dir），但可以通过设置 `out_dir` 改变保存路径。

```python
ckpt_cfg = dict(type='CheckpointHook', internal=5, out_dir='/path/of/directory')
```

如果只想保存一定数量的权重，可以通过设置 `max_keep_ckpts` 参数实现最多保存 `max_keep_ckpts` 个权重，当保存的权重数超过 `max_keep_ckpts` 时，前面的权重会被删除。

```python
ckpt_cfg = dict(type='CheckpointHook', internal=5, max_keep_ckpts=2)
```

假如一共训练 20 个 epoch，那么会在第 5, 10, 15, 20 个 epoch 保存模型，但是在第 15 个 epoch 的时候会删除第 5 个 epoch 保存的权重，在第 20 个 epoch 的时候会删除第 10 个 epoch 的权重，最终只有第 15 和第 20 个 epoch 的权重才会被保存。

考虑到分布式训练过程，如果有必要（例如分布式训练中没有使用同步 BN，而是普通 BN），则可以设置参数 `sync_buffer=True`，在保存权重前，会对模型 buffers（典型的例如 BN 的全局均值和方差参数）进行跨卡同步，让每张卡的 buffers 参数都相同，此时在主进程保存的权重和 buffer 才是符合期望的行为。

```python
ckpt_cfg = dict(type='CheckpointHook', internal=5, sync_buffer=True)
```

### OptimizerHook

### ParamSchedulerHook

### LoggerHook

### EMAHook

### IterTimerHook

### DistSamplerSeedHook

### EmptyCacheHook

### ProfilerHook

### SyncBuffersHook

## 定制钩子

如果 MMEngine 提供的钩子不能满足需求，我们可以定制自己的钩子，只需继承 Hook 类并重写相应的位点。

例如，如果希望在训练的过程中判断损失值是否有效，如果无穷大则无效，我们可以在每次迭代后判断损失值，因此只需重写  `after_train_iter` 位点。

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
