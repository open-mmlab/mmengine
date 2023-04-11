# 钩子

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

可以看到注册到 Linear 模块的 `forward_hook_fn` 钩子被调用，在该钩子中打印了 Linear 模块的权重、偏置、模块的输入以及输出。更多关于 PyTorch 钩子的用法可以阅读 [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)。

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

在 MMEngine 中，我们将训练过程抽象成执行器（Runner），执行器除了完成环境的初始化，另一个功能是在特定的位点调用钩子完成定制化逻辑。更多关于执行器的介绍请阅读[执行器文档](../tutorials/runner.md)。

为了方便管理，MMEngine 将位点定义为方法并集成到[钩子基类（Hook）](mmengine.hooks.Hook)中，我们只需继承钩子基类并根据需求在特定位点实现定制化逻辑，再将钩子注册到执行器中，便可自动调用钩子中相应位点的方法。

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
- before_val_epoch
- after_val_epoch
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

你可能还想阅读[钩子的用法](../tutorials/hook.md)或者[钩子的 API 文档](../api/hooks)。
