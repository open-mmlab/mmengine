# Hook

Hook programming is a programming pattern in which a mount point is set in one or more locations of a program. When the program runs to a mount point, all methods registered to it at runtime are automatically called. Hook programming can increase the flexibility and extensibility of the program since users can register custom methods to the mount point to be called without modifying the code in the program.

## Examples

Here is an example of how it works.

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

Output of the above example.

```
hello
do something here
goodbye
```

As we can see, the `main` function calls `print` defined in hooks in two locations without making any changes.

Hook is also used everywhere in PyTorch, for example in the neural network module (nn.Module) to get the forward input and output of the module as well as the reverse input and output. For example, the [`register_forward_hook`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) method registers a forward hook with the module, and the hook can get the forward input and output of the module.

The following is an example of the `register_forward_hook` usage.

```python
import torch
import torch.nn as nn

def forward_hook_fn(
    module,  # object to be registered hooks
    input,   # forward input of module
    output,  # forward output of module
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
# Register forward_hook_fn to each submodule of model
for module in model.children():
    module.register_forward_hook(forward_hook_fn)

x = torch.Tensor([[0.0, 1.0, 2.0]])
y = model(x)
```

Output of the above example.

```python
"forward_hook_fn" is invoked by Linear(in_features=3, out_features=1, bias=True)
weight: tensor([[-0.4077,  0.0119, -0.3606]])
bias: tensor([-0.2943])
input: (tensor([[0., 1., 2.]]),)
output: tensor([[-1.0036]], grad_fn=<AddmmBackward>)
```

We can see that the `forward_hook_fn` hook registered to the `nn.Linear` module is called, and in that hook the weights, biases, module inputs, and outputs of the Linear module are printed. For more information on the use of PyTorch hooks you can read [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).

## Design on MMEngine

Before introducing the design of the `Hook` in MMEngine, let's briefly introduce the basic steps of model training using PyTorch (copied from [PyTorch Tutorials](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)).

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

The above pseudo-code is the basic step to train a model. If we want to add custom operations to the above code, we need to modify and extend the `main` function continuously. To increase the flexibility and extensibility of the `main` function, we can insert mount points into the `main` function and implement the logic of calling hooks at the corresponding mount points. In this case, we only need to insert hooks into these locations to implement custom logic, such as loading model weights, updating model parameters, etc.

```python
def main():
    ...
    call_hooks('before_run', hooks)
    call_hooks('after_load_checkpoint', hooks)
    call_hooks('before_train', hooks)
    for i in range(max_epochs):
        call_hooks('before_train_epoch', hooks)
        for inputs, labels in train_dataloader:
            call_hooks('before_train_iter', hooks)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            call_hooks('after_train_iter', hooks)
            loss.backward()
            optimizer.step()
        call_hooks('after_train_epoch', hooks)

        call_hooks('before_val_epoch', hooks)
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                call_hooks('before_val_iter', hooks)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                call_hooks('after_val_iter', hooks)
        call_hooks('after_val_epoch', hooks)

        call_hooks('before_save_checkpoint', hooks)
    call_hooks('after_train', hooks)

    call_hooks('before_test_epoch', hooks)
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            call_hooks('before_test_iter', hooks)
            outputs = net(inputs)
            accuracy = ...
            call_hooks('after_test_iter', hooks)
    call_hooks('after_test_epoch', hooks)

    call_hooks('after_run', hooks)
```

In MMEngine, we encapsulates the training process into an executor (`Runner`). The `Runner` calls hooks at specific mount points to complete the customization logic. For more information about `Runner`, please read the [Runner documentation](../tutorials/runner.md).

To facilitate management, MMEngine defines mount points as methods and integrates them into [Base Hook](mmengine.hooks.Hook). We just need to inherit the base hook and implement custom logic at specific location according to our needs, then register the hooks to the `Runner`. Those hooks will be called automatically.

There are 22 mount points in the [Base Hook](mmengine.hooks.Hook).

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

Further readings: [Hook tutorial](../tutorials/hook.md) and [Hook API documentations](../api/hooks)
