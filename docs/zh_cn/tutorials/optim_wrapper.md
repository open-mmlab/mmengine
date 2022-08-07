# 优化器封装（OptimWrapper）

MMEngine
实现了优化器封装，为用户提供了统一的优化器访问接口。优化器封装能够支持不同的训练策略，包括混合精度训练、梯度累加和梯度截断，用户可以根据需求选择合适的训练策略。优化器封装为梯度反传、参数更新、梯度清零、导出状态字典、获取参数学习率/动量提供了统一的接口。除此之外，优化器封装还定义了一套标准的参数更新流程。用户可以基于这一套流程，通过配置不同的优化器封装，实现同一套代码，不同训练策略的切换。

## 优化器封装 vs 优化器

这里我们分别使用 Pytorch 内置的优化器和优化器封装进行普通的单精度训练、混合精度训练和梯度累加，对比二者实现上的区别。优化器封装同样实现了一些和优化器相同的接口，

### 训练模型

**1. 使用 SGD 优化器 + 单精度训练**

```python
import torch
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F


inputs = [torch.zeros(10, 1, 1)] * 10
targets = [torch.ones(10, 1, 1)] * 10
dataset = zip(inputs, targets)
model = nn.Linear(1, 1)
optimizer = SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()

for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**2. 使用 SGD 优化器 + 混合精度训练**

```python
import torch
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


inputs = [torch.zeros(10, 1, 1)] * 10
targets = [torch.ones(10, 1, 1)] * 10
dataset = zip(inputs, targets)
model = nn.Linear(1, 1).cuda()
optimizer = SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()

for input, target in dataset:
    optimizer.zero_grad()
    with autocast():
        output = model(input.cuda())
    loss = F.mse_loss(output, target.cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**3. 使用 SGD 优化器 + 混合精度训练 + 梯度累加**

```python
import torch
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


inputs = [torch.zeros(10, 1, 1)] * 10
targets = [torch.ones(10, 1, 1)] * 10
dataset = zip(inputs, targets)
model = nn.Linear(1, 1).cuda()
optimizer = SGD(model.parameters(), lr=0.01)

for idx, (input, target) in enumerate(dataset):
    optimizer.zero_grad()
    with autocast():
        output = model(input.cuda())
    loss = F.mse_loss(output, target.cuda())
    loss.backward()
    if idx % 2 == 0:
        optimizer.step()
        optimizer.zero_grad()
```

如上例所示，三种不同的训练策略在流程上基本一致，只是细节方面有一些差异，例如混合精度训练需要开启 `autocast` 这个上下文管理器，梯度累加需要反传梯度多次才更新一次参数。

优化器封装对差异做了统一的抽象，用户只需要构造出符合需求的优化器封装实例，调用统一的接口即可实现不同策略的参数优化逻辑。

**4. 使用优化器封装实现单精度训练、混合精度训练和梯度累加**：

```python
import torch
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F
from mmengine.optim import OptimWrapper, AmpOptimWrapper

inputs = [torch.zeros(10, 1, 1)] * 10
targets = [torch.ones(10, 1, 1)] * 10
dataset = zip(inputs, targets)
model = nn.Linear(1, 1).cuda()
optimizer = SGD(model.parameters(), lr=0.01)

# 没有任何训练策略的优化器封装
optim_wapper = OptimWrapper(optimizer=optimizer)
# 混合精度训练的优化器封装
amp_optim_wapper = AmpOptimWrapper(optimizer=optimizer)
# 混合精度训练 + 梯度累加的优化器封装
amp_acc_optim_wapper = AmpOptimWrapper(optimizer=optimizer,
                                       accumulative_counts=2)

optim_wrappers = [optim_wapper, amp_optim_wapper, amp_acc_optim_wapper]
# 使用三种训练策略
for optim_wrapper in optim_wrappers:
    for idx, (input, target) in enumerate(dataset):
        optimizer.zero_grad()
        # 使用优化器封装后，需要使用 optim_context 开启混合精度训练上下文
        with optim_wrapper.optim_context(model):
            output = model(input.cuda())
        loss = F.mse_loss(output, target.cuda())
        # update_params 实现了标准的梯度计算、参数更新和梯度清零的逻辑
        optim_wapper.update_params(loss)
```

显而易见的，使用优化器封装后，我们可以使用统一的代码实现不同的训练策略。其中`OptimWrapper` 为标准的单精度优化器封装，实现了梯度累加、梯度截断的功能，用户可以在构造实例时配置 `accumulative_counts` 和 `clip_grad` 参数来开启相应功能、 用于训练普通的单精度模型，`AmpOptimWrapper` 在 `OptimWrapper` 的基础上实现了混合精度相关的功能，在 `optim_context` 的上下文中进行网络的前向推理，即可实现混合精度训练。

优化器封装还提供了 `update_params` 接口，实现了标准的参数更新流程，即：梯度反传、参数更新和梯度清零。因此上例中用户可以直接调用 `update_params`，而无需依次调用 `backward`、`step` 和 `zero_grad`。

除了 `update_params`， 优化器封装同样提供了更细粒度的接口，方便用户实现一些自定义的参数更新逻辑：

- `backward(loss: Tensor)`：传入损失，用于计算参数梯度，。
- `step(**kwargs)`： 同 `optimizer.step`，用于更新参数。
- `zero_grad`： 同 `optimizer.zero_grad`，用于参数的梯度。

下例手动实现了梯度累加逻辑，适用于单精度训练和混合精度训练。

```python
import torch
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F
from mmengine.optim import OptimWrapper, AmpOptimWrapper

inputs = [torch.zeros(10, 1, 1)] * 10
targets = [torch.ones(10, 1, 1)] * 10
dataset = zip(inputs, targets)
model = nn.Linear(1, 1).cuda()
optimizer = SGD(model.parameters(), lr=0.01)

# 没有任何训练策略的优化器封装
optim_wapper = OptimWrapper(optimizer=optimizer)
# 混合精度训练的优化器封装
amp_optim_wapper = AmpOptimWrapper(optimizer=optimizer)

optim_wrappers = [optim_wapper, amp_optim_wapper]
# 使用三种训练策略
for optim_wrapper in optim_wrappers:
    for idx, (input, target) in enumerate(dataset):
        optimizer.zero_grad()
        with optim_wrapper.optim_context(model):
            output = model(input.cuda())
        loss = F.mse_loss(output, target.cuda())
        optim_wapper.backward(loss)
        if idx % 2 == 0:
            optim_wapper.step()
            optim_wapper.zero_grad()
```

```{note}
使用梯度累加时，我们建议在构造优化器实例时指定 `accumulative_counts`，并组合使用 `optim_context` 和 `update_params` 以避免不必要的梯度同步，起到训练加速的效果。
```

```{note}
建议用户无论是否开启混合精度训练，都在模型前向推理前开启 `optim_context` 上下文。否则切换使用 `AmpOptimWrapper` 时，不会开启混合精度训练的功能。
```

### 获取学习率/动量：

优化器封装提供了 `get_lr` 和 `get_momentum` 接口用于获取优化器的一个参数组的学习率

```python
import torch.nn as nn
from torch.optim import sgd

from mmengine.optim import optimwrapper

model = nn.linear(1, 1)
optimizer = sgd(model.parameters(), lr=0.01)
optim_wrapper = optimwrapper(optimizer)

print(optimizer.param_groups[0]['lr'])  # -1.01
print(optimizer.param_groups[0]['momentum']) # 0
print(optim_wrapper.get_lr())  # {'lr': [0.01]}
print(optim_wrapper.get_momentum())  # {'momentum': [0]}
```

### 导出/加载状态字典

优化器封装和优化器一样，提供了 `state_dict` 和 `load_state_dict` 接口，用于导出/加载优化器状态，对于 `AmpOptimWrapper`，优化器封装还会额外导出混合精度训练相关的参数：

```python
import torch.nn as nn
from torch.optim import SGD
from mmengine.optim import OptimWrapper, AmpOptimWrapper

model = nn.Linear(1, 1)
optimizer = SGD(model.parameters(), lr=0.01)

optim_wapper = OptimWrapper(optimizer=optimizer)
amp_optim_wapper = AmpOptimWrapper(optimizer=optimizer)
# 导出状态字典
optim_state_dict = optim_wapper.state_dict()
amp_optim_state_dict = amp_optim_wapper.state_dict()

print(optim_state_dict)
# {'state': {}, 'param_groups': [{'lr': 0.01, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'params': [0, 1]}]}
print(amp_optim_state_dict)
# {'state': {}, 'param_groups': [{'lr': 0.01, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'params': [0, 1]}], 'loss_scaler': {}}
# amp 会额外存储 loss_scaler 字段
optim_wapper_new = OptimWrapper(optimizer=optimizer)
amp_optim_wapper_new = AmpOptimWrapper(optimizer=optimizer)

# 加载状态字典
amp_optim_wapper_new.load_state_dict(amp_optim_state_dict)
optim_wapper_new.load_state_dict(optim_state_dict)
```

### 使用多个优化器

考虑到生成对抗网络之类的算法通常需要使用多个优化器来训练生成器和判别器，因此优化器封装提供了优化器封装的容器类：`OptimWrapperDict` 来管理多个优化器封装。`OptimWrapperDict` 以字典的形式存储优化器封装，并允许用户像字典一样访问、遍历其中的元素，即优化器封装实例。

与普通的优化器封装不同，`OptimWrapperDict` 没有实现 `update_params`、 `optim_context`, `backward`、`step` 等方法，无法被直接用于训练模型。我们建议直接访问 `OptimWrapperDict` 管理的优化器实例，来实现参数更新逻辑。

你或许会好奇，既然 `OptimWrapperDict` 没有训练的功能，那为什么不直接使用 `dict` 来管理多个优化器。事实上，`OptimWrapperDict` 的核心功能是支持批量导出/加载所有优化器封装的状态字典；支持获取多个优化器封装的学习率、动量。如果没有 `OptimWrapperDict`，`MMEngine` 就需要在很多位置对优化器封装的类型做 `if else` 判断，以获取所有优化器封装的状态。

```python
from torch.optim import SGD
import torch.nn as nn

from mmengine.optim import OptimWrapper, OptimWrapperDict

gen = nn.Linear(1, 1)
disc = nn.Linear(1, 1)
optimizer_gen = SGD(gen.parameters(), lr=0.01)
optimizer_disc = SGD(disc.parameters(), lr=0.01)

optim_wapper_gen = OptimWrapper(optimizer=optimizer_gen)
optim_wapper_disc = OptimWrapper(optimizer=optimizer_disc)
optim_dict = OptimWrapperDict(gen=optim_wapper_gen, disc=optim_wapper_disc)

print(optim_dict.get_lr())  # {'gen.lr': [0.01], 'disc.lr': [0.01]}
print(optim_dict.get_momentum())  # {'gen.momentum': [0], 'disc.momentum': [0]}
```

如上例所示，`OptimWrapperDict` 可以非常方便的导出所有优化器封装的学习率和动量，同样的，优化器封装也能够导出/加载所有优化器封装的状态字典。

## 在[执行器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html)中配置优化器封装

### 简单配置

优化器封装需要接受 `optimizer` 参数，因此我们首先需要为优化器封装配置 `optimizer`。
MMEngine 会自动将 PyTorch 中的所有优化器都添加进 `OPTIMIZERS` 注册表中，用户可以用字典的形式来指定优化器，所有支持的优化器见 [PyTorch 优化器列表](https://pytorch.org/docs/stable/optim.html#algorithms)。

以配置一个 SGD 优化器封装为例：

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
```

这样我们就配置好了一个优化器类型为 SGD 的优化器封装，学习率、动量等参数如配置所示。考虑到 `OptimWrapper` 为标准的单精度训练，因此我们也可以不配置 `type` 字段：

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(optimizer=optimizer)
```

要想开启混合精度训练和梯度累加，需要将 `type` 切换成 `AmpOptimWrapper`，并指定 `accumulative_counts` 参数

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer, accumulative_counts=2)
```

### 进阶配置

PyTorch 的优化器支持对模型中的不同参数设置不同的超参数，例如对一个分类模型的骨干（backbone）和分类头（head）设置不同的学习率：

```python
from torch.optim import SGD
import torch.nn as nn

model = nn.ModuleDict(dict(backbone=nn.Linear(1, 1), head=nn.Linear(1, 1)))
SGD([
    {'params': model.backbone.parameters()},
    {'params': model.head.parameters(), 'lr': 1e-3}], lr=0.01, momentum=0.9)
```

上面的例子中，模型的骨干部分使用了 0.01 学习率，而模型的头部则使用了 1e-3 学习率。
用户可以将模型的不同部分参数和对应的超参组成一个字典的列表传给优化器，来实现对模型优化的细粒度调整。

在 MMEngine 中，我们通过优化器封装构造器（optimizer wrapper constructor），让用户能够直接通过设置优化器封装配置文件中的 `paramwise_cfg` 字段而非修改代码来实现对模型的不同部分设置不同的超参。

#### 为不同类型的参数设置不同的超参系数

MMEngine 提供的默认优化器封装构造器支持对模型中不同类型的参数设置不同的超参系数。
例如，我们可以在 `paramwise_cfg` 中设置 `norm_decay_mult=0` ，从而将正则化层（normalization layer）的权重（weight）和偏置（bias）的权值衰减系数（weight decay）设置为 0，
来实现 [Bag of Tricks](https://arxiv.org/abs/1812.01187) 论文中提到的不对正则化层进行权值衰减的技巧。

示例：

```python
optimizer = dict(
    optimizer=dict(
        type='SGD',
        lr=0.01,
        weight_decay=0.0001),
    paramwise_cfg=dict(norm_decay_mult=0))
```

除了可以对偏置的权重衰减进行配置外，MMEngine 的默认优化器构造器的 `paramwise_cfg` 还支持对更多不同类型的参数设置超参系数，支持的配置如下：

`bias_lr_mult`：偏置的学习率系数（不包括正则化层的偏置以及可变形卷积的 offset），默认值为 1

`bias_decay_mult`：偏置的权值衰减系数（不包括正则化层的偏置以及可变形卷积的 offset），默认值为 1

`norm_decay_mult`：正则化层权重和偏置的权值衰减系数，默认值为 1

`dwconv_decay_mult`：Depth-wise 卷积的权值衰减系数，默认值为 1

`bypass_duplicate`：是否跳过重复的参数，默认为 `False`

`dcn_offset_lr_mult`：可变形卷积（Deformable Convolution）的学习率系数，默认值为 1

#### 为模型不同部分的参数设置不同的超参系数

此外，与上文 PyTorch 的示例一样，在 MMEngine 中我们也同样可以对模型中的任意模块设置不同的超参，只需要在 `paramwise_cfg` 中设置 `custom_keys` 即可：

```python
optimizer = dict(
    optimizer=dict(
        type='SGD',
        lr=0.01,
        weight_decay=0.0001),
        paramwise_cfg=dict(
            custom_keys={
                'backbone.layer0': dict(lr_mult=0, decay_mult=0),
                'backbone': dict(lr_mult=1),
                'head': dict(lr_mult=0.1)
            }
        )
)
```

上面的配置文件实现了对模型的骨干第一层的学习率和权重衰减设置为 0，骨干的其余部分部分使用 0.01 学习率，而对模型的头部则使用 1e-3 学习率。

### 高级配置

与 MMEngine 中的其他模块一样，优化器封装构造器也同样由 [注册表](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/param_scheduler.html) 来管理。
用户可以实现自己的优化器构造策略来实现自定义的超参设置策略，并添加进 `OPTIM_WRAPPER_CONSTRUCTORS` 注册表中。

例如，我们想实现一个叫做`LayerDecayOptimWrapperConstructor`的优化器封装构造器，来实现对模型的不同深度的层自动设置递减的学习率。
我们可以通过继承 `DefaultOptimizerConstructor` 来实现这一策略，并将其添加进注册表中：

```python
@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LayerDecayOptimizerConstructor(DefaultOptimizerConstructor):
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        ...
```

然后将优化器配置文件中的 `constructor` 字段设置为类名来指定使用这个自定义的优化器构造器：

```python
optimizer = dict(
    optimizer=dict(
        type='SGD',
        lr=0.01,
        weight_decay=0.0001),
    constructor='LayerDecayOptimizerConstructor')
```

`MMEngine` 没有实现多优化器的优化器封装构造器，因此如果要使用 `OptimWrapperDict`，也需要实现自定义的构造器

```python
@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MultipleOptimizerConstructor:
    ...
```

### 在训练过程中调整超参

优化器中的超参数在构造时只能设置为一个定值，仅仅使用优化器封装，并不能在训练过程中调整学习率等参数。
在 MMEngine 中，我们实现了参数调度器（Parameter Scheduler），以便能够在训练过程中调整参数。关于参数调度器的用法请见[优化器参数调整策略](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/param_scheduler.html)
