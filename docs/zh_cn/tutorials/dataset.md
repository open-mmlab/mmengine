# 数据集（Dataset）与数据加载器（DataLoader）

```{hint}
如果你没有接触过 PyTorch 的数据集与数据加载器，我们推荐先浏览 [PyTorch 官方教程](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)以了解一些基本概念
```

数据集与数据加载器是 MMEngine 中训练流程的必要组件，它们的概念来源于 PyTorch，并且在含义上与 PyTorch 保持一致。通常来说，数据集定义了数据的总体数量、读取方式以及预处理，而数据加载器则在不同的设置下迭代地加载数据，如批次大小（`batch_size`）、随机乱序（`shuffle`）、并行（`num_workers`）等。数据集经过数据加载器封装后构成了数据源。在本篇教程中，我们将按照从外（数据加载器）到内（数据集）的顺序，逐步介绍它们在 MMEngine 执行器中的用法，并给出一些常用示例。读完本篇教程，你将会：

- 掌握如何在 MMEngine 的执行器中配置数据加载器
- 学会在配置文件中使用已有（如 `torchvision`）数据集
- 了解如何使用自己的数据集

## 数据加载器详解

在执行器（`Runner`）中，你可以分别配置以下 3 个参数来指定对应的数据加载器

- `train_dataloader`：在 `Runner.train()` 中被使用，为模型提供训练数据
- `val_dataloader`：在 `Runner.val()` 中被使用，也会在 `Runner.train()` 中每间隔一段时间被使用，用于模型的验证评测
- `test_dataloader`：在 `Runner.test()` 中被使用，用于模型的测试

MMEngine 完全支持 PyTorch 的原生 `DataLoader`，因此上述 3 个参数均可以直接传入构建好的 `DataLoader`，如[15分钟上手](../get_started/15_minutes.md)中的例子所示。同时，借助 MMEngine 的[注册机制](../advanced_tutorials/registry.md)，以上参数也可以传入 `dict`，如下面代码（以下简称例 1）所示。字典中的键值与 `DataLoader` 的构造参数一一对应。

```python
runner = Runner(
    train_dataloader=dict(
        batch_size=32,
        sampler=dict(
            type='DefaultSampler',
            shuffle=True),
        dataset=torchvision.datasets.CIFAR10(...),
        collate_fn=dict(type='default_collate')
    )
)
```

在这种情况下，数据加载器会在实际被用到时，在执行器内部被构建。

```{note}
关于 `DataLoader` 的更多可配置参数，你可以参考 [PyTorch API 文档](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
```

```{note}
如果你对于构建的具体细节感兴趣，你可以参考 [build_dataloader](mmengine.runner.Runner.build_dataloader)
```

细心的你可能会发现，例 1 **并非**直接由[15分钟上手](../get_started/15_minutes.md)中的示例代码简单修改而来。你可能本以为将 `DataLoader` 简单替换为 `dict` 就可以无缝切换，但遗憾的是，基于注册机制构建时 MMEngine 会有一些隐式的转换和约定。我们将介绍其中的不同点，以避免你使用配置文件时产生不必要的疑惑。

### sampler 与 shuffle

与 15 分钟上手明显不同，例 1 中我们添加了 `sampler` 参数，这是由于在 MMEngine 中我们要求通过 `dict` 传入的数据加载器的配置**必须包含 `sampler` 参数**。同时，`shuffle` 参数也从 `DataLoader` 中移除，这是由于在 PyTorch 中 **`sampler` 与 `shuffle` 参数是互斥的**，见 [PyTorch API 文档](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)。

```{note}
事实上，在 PyTorch 的实现中，`shuffle` 只是一个便利记号。当设置为 `True` 时 `DataLoader` 会自动在内部使用 `RandomSampler`
```

当考虑 `sampler` 时，例 1 代码**基本**可以认为等价于下面的代码块

```python
from mmengine.dataset import DefaultSampler

dataset = torchvision.datasets.CIFAR10(...)
sampler = DefaultSampler(dataset, shuffle=True)

runner = Runner(
    train_dataloader=DataLoader(
        batch_size=32,
        sampler=sampler,
        dataset=dataset,
        collate_fn=default_collate
    )
)
```

```{warning}
上述代码的等价性只有在：1）使用单进程训练，以及 2）没有配置执行器的 `randomness` 参数时成立。这是由于使用 `dict` 传入 `sampler` 时，执行器会保证它在分布式训练环境设置完成后才被惰性构造，并接收到正确的随机种子。这两点在手动构造时需要额外工作且极易出错。因此，上述的写法只是一个示意而非推荐写法。我们**强烈建议 `sampler` 以 `dict` 的形式传入**，让执行器处理构造顺序，以避免出现问题。
```

### DefaultSampler

上面例子可能会让你好奇：`DefaultSampler` 是什么，为什么要使用它，是否有其他选项？事实上，`DefaultSampler` 是 MMEngine 内置的一种采样器，它屏蔽了单进程训练与多进程训练的细节差异，使得单卡与多卡训练可以无缝切换。如果你有过使用 PyTorch `DistributedDataParallel` 的经验，你一定会对其中更换数据加载器的 `sampler` 参数有所印象。但在 MMEngine 中，这一细节通过 `DefaultSampler` 而被屏蔽。

除了 `Dataset` 本身之外，`DefaultSampler` 还支持以下参数配置：

- `shuffle` 设置为 `True` 时会打乱数据集的读取顺序
- `seed` 打乱数据集所用的随机种子，通常不需要在此手动设置，会从 `Runner` 的 `randomness` 入参中读取
- `round_up` 设置为 `True` 时，与 PyTorch `DataLoader` 中设置 `drop_last=False` 行为一致。如果你在迁移 PyTorch 的项目，你可能需要注意这一点。

```{note}
更多关于 `DefaultSampler` 的内容可以参考 [API 文档](mmengine.dataset.DefaultSampler)
```

`DefaultSampler` 适用于绝大部分情况，并且我们保证在执行器中使用它时，随机数等容易出错的细节都被正确地处理，防止你陷入多进程训练的常见陷阱。如果你想要使用基于迭代次数 (iteration-based) 的训练流程，你也许会对 [InfiniteSampler](mmengine.dataset.InfiniteSampler) 感兴趣。如果你有更多的进阶需求，你可能会想要参考上述两个内置 `sampler` 的代码，实现一个自定义的 `sampler` 并注册到 `DATA_SAMPLERS` 根注册器中。

```python
@DATA_SAMPLERS.register_module()
class MySampler(Sampler):
    pass

runner = Runner(
    train_dataloader=dict(
        sampler=dict(type='MySampler'),
        ...
    )
)
```

### 不起眼的 collate_fn

PyTorch 的 `DataLoader` 中，`collate_fn` 这一参数常常被使用者忽略，但在 MMEngine 中你需要额外注意：当你传入 `dict` 来构造数据加载器时，MMEngine 会默认使用内置的 [pseudo_collate](mmengine.dataset.pseudo_collate)，这一点明显区别于 PyTorch 默认的 [default_collate](https://pytorch.org/docs/stable/data.html#torch.utils.data.default_collate)。因此，当你迁移 PyTorch 项目时，需要在配置文件中手动指明 `collate_fn` 以保持行为一致。

```{note}
MMEngine 中使用 `pseudo_collate` 作为默认值，主要是由于历史兼容性原因，你可以不必过于深究，只需了解并避免错误使用即可。
```

MMengine 中提供了 2 种内置的 `collate_fn`：

- `pseudo_collate`，缺省时的默认参数。它不会将数据沿着 `batch` 的维度合并。详细说明可以参考 [pseudo_collate](mmengine.dataset.pseudo_collate)
- `default_collate`，与 PyTorch 中的 `default_collate` 行为几乎完全一致，会将数据转化为 `Tensor` 并沿着 `batch` 维度合并。一些细微不同和详细说明可以参考 [default_collate](mmengine.dataset.default_collate)

如果你想要使用自定义的 `collate_fn`，你也可以将它注册到 `FUNCTIONS` 根注册器中来使用

```python
@FUNCTIONS.register_module()
def my_collate_func(data_batch: Sequence) -> Any:
    pass

runner = Runner(
    train_dataloader=dict(
        ...
        collate_fn=dict(type='my_collate_func')
    )
)
```

## 数据集详解

数据集通常定义了数据的数量、读取方式与预处理，并作为参数传递给数据加载器供后者分批次加载。由于我们使用了 PyTorch 的 `DataLoader`，因此数据集也自然与 PyTorch `Dataset` 完全兼容。同时得益于注册机制，当数据加载器使用 `dict` 在执行器内部构建时，`dataset` 参数也可以使用 `dict` 传入并在内部被构建。这一点使得编写配置文件成为可能。

### 使用 torchvision 数据集

`torchvision` 中提供了丰富的公开数据集，它们都可以在 MMEngine 中直接使用，例如 [15 分钟上手](../get_started/15_minutes.md)中的示例代码就使用了其中的 `Cifar10` 数据集，并且使用了 `torchvision` 中内置的数据预处理模块。

但是，当需要将上述示例转换为配置文件时，你需要对 `torchvision` 中的数据集进行额外的注册。如果你同时用到了 `torchvision` 中的数据预处理模块，那么你也需要编写额外代码来对它们进行注册和构建。下面我们将给出一个等效的例子来展示如何做到这一点。

```python
import torchvision.transforms as tvt
from mmengine.registry import DATASETS, TRANSFORMS
from mmengine.dataset.base_dataset import Compose

# 注册 torchvision 的 CIFAR10 数据集
# 数据预处理也需要在此一起构建
@DATASETS.register_module(name='Cifar10', force=False)
def build_torchvision_cifar10(transform=None, **kwargs):
    if isinstance(transform, dict):
        transform = [transform]
    if isinstance(transform, (list, tuple)):
        transform = Compose(transform)
    return torchvision.datasets.CIFAR10(**kwargs, transform=transform)

# 注册 torchvision 中用到的数据预处理模块
DATA_TRANSFORMS.register_module('RandomCrop', module=tvt.RandomCrop)
DATA_TRANSFORMS.register_module('RandomHorizontalFlip', module=tvt.RandomHorizontalFlip)
DATA_TRANSFORMS.register_module('ToTensor', module=tvt.ToTensor)
DATA_TRANSFORMS.register_module('Normalize', module=tvt.Normalize)

# 在 Runner 中使用
runner = Runner(
    train_dataloader=dict(
        batch_size=32,
        sampler=dict(
            type='DefaultSampler',
            shuffle=True),
        dataset=dict(type='Cifar10',
            root='data/cifar10',
            train=True,
            download=True,
            transform=[
                dict(type='RandomCrop', size=32, padding=4),
                dict(type='RandomHorizontalFlip'),
                dict(type='ToTensor'),
                dict(type='Normalize', **norm_cfg)])
    )
)
```

```{note}
上述例子中大量使用了[注册机制](../advanced_tutorials/registry.md)，并且用到了 MMEngine 中的 [Compose](mmengine.dataset.Compose)。如果你急需在配置文件中使用 `torchvision` 数据集，你可以参考上述代码并略作修改。但我们更加推荐你有需要时在下游库（如 [MMDet](https://github.com/open-mmlab/mmdetection) 和 [MMPretrain](https://github.com/open-mmlab/mmpretrain) 等）中寻找对应的数据集实现，从而获得更好的使用体验。
```

### 自定义数据集

你可以像使用 PyTorch 一样，自由地定义自己的数据集，或将之前 PyTorch 项目中的数据集拷贝过来。如果你想要了解如何自定义数据集，可以参考 [PyTorch 官方教程](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)

### 使用 MMEngine 的数据集基类

除了直接使用 PyTorch 的 `Dataset` 来自定义数据集之外，你也可以使用 MMEngine 内置的 `BaseDataset`，参考[数据集基类](../advanced_tutorials/basedataset.md)文档。它对标注文件的格式做了一些约定，使得数据接口更加统一、多任务训练更加便捷。同时，数据集基类也可以轻松地搭配内置的[数据变换](../advanced_tutorials/data_transform.md)使用，减轻你从头搭建训练流程的工作量。

目前，`BaseDataset` 已经在 OpenMMLab 2.0 系列的下游仓库中被广泛使用。
