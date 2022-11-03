# 数据集（Dataset）与数据加载器（DataLoader）

```{hint}
如果你没有接触过 PyTorch 的数据集与数据加载器，我们推荐先浏览 [PyTorch 官方教程](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)以了解一些基本概念
```

数据集与数据加载器是 MMEngine 中训练流程的必要组件，它们的概念来源于 PyTorch，并且在含义上与 PyTorch 保持一致。通常来说，数据集定义了数据的总体数量、获取方式以及预处理，而数据加载器则在不同的设置下（如 `batch_size`、随机乱序、多进程）迭代地加载数据。两者共同构成了数据源。在本篇教程中，我们将按照从外（数据加载器）到内（数据集）的顺序，逐步介绍它们在 MMEngine 执行器中的用法，并给出一些常用示例。读完本篇教程，你将会：

- 掌握如何在 MMEngine 的执行器中配置数据加载器
- 了解如何使用自己的数据集
- 学会在配置文件中使用已有（如 `torchvision`）数据集

## 数据加载器详解

在执行器（`Runner`）中，你可以分别配置以下 3 个参数来指定对应的数据加载器

- `train_dataloader`：在 `Runner.train()` 中被使用，为模型提供训练数据
- `val_dataloader`：在 `Runner.val()` 中被使用，也会在 `Runner.train()` 中每间隔一段时间被使用，用于模型的验证评测
- `test_dataloader`：在 `Runner.test()` 中被使用，用于模型的测试

MMEngine 完全支持 PyTorch 的原生 `DataLoader`，因此上述 3 个参数均可以直接传入构建好的 `DataLoader`，如[15分钟上手](../get_started/15_minutes.md)中的例子所示。同时，借助 MMEngine 的[注册机制](../advanced_tutorials/registry.md)，以上参数也可以传入 `dict`，如下面代码所示。在这种情况下，数据加载器会在实际被用到时、在执行器内部被构建。

```python
runner = Runner(
    train_dataloader=dict(
        batch_size=32,
        sampler=dict(
            type='DefaultSampler',
            shuffle=True),
        dataset=torchvision.datasets.CIFAR10(...),
        collate_fn=default_collate
    )
)
```

细心的你可能会发现，上面的示例代码**并非**直接由[15分钟上手](../get_started/15_minutes.md)中的示例代码简单修改而来。你可能本以为将 `DataLoader` 简单替换为 `dict` 就可以无缝切换，但遗憾的是，基于注册机制构建时 MMEngine 会有一些隐式的转换和约定。我们将介绍其中的不同点，以便于你使用配置文件时产生不必要的疑惑。

### 从 sampler 开始讲起

sampler 是什么
MMEngine 的 DefaultSampler
DataLoader.drop_last \<--> DefaultSampler.round_up
shuffle
分布式与 seed

### 不起眼的 collate_fn

default_collate \<--> pseudo_collate

## 数据集详解

### 自定义数据集

### 使用 torchvision 数据集

### 使用 MMEngine 的数据集基类

## 在配置文件中使用 torchvision 数据集
