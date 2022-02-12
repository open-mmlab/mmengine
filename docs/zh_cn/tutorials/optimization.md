# 优化（Optimization）

## 如何调整学习率

与 PyTorch 的 `torch.optim.lr_scheduler` 一样，MMEngine 提供了丰富的参数调度器（parameter schedulers）用来调整优化器的参数组（parameter groups）中的学习率。
通过 MMEngine 中的调度器，你可以基于 epoch 或者 iteration 调整学习率，也可以将多个调度器进行组合，或者自定义你自己的调度器策略。

### 使用单一的学习率调度器

在 MMEngine 中，如果你只需要使用一个学习率调度器, 那么只需要设置配置文件中的 `scheduler` 字段即可，例如:

```python
scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1)
```

这样我们就设置了一个按轮次（epoch）来进行调整学习率的调度器，第 8, 11 轮次时，学习率将会被调整为原来的 0.1 倍。

### 学习率调整频率

MMEngine 支持两种不同的学习率调整频率：按轮次（epoch） 调整学习率和按迭代（iteration）调整学习率。可以通过设置调度器的 `by_epoch` 参数来指定学习率调整频率。
下面以 `MultiStepLR` 调度器为例来说明这两种调度频率的区别。

按轮次调整学习率的例子:

```python
scheduler = dict(type='MultiStepLR',
                 by_epoch=True,  # 按轮次调整学习率
                 milestones=[8, 11],  # 学习率在第 8 和 11 轮时衰减
                 gamma=0.1)
```

按迭代调整学习率的例子:

```python
scheduler = dict(type='MultiStepLR',
                 by_epoch=False,  # 按迭代调整学习率
                 milestones=[60000, 80000],  # 学习率在第 60000 和 80000 次迭代时衰减
                 gamma=0.1)
```

### 组合多个学习率调度器（例如学习率预热）

MMEngine 支持组合多个调度器一起使用, 只需要将配置文件中的 `scheduler` 字段设置为一系列调度器配置的列表即可。
以最常见的学习率预热为例：

```python
scheduler = [
    # 线性学习率预热调度器
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=False,  # 按迭代更新学习率
         begin=0,
         end=500),  # 预热前 500 次迭代

    # 主学习率调度器
    dict(type='MultiStepLR',
         by_epoch=True,  # 按轮次更新学习率
         milestones=[8, 11],
         gamma=0.1)
]
```

通过设置线性学习率调度器的 `begin` 和 `end` 参数，可以指定前500次迭代进行线性的学习率预热（关于`begin` 和 `end` 参数的含义可见下文的“调度器生效区间”），之后使用 `MultiStepLR` 调度器来按轮次调整学习率。

### 调度器生效区间

与 PyTorch 的调度器不同的是，MMEngine 中的调度器可以指定生效区间。生效区间通常只在多个调度器组合时才需要去设置，如果没有这样的需求，可以省略这部分设置。

通过设置 `begin` 和 `end` 参数，可以指定调度器在指定区间内生效。生效区间是前闭后开的，即 [begin, end)。当 `by_epoch` 参数为 True 时，生效区间是按轮次（epoch）来指定的，当 `by_epoch` 参数为 False 时，生效区间是按迭代（iteration）来指定的。


示例:

```python
scheduler = [
    # 在 [0, 1000) 迭代时使用线性学习率
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=1000),
    # 在 [1000, 9000) 迭代时使用余弦学习率
    dict(type='CosineAnnealingLR',
         T_max=8000,
         by_epoch=False,
         begin=1000,
         end=9000)
]
```

在大部分组合使用调度器的情况下，我们会避免调度器的生效区间发生重叠以在训练的不同阶段使用不同的学习率调度策略。

和 PyTorch 的调度器一样，MMEngine 同样也允许多组调度器叠加使用。当叠加使用时，学习率的调整会按照调度器配置文件中的顺序触发。我们推荐使用[学习率可视化工具]()来可视化叠加后的学习率，以避免学习率的调整与预期不符。


## 如何调整动量（Momentum）

和学习率一样, 动量也是优化器参数组中一组可以调度的参数。 动量调度器（momentum scheduler）的使用方法和学习率调度器完全一样。同样也只需要将动量调度器的配置填入配置文件中的 `scheduler` 字段即可。

示例:

```python
scheduler = [
    dict(type='LinearMomentum',
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=1000)
]
```

## 如何调整自定义参数

MMEngine 还提供了一组通用的参数调度器（parameter schedulers）用于调度优化器的 `param_groups` 中的其他参数。你可以通过设置参数调度器的 `param_name` 变量来选择你想要调度的参数。

下面是一个通过自定义参数名来调度的例子:

```python
scheduler = [
    dict(type='LinearParamScheduler',
         param_name='lr',  # 调度 `optimizer.param_groups` 中名为 'lr' 的变量
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=1000)
]
```

这里设置的参数名是`lr`，因此这个调度器的作用等同于学习率调度器。当然，你也可以设置 `optimizer.param_groups` 中的其他参数名进行调度。
