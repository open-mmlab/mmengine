# 优化器（Optimizer）

在模型训练过程中，我们需要使用优化算法对模型的参数进行优化。在PyTorch的torch.optim中包含了各种优化算法的实现，这些优化算法的类被称为优化器。
在 PyTorch 中，用户可以通过构建一个优化器对象来优化模型的参数，下面是一个简单的例子：

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

关于 PyTorch 中优化器的详细介绍可以参考[PyTorch优化器文档](https://pytorch.org/docs/stable/optim.html#)

MMEngine 中支持所有的 PyTorch 优化器，用户可以直接构建 PyTorch 优化器对象并将它传给执行器（Runner）。
和 PyTorch 文档中所给示例不同，MMEngine 中通常不需要手动来实现训练循环以及调用` optimizer.step()`，执行器会自动对损失函数进行反向传播并调用优化器的 `step` 函数。

同时，我们也支持通过配置文件从注册器中构建优化器。更进一步的，我们提供了优化器构造器（optimizer constructor）来对模型的优化进行更细粒度的调整。

## 使用配置文件构建优化器

MMEngine 会自动将 PyTorch 中的所有优化器都添加进注册表中，用户可以通过设置配置文件中的 `optimizer` 字段来指定优化器，所有支持的优化器见[PyTorch优化器列表](https://pytorch.org/docs/stable/optim.html#algorithms)。

以配置一个 SGD 优化器为例：

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
```

我们只需要指定 `optimizer` 字段中的 `type` 为 SGD， 并设置学习率等参数，执行器会根据此字段以及执行器中的模型参数自动构建优化器。

## 细粒度调整模型超参

PyTorch 的优化器支持对模型中的不同参数设置不同的超参数，例如对一个分类模型的骨干和分类头设置不同的学习率：

```python
optim.SGD([
                {'params': model.backbone.parameters()},
                {'params': model.head.parameters(), 'lr': 1e-3}
            ], lr=0.01, momentum=0.9)
```

上面的例子中，模型的骨干部分使用了 0.01 学习率，而模型的头部则使用了 1e-3 学习率。
用户可以将模型的不同部分参数和对应的超参组成一个字典的列表传给优化器，来实现对模型优化的细粒度调整。

在 MMEngine 中，我们通过优化器构造器（optimizer constructor），让用户能够直接通过设置优化器配置文件中的 `paramwise_cfg` 字段而非修改代码来实现对模型的不同部分设置不同的超参。

一个简单的例子：

```python
optimizer = dict(type='SGD',
                 lr=0.01,
                 weight_decay=0.0001,
                 paramwise_cfg=dict(bias_decay_mult=0))
```

我们可以在 `paramwise_cfg` 中设置 `norm_decay_mult=0` ，来实现 [Bag of Tricks](https://arxiv.org/abs/1812.01187) 论文中提到的不对 bias 进行权值衰减（weight decay）的技巧。

除了可以对 bias 的权重衰减进行配置外，MMEngine 的默认优化器构造器的 `paramwise_cfg` 还支持更多的配置，支持的配置如下：

`bias_lr_mult`：bias 的学习率系数，默认值为 1

`bias_decay_mult`：bias 的权值衰减系数，默认值为 1

`norm_decay_mult`：bias 的权值衰减系数，默认值为 1

`dwconv_decay_mult`：Depth-wise 卷积的权值衰减系数，默认值为 1

`bypass_duplicate`：是否跳过重复的参数，默认为 `False`

`dcn_offset_lr_mult`：可变形卷积（Deformable Convolution）的学习率系数，默认值为 1

此外，与上文 PyTorch 的示例一样，在 MMEngine 中我们也同样可以对模型中的任意模块设置不同的超参，只需要在 `paramwise_cfg` 中设置 `custom_keys` 即可：

```python
optimizer = dict(type='SGD',
                 lr=0.01,
                 weight_decay=0.0001,
                 paramwise_cfg=dict(
                     custom_keys={
                        'backbone': dict(lr_mult=1),
                        'head': dict(lr_mult=0.1),
                        }
                 ))
```

上面的配置文件也同样实现了对模型的骨干部分使用 0.01 学习率，而对模型的头部则使用 1e-3 学习率。

## 在训练过程中调整超参

优化器中的超参数在构造时只能设置为一个定值，仅仅使用优化器，并不能在训练过程中调整学习率等参数。
在 MMEngine 中，我们实现了参数调度器（Parameter Scheduler），以便能够在训练过程中调整参数。关于参数调度器的用法请见[优化器参数调整策略](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/param_scheduler.html)
