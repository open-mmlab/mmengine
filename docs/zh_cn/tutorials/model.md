# 模型（Model）

## Runner 与 model

在 [Runner 教程的基本数据流](./runner.md#基本数据流)中我们提到，DataLoader、model 和 evaluator 之间的数据流通遵循了一些规则，我们先来回顾一下基本数据流的伪代码：

```python
# 训练过程
for data_batch in train_dataloader:
    data_batch = model.data_preprocessor(data_batch, training=True)
    if isinstance(data_batch, dict):
        losses = model(**data_batch, mode='loss')
    elif isinstance(data_batch, (list, tuple)):
        losses = model(*data_batch, mode='loss')
    else:
        raise TypeError()
# 验证过程
for data_batch in val_dataloader:
    data_batch = model.data_preprocessor(data_batch, training=False)
    if isinstance(data_batch, dict):
        outputs = model(**data_batch, mode='predict')
    elif isinstance(data_batch, (list, tuple)):
        outputs = model(**data_batch, mode='predict')
    else:
        raise TypeError()
    evaluator.process(data_samples=outputs, data_batch=data_batch)
metrics = evaluator.evaluate(len(val_dataloader.dataset))
```

在 Runner 的教程中，我们简单介绍了模型和前后组件之间的数据流通关系，提到了 `data_preprocessor` 的概念，对 model 有了一定的了解。然而在 Runner 实际运行的过程中，模型的功能和调用关系，其复杂程度远超上述伪代码。为了让你能够不感知模型和外部组件的复杂关系，进而聚焦精力到算法本身，我们设计了 [BaseModel](mmengine.model.BaseModel)。大多数情况下你只需要让 model 继承 `BaseModel`，并按照要求实现 `forward` 接口，就能完成训练、测试、验证的逻辑。

在继续阅读模型教程之前，我们先抛出两个问题，希望你在阅读完 model 教程后能够找到相应的答案：

1. 我们在什么位置更新模型参数？如果我有一些非常复杂的参数更新逻辑，又该如何实现？
2. 为什么要有 data_preprocessor 的概念？它又可以实现哪些功能？

## 接口约定

在训练深度学习任务时，我们通常需要定义一个模型来实现算法的主体。在基于 MMEngine 开发时，定义的模型由执行器管理，且需要实现 `train_step`、`val_step` 和 `test_step` 方法。
对于检测、识别、分割一类的深度学习任务，上述方法通常为标准的流程，例如在 `train_step` 里更新参数，返回损失；`val_step` 和 `test_step` 返回预测结果。因此 MMEngine 抽象出模型基类 [BaseModel](mmengine.model.BaseModel)，实现了上述接口的标准流程。

得益于 `BaseModel` 我们只需要让模型继承自模型基类，并按照一定的规范实现 `forward`，就能让模型在执行器中运行起来。

```{note}
模型基类继承自[模块基类](../advanced_tutorials/initialize.md)，能够通过配置 `init_cfg` 灵活地选择初始化方式。
```

[**forward**](mmengine.model.BaseModel.forward): `forward` 的入参需通常需要和 [DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) 的输出保持一致 (自定义[数据预处理器](#数据预处理器datapreprocessor)除外)，如果 `DataLoader` 返回元组类型的数据 `data`，`forward` 需要能够接受 `*data` 的解包后的参数；如果返回字典类型的数据 `data`，`forward` 需要能够接受 `**data` 解包后的参数。 `mode` 参数用于控制 forward 的返回结果：

- `mode='loss'`：`loss` 模式通常在训练阶段启用，并返回一个损失字典。损失字典的 key-value 分别为损失名和可微的 `torch.Tensor`。字典中记录的损失会被用于更新参数和记录日志。模型基类会在 `train_step` 方法中调用该模式的 `forward`。
- `mode='predict'`： `predict` 模式通常在验证、测试阶段启用，并返回列表/元组形式的预测结果，预测结果需要和 [process](mmengine.evaluator.Evaluator.process) 接口的参数相匹配。OpenMMLab 系列算法对 `predict` 模式的输出有着更加严格的约定，需要输出列表形式的[数据元素](../advanced_tutorials/data_element.md)。模型基类会在 `val_step`，`test_step` 方法中调用该模式的 `forward`。
- `mode='tensor'`：`tensor` 和 `predict` 模式均返回模型的前向推理结果，区别在于 `tensor` 模式下，`forward` 会返回未经后处理的张量，例如返回未经非极大值抑制（nms）处理的检测结果，返回未经 `argmax` 处理的分类结果。我们可以基于 `tensor` 模式的结果进行自定义的后处理。

[**train_step**](mmengine.model.BaseModel.train_step): 执行 `forward` 方法的 `loss` 分支，得到损失字典。模型基类基于[优化器封装](./optim_wrapper.md) 实现了标准的梯度计算、参数更新、梯度清零流程。其等效伪代码如下：

```python
def train_step(self, data, optim_wrapper):
    data = self.data_preprocessor(data, training=True)  # 按下不表，详见数据与处理器一节
    loss = self(**data, mode='loss')  # loss 模式，返回损失字典，假设 data 是字典，使用 ** 进行解析。事实上 train_step 兼容 tuple 和 dict 类型的输入。
    parsed_losses, log_vars = self.parse_losses() # 解析损失字典，返回可以 backward 的损失以及可以被日志记录的损失
    optim_wrapper.update_params(parsed_losses)  # 更新参数
    return log_vars
```

[**val_step**](mmengine.model.BaseModel.val_step): 执行 `forward` 方法的 `predict` 分支，返回预测结果：

```python
def val_step(self, data, optim_wrapper):
    data = self.data_preprocessor(data, training=False)
    outputs = self(**data, mode='predict') # 预测模式，返回预测结果
    return outputs
```

[**test_step**](mmengine.model.BaseModel.test_step): 同 `val_step`

看到这我们就可以给出一份 **基本数据流伪代码 plus**：

```python
# 训练过程
for data_batch in train_dataloader:
    loss_dict = model.train_step(data_batch)
# 验证过程
for data_batch in val_dataloader:
    preds = model.test_step(data_batch)
    evaluator.process(data_samples=outputs, data_batch=data_batch)
metrics = evaluator.evaluate(len(val_dataloader.dataset))
```

没错，抛开 Hook 不谈，[loop](mmengine.runner.EpochBasedTrainLoop) 调用 model 的过程和上述代码一模一样！看到这，我们再回过头去看 [15 分钟上手 MMEngine](../get_started/15_minutes.md) 里的模型定义部分，就有一种看山不是山的感觉：

```python
import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel

class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels

    # 下面的 3 个方法已在 BaseModel 实现，这里列出是为了
    # 解释调用过程
    def train_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        loss = self(*data, mode='loss')  # CIFAR10 返回 tuple，因此用 * 解包
        parsed_losses, log_vars = self.parse_losses()
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        outputs = self(*data, mode='predict')
        return outputs

    def test_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        outputs = self(*data, mode='predict')
        return outputs
```

看到这里，相信你对数据流有了更加深刻的理解，也能够回答 [Runner 与 model](#runner-与-model) 里提到的第一个问题：

`BaseModel.train_step` 里实现了默认的参数更新逻辑，如果我们想实现自定义的参数更新流程，可以重写 `train_step` 方法。但是需要注意的是，我们需要保证 `train_step` 最后能够返回损失字典。

## 数据预处理器（DataPreprocessor）

如果你的电脑配有 GPU（或其他能够加速训练的硬件，如 MPS、IPU 等），并且运行了 [15 分钟上手 MMEngine](../get_started/15_minutes.md) 的代码示例，你会发现程序是在 GPU 上运行的，那么 `MMEngine` 是在何时把数据和模型从 CPU 搬运到 GPU 的呢？

事实上，执行器会在构造阶段将模型搬运到指定设备，而数据则会在上一节提到的 `self.data_preprocessor` 这一行搬运到指定设备，进一步将处理好的数据传给模型。看到这里相信你会疑惑：

1. `MMResNet50` 并没有配置 `data_preprocessor`，为什么却可以访问到 `data_preprocessor`，并且把数据搬运到 GPU？

2. 为什么不直接在模型里调用 `data.to(device)` 搬运数据，而需要有 `data_preprocessor` 这一层抽象？它又能实现哪些功能？

首先回答第一个问题：`MMResNet50` 继承了 `BaseModel`。在执行 `super().__init__` 时，如果不传入任何参数，会构造一个默认的 `BaseDataPreprocessor`，其等效简易实现如下：

```python
class BaseDataPreprocessor(nn.Module):
    def forward(self, data, training=True):  # 先忽略 training 参数
        # 假设 data 是 CIFAR10 返回的 tuple 类型数据，事实上
        # BaseDataPreprocessor 可以处理任意类型的数
        # BaseDataPreprocessor 同样可以把数据搬运到多种设备，这边方便
        # 起见写成 .cuda()
        return tuple(_data.cuda() for _data in data)
```

`BaseDataPreprocessor` 会在训练过程中，将各种类型的数据搬运到指定设备。

在回答第二个问题之前，我们不妨先再思考几个问题

1. 数据归一化操作应该在哪里进行，[transform](../advanced_tutorials/data_transform.md) 还是 model？

   听上去好像都挺合理，放在 transform 里可以利用 Dataloader 的多进程加速，放在 model 里可以搬运到 GPU 上，利用GPU 资源加速归一化。然而在我们纠结 CPU 归一化快还是 GPU 归一化快的时候，CPU 到 GPU 的数据搬运耗时相较于前者，可算的上是“降维打击”。
   事实上对于归一化这类计算量较低的操作，其耗时会远低于数据搬运，因此优化数据搬运的效率就显得更加重要。设想一下，如果我能够在数据仍处于 uint8 时、归一化之前将其搬运到指定设备上（归一化后的 float 型数据大小是 unit8 的 4 倍），就能降低带宽，大大提升数据搬运的效率。这种“滞后”归一化的行为，也是我们设计数据预处理器（data preprocessor） 的主要原因之一。数据预处理器会先搬运数据，再做归一化，提升数据搬运的效率。

2. 我们应该如何实现 MixUp、Mosaic 一类的数据增强？

   尽管看上去 MixUp 和 Mosaic 只是一种特殊的数据变换，按理说应该在 transform 里实现。考虑到这两种增强会涉及到“将多张图片融合成一张图片”的操作，在 transform 里实现他们的难度就会很大，因为目前 transform 的范式是对一张图片做各种增强，我们很难在一个 transform 里去额外读取其他图片（transform 里无法访问到 dataset）。然而如果基于 Dataloader 采样得到的 `batch_data` 去实现 Mosaic 或者 Mixup，事情就会变得非常简单，因为这个时候我们能够同时访问多张图片，可以轻而易举的完成图片融合的操作：

   ```python
   class MixUpDataPreprocessor(nn.Module):
       def __init__(self, num_class, alpha):
           self.alpha = alpha

       def forward(self, data, training=True):
           data = tuple(_data.cuda() for _data in data)
           # 验证阶段无需进行 MixUp 数据增强
           if not training:
               return data

           label = F.one_hot(label)  # label 转 onehot 编码
           batch_size = len(label)
           index = torch.randperm(batch_size)  # 计算用于叠加的图片数
           img, label = data
           lam = np.random.beta(self.alpha, self.alpha)  # 融合因子

           # 原图和标签的 MixUp.
           img = lam * img + (1 - lam) * img[index, :]
           label = lam * batch_scores + (1 - lam) * batch_scores[index, :]
           # 由于此时返回的是 onehot 编码的 label，model 的 forward 也需要做相应调整
           return tuple(img, label)
   ```

   因此，除了数据搬运和归一化，`data_preprocessor` 另一大功能就是数据批增强（BatchAugmentation）。数据预处理器的模块化也能帮助我们实现算法和数据增强之间的自由组合。

3. 如果 DataLoader 的输出和模型的输入类型不匹配怎么办，是修改 DataLoader 还是修改模型接口？

   答案是都不合适。理想的解决方案是我们能够在不破坏模型和数据已有接口的情况下完成适配。这个时候数据预处理器也能承担类型转换的工作，例如将传入的 data 从 `tuple` 转换成指定字段的 `dict`。

看到这里，相信你已经能够理解数据预处理器存在的合理性，并且也能够自信地回答教程最初提出的两个问题！但是你可能还会疑惑 `train_step` 接口中传入的 `optim_wrapper` 又是什么，`test_step` 和 `val_step` 返回的结果和 evaluator 又有怎样的关系，这些问题会在[模型精度评测教程](./evaluation.md)和[优化器封装](./optim_wrapper.md)得到解答。
