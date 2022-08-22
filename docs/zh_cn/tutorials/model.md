# 模型

在训练深度学习任务时，我们通常需要定义一个模型来实现算法的主体。在基于 MMEngine 开发时，模型由[执行器](./runner.md)管理，需要实现 `train_step`，`val_step` 和 `test_step` 方法。

对于检测、识别、分割一类的深度学习任务，上述方法通常为标准的流程，例如在 `train_step` 里更新参数，返回损失；`val_step` 和 `test_step` 返回预测结果。因此 MMEngine 抽象出模型基类 [BaseModel](mmengine.model.BaseModel)，实现了上述接口的标准流程。我们只需要让模型继承自模型基类，并按照一定的规范实现 `forward`，就能让模型在执行器中运行起来。

模型基类继承自[模块基类](./initialize.md)，能够通过配置 `init_cfg` 灵活的选择初始化方式。

## 接口约定

[forward](mmengine.model.BaseModel.forward): `forward` 的入参需要和 [DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) 的输出保持一致 (自定义[数据处理器](#数据处理器datapreprocessor)除外)，如果 `DataLoader` 返回元组类型的数据 `data`，`forward` 需要能够接受 `*data` 的解包后的参数；如果返回字典类型的数据 `data`，`forward` 需要能够接受 `**data` 解包后的参数。 `mode` 参数用于控制 forward 的返回结果：

- `mode='loss'`：`loss` 模式通常在训练阶段启用，并返回一个损失字典。损失字典的 key-value 分别为损失名和可微的 `torch.Tensor`。字典中记录的损失会被用于更新参数和记录日志。模型基类会在 `train_step` 方法中调用该模式的 `forward`。
- `mode='predict'`： `predict` 模式通常在验证、测试阶段启用，并返回列表/元组形式的预测结果，预测结果需要和 [process](mmengine.evaluator.Evaluator.process) 接口的参数相匹配。OpenMMLab 系列算法对 `predict` 模式的输出有着更加严格的约定，需要输出列表形式的[数据元素](./data_element.md)。模型基类会在 `val_step`，`test_step` 方法中调用该模式的 `forward`。
- `mode='tensor'`：`tensor` 和 `predict` 均用于返回模型的预测结果，区别在于 OpenMMLab 系列的算法库要求 `predict` 模式返回数据元素列表，而 `tensor` 模式则返回 `torch.Tensor` 类型的结果。`tensor` 模式为 `forward` 的默认模式，如果我们想获取一张或一个批次（batch）图片的推理结果，可以直接调用 `model(inputs)` 来获取预测结果。

[train_step](mmengine.model.BaseModel.train_step): 调用 `loss` 模式的 `forward` 接口，得到损失字典。模型基类基于[优化器封装](.optim_wrapper.md) 实现了标准的梯度计算、参数更新、梯度清零流程。

[val_step](mmengine.model.BaseModel.val_step): 调用 `predict` 模式的 `forward`，返回预测结果，预测结果会被进一步传给[钩子（Hook）](./hook.md)的 `after_train_iter` 和 `after_val_iter` 接口。

[test_step](mmengine.model.BaseModel.test_step): 同 `val_step`，预测结果会被进一步传给 `after_test_iter` 接口。

基于上述接口约定，我们定义了继承自模型基类的 `NeuralNetwork`，配合执行器来训练 `FashionMNIST`：

```python
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from mmengine.model import BaseModel
from mmengine.evaluator import BaseMetric
from mmengine import Runner


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(dataset=training_data, batch_size=64)
test_dataloader = DataLoader(dataset=test_data, batch_size=64)


class NeuralNetwork(BaseModel):
    def __init__(self, data_preprocessor=None):
        super(NeuralNetwork, self).__init__(data_preprocessor)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, img, label, mode='tensor'):
        x = self.flatten(img)
        pred = self.linear_relu_stack(x)
        loss = self.loss(pred, label)
        if mode == 'loss':
            return dict(loss=loss)
        else:
            return pred.argmax(1), loss.item()


class FashionMnistMetric(BaseMetric):
    def process(self, data, preds) -> None:
        self.results.append(((data[1] == preds[0].cpu()).sum(), preds[1], len(preds[0])))

    def compute_metrics(self, results):
        correct, loss, batch_size = zip(*results)
        test_loss, correct = sum(loss) / len(self.results), sum(correct) / sum(batch_size)
        return dict(Accuracy=correct, Avg_loss=test_loss)


runner = Runner(
    model=NeuralNetwork(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=1e-3)),
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_cfg=dict(fp16=True),
    val_dataloader=test_dataloader,
    val_evaluator=dict(metrics=FashionMnistMetric()))
runner.train()
```

相比于 [Pytorch 官方示例](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#)，MMEngine 的代码更加简洁，记录的日志也更加丰富。

在这个例子中，`NeuralNetwork.forward` 存在着以下跨模块的接口约定：

- 由于 `train_dataloader` 会返回一个 `(img, label)` 形式的元组，因此 `forward` 接口的前两个参数分别需要为 `img` 和 `label`。
- 由于 `forward` 在 `predict` 模式下会返回 `(pred, loss)` 形式的元组，因此 `process` 的 preds 参数应当同样为相同形式的元组。

## 数据处理器（DataPreprocessor）

如果你的电脑配有 GPU（或其他能够加速训练的硬件，如 mps、ipu 等），并运行了上节的代码示例。你会发现 Pytorch 的示例是在 CPU 上运行的，而 MMEngine 的示例是在 GPU 上运行的。`MMEngine` 是在何时把数据和模型从 CPU 搬运到 GPU 的呢？

事实上，执行器会在构造阶段将模型搬运到指定设备，而数据则会在 `train_step`、`val_step`、`test_step` 中，被[基础数据处理器（BaseDataPreprocessor）](mmengine.model.BaseDataPreprocessor)搬运到指定设备，进一步将处理好的数据传给模型。数据处理器作为模型基类的一个属性，会在模型基类的构造过程中被实例化。

为了体现数据处理器起到的作用，我们仍然以[上一节](#模型基类basemodel)训练 FashionMNIST 为例, 实现了一个简易的数据处理器，用于搬运数据和归一化：

```python
from torch.optim import SGD
from mmengine.model import BaseDataPreprocessor, BaseModel


class NeuralNetwork1(NeuralNetwork):

    def __init__(self, data_preprocessor):
        super().__init__(data_preprocessor=data_preprocessor)
        self.data_preprocessor = data_preprocessor

    def train_step(self, data, optimizer):
        img, label = self.data_preprocessor(data)
        loss = self(img, label, mode='loss')['loss'].sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return dict(loss=loss)

    def test_step(self, data):
        img, label = self.data_preprocessor(data)
        return self(img, label, mode='predict')

    def val_step(self, data):
        img, label = self.data_preprocessor(data)
        return self(img, label, mode='predict')


class NormalizeDataPreprocessor(BaseDataPreprocessor):

    def forward(self, data, training=False):
        img, label = [item for item in data]
        img = (img - 127.5) / 127.5
        return img, label


model = NeuralNetwork1(data_preprocessor=NormalizeDataPreprocessor())
optimizer = SGD(model.parameters(), lr=0.01)
data = (torch.full((3, 28, 28), fill_value=127.5), torch.ones(3, 10))

model.train_step(data, optimizer)
model.val_step(data)
model.test_step(data)
```

上例中，我们实现了 `BaseModel.train_step`、`BaseModel.val_step` 和 `BaseModel.test_step` 的简化版。数据经 `NormalizeDataPreprocessor.forward` 归一化处理，解包后传给 `NeuralNetwork.forward`，进一步返回损失或者预测结果。如果想实现自定义的参数优化或预测逻辑，可以自行实现 `train_step`、`val_step` 和 `test_step`，具体例子可以参考：[使用 MMEngine 训练生成对抗网络](../examples/train_a_gan.md)

```{note}
上例中数据处理器的 training 参数用于区分训练、测试阶段不同的批增强策略，`train_step` 会传入 `training=True`，`test_step` 和 `val_step` 则会传入 `trainig=Fasle`。
```

```{note}
通常情况下，我们要求 DataLoader 的 `data` 数据解包后（字典类型的被 **data 解包，元组列表类型被 *data 解包）能够直接传给模型的 `forward`。但是如果数据处理器修改了 data 的数据类型，则要求数据处理器的 `forward` 的返回值与模型 `forward` 的入参相匹配。
```
