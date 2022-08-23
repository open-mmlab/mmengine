# 执行器（Runner）

深度学习算法的训练、验证和测试通常都拥有相似的流程，因此 MMEngine 提供了执行器以帮助用户简化这些任务的实现流程。 用户只需要准备好模型训练、验证、测试所需要的模块构建执行器，便能够通过简单调用执行器的接口来完成这些任务。用户如果需要使用这几项功能中的某一项，只需要准备好对应功能所依赖的模块即可。

用户可以手动构建这些模块的实例，也可以通过编写[配置文件](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/config.html)，
由执行器自动从[注册器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/registry.html)中构建所需要的模块，我们推荐使用后一种方式。

## 手动构建模块来使用执行器

### 手动构建模块进行训练

如上文所说，使用执行器的某一项功能时需要准备好对应功能所依赖的模块。以使用执行器的训练功能为例，用户需要准备[模型](TODO) 、[优化器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optimizer.html) 、
[参数调度器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/param_scheduler.html) 还有训练[数据集](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/basedataset.html) 。

```python
# 准备训练任务所需要的模块
import torch
from torch import nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from mmengine.model import BaseModel
from mmengine.optim.scheduler import MultiStepLR

# 定义一个多层感知机网络
class Network(BaseModel):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch_inputs: torch.Tensor, data_samples = None, mode: str = 'tensor'):
        x = batch_inputs.flatten(1)
        x = self.mlp(x)
        if mode == 'loss':
            return {'loss': self.loss(x, data_samples)}
        elif mode == 'predict':
            return x.argmax(1)
        else:
            return x

model = Network()

# 构建优化器
optimzier = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# 构建参数调度器用于调整学习率
lr_scheduler = MultiStepLR(milestones=[2], by_epoch=True)
# 构建手写数字识别 (MNIST) 数据集
train_dataset = datasets.MNIST(root="MNIST", download=True, train=True, transform=transforms.ToTensor())
# 构建数据加载器
train_dataloader = DataLoader(dataset=train_dataset, batch_size=10, num_workers=2)
```

在创建完符合上述文档规范的模块的对象后，就可以使用这些模块初始化执行器：

```python
from mmengine.runner import Runner


# 训练相关参数设置，按轮次训练，训练3轮
train_cfg = dict(by_epoch=True, max_epoch=3)

# 初始化执行器
runner = Runner(model,
                work_dir='./train_mnist',  # 工作目录，用于保存模型和日志
                train_cfg=train_cfg,
                train_dataloader=train_dataloader,
                optim_wrapper=dict(optimizer=optimizer),
                param_scheduler=lr_scheduler)
# 执行训练
runner.train()
```

上面的例子中，我们手动构建了一个多层感知机网络和手写数字识别 (MNIST) 数据集，以及训练所需要的优化器和学习率调度器，使用这些模块初始化了执行器，并且设置了训练配置 `train_cfg`，让执行器将模型训练3个轮次，最后通过调用执行器的 `train` 方法进行模型训练。

用户也可以修改 `train_cfg` 使执行器按迭代次数控制训练：

```python
# 训练相关参数设置，按迭代次数训练，训练9000次迭代
train_cfg = dict(by_epoch=False, max_epoch=9000)
```

### 手动构建模块进行测试

再举一个模型测试的例子，模型的测试需要用户准备模型和训练好的权重路径、测试数据集以及[评测器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/evaluator.html) ：

```python
from mmengine.evaluator import BaseMetric


class MnistAccuracy(BaseMetric):
    def process(self, data, preds) -> None:
        self.results.append(((data[1] == preds.cpu()).sum(), len(preds)))
    def compute_metrics(self, results):
        correct, batch_size = zip(*results)
        acc = sum(correct) / sum(batch_size)
        return dict(accuracy=acc)

model = Network()
test_dataset = datasets.MNIST(root="MNIST", download=True, train=False, transform=transforms.ToTensor())
test_dataloader = DataLoader(dataset=test_dataset)
metric = MnistAccuracy()
test_evaluator = Evaluator(metric)

# 初始化执行器
runner = Runner(model=model, test_dataloader=test_dataloader, test_evaluator=test_evaluator,
                load_from='./train_mnist/epoch_3.pth', work_dir='./test_mnist')

# 执行测试
runner.test()
```

这个例子中我们重新手动构建了一个多层感知机网络，以及测试用的手写数字识别数据集和使用 (Accuracy) 指标的评测器，并使用这些模块初始化执行器，最后通过调用执行器的 `test` 函数进行模型测试。

### 手动构建模块在训练过程中进行验证

在模型训练过程中，通常会按一定的间隔在验证集上对模型的进行进行验证。在使用 MMEngine 时，只需要构建训练和验证的模块，并在训练配置中设置验证间隔即可

```python
# 准备训练任务所需要的模块
optimzier = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lr_scheduler = MultiStepLR(milestones=[2], by_epoch=True)
train_dataset = datasets.MNIST(root="MNIST", download=True, train=True, transform=transforms.ToTensor())
train_dataloader = DataLoader(dataset=train_dataset, batch_size=10, num_workers=2)

# 准备验证需要的模块
val_dataset = datasets.MNIST(root="MNIST", download=True, train=False, transform=transforms.ToTensor())
val_dataloader = Dataloader(dataset=val_dataset)
metric = MnistAccuracy()
val_evaluator = Evaluator(metric)


# 训练相关参数设置
train_cfg = dict(by_epoch=True,  # 按轮次训练
                 max_epochs=5,  # 训练5轮
                 val_begin=2,  # 从第 2 个 epoch 开始验证
                 val_interval=1)  # 每隔1轮进行1次验证

# 初始化执行器
runner = Runner(model=model, optim_wrapper=dict(optimizer=optimzier), param_scheduler=lr_scheduler,
                train_dataloader=train_dataloader, val_dataloader=val_dataloader, val_evaluator=val_evaluator,
                train_cfg=train_cfg, work_dir='./train_val_mnist')
# 执行训练
runner.train()
```

## 通过配置文件使用执行器

OpenMMLab 的开源项目普遍使用注册器 + 配置文件的方式来管理和构建模块，MMEngine 中的执行器也推荐使用配置文件进行构建。
下面是一个通过配置文件使用执行器的例子：

```python
from mmengine import Config
from mmengine.runner import Runner

# 加载配置文件
config = Config.fromfile('configs/resnet/resnet50_8xb32_in1k.py')

# 通过配置文件初始化执行器
runner = Runner.build_from_cfg(config)

# 执行训练
runner.train()

# 执行测试
runner.test()
```

与手动构建模块来使用执行器不同的是，通过调用 Runner 类的 `build_from_cfg` 方法，执行器能够自动读取配置文件中的模块配置，从相应的注册器中构建所需要的模块，用户不再需要考虑训练和测试分别依赖哪些模块，也不需要为了切换训练的模型和数据而大量改动代码。

下面是一个典型的使用配置文件调用 MMClassification 中的模块训练分类器的简单例子：

```python
# 工作目录，保存权重和日志
work_dir = './train_resnet'
# 默认注册器域
default_scope = 'mmcls'  # 默认使用 `mmcls` (MMClassification) 注册器中的模块
# 模型配置
model = dict(type='ImageClassifier',
             backbone=dict(type='ResNet', depth=50),
             neck=dict(type='GlobalAveragePooling'),
             head=dict(type='LinearClsHead',num_classes=1000))
# 数据配置
train_dataloader = dict(dataset=dict(type='ImageNet', pipeline=[...]),
                        sampler=dict(type='DefaultSampler', shuffle=True),
                        batch_size=32,
                        num_workers=4)
val_dataloader = ...
test_dataloader = ...

# 优化器配置
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))
# 参数调度器配置
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)
#验证和测试的评测器配置
val_evaluator = dict(type='Accuracy')
test_evaluator = dict(type='Accuracy')

# 训练、验证、测试流程配置
train_cfg = dict(
    by_epoch=True,
    max_epochs=100,
    val_begin=20,  # 从第 20 个 epoch 开始验证
    val_interval=1  # 每隔一个 epoch 进行一次验证
)
val_cfg = dict()
test_cfg = dict()

# 自定义钩子 (可选)
custom_hooks = [...]

# 默认钩子 (可选，未在配置文件中写明时将使用默认配置)
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),  # 运行时信息钩子
    timer=dict(type='IterTimerHook'),  # 计时器钩子
    sampler_seed=dict(type='DistSamplerSeedHook'),  # 为每轮次的数据采样设置随机种子的钩子
    logger=dict(type='TextLoggerHook'),  # 训练日志钩子
    param_scheduler=dict(type='ParamSchedulerHook'),  # 参数调度器执行钩子
    checkpoint=dict(type='CheckpointHook', interval=1),  # 模型保存钩子
)

# 环境配置 (可选，未在配置文件中写明时将使用默认配置)
env_cfg = dict(
    cudnn_benchmark=False,  # 是否使用 cudnn_benchmark
    dist_cfg=dict(backend='nccl'),  # 分布式通信后端
    mp_cfg=dict(mp_start_method='fork')  # 多进程设置
)
# 日志处理器 (可选，未在配置文件中写明时将使用默认配置)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
# 日志等级配置
log_level = 'INFO'

# 加载权重的路径 (None 表示不加载)
load_from = None
# 从加载的权重文件中恢复训练
resume = False
```

一个完整的配置文件主要由模型、数据、优化器、参数调度器、评测器等模块的配置，训练、验证、测试等流程的配置，还有执行流程过程中的各种钩子模块的配置，以及环境和日志等其他配置的字段组成。
通过配置文件构建的执行器采用了懒初始化 (lazy initialization)，只有当调用到训练或测试等执行函数时，才会根据配置文件去完整初始化所需要的模块。

关于配置文件的更详细的使用方式，请参考[配置文件教程](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/config.md)

## 加载权重或恢复训练

执行器可以通过 `load_from` 参数加载检查点（checkpoint）文件中的模型权重，只需要将 `load_from` 参数设置为检查点文件的路径即可。

```python
runner = Runner(model=model, test_dataloader=test_dataloader, test_evaluator=test_evaluator,
                load_from='./resnet50.pth')
```

如果是通过配置文件使用执行器，只需修改配置文件中的 `load_from` 字段即可。

用户也可通过设置 `resume=True` 来，加载检查点中的训练状态信息来恢复训练。当 `load_from` 和 `resume=True` 同时被设置时，执行器将加载 `load_from` 路径对应的检查点文件中的训练状态。

如果仅设置 `resume=True`，执行器将会尝试从 `work_dir` 文件夹中寻找并读取最新的检查点文件。

你可能还想阅读[执行器的设计](../design/runner.md)或者[执行器的 API 文档](https://mmengine.readthedocs.io/zh_CN/latest/api/runner.html)。
