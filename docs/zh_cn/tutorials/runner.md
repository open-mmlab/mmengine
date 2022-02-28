# 执行器（Runner）

OpenMMLab 的算法库中提供了各种算法模型的训练、测试、推理功能，这些功能在不同算法方向上都有着相似的接口。
因此， MMEngine 抽象出了执行器来负责通用的算法模型的训练、测试、推理任务。
用户一般可以直接使用 MMEngine 中的默认执行器，也可以对执行器进行定制化修改以满足更高级的需求。

在介绍如何使用执行器之前，我们先举几个例子来帮助用户理解为什么需要执行器。

下面是一段使用 PyTorch 进行模型训练的伪代码：

```python
model = ResNet()
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
train_dataset = ImageNetDataset(...)
train_dataloader = DataLoader(train_dataset, ...)

for i in range(max_epochs):
    for data_batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(data_batch)
        loss = loss_func(outputs, data_batch)
        loss.backward()
        optimizer.step()
```

下面是一段使用 PyTorch 进行模型测试的伪代码：

```python
model = ResNet()
model.load_state_dict(torch.load(CKPT_PATH))
model.eval()

test_dataset = ImageNetDataset(...)
test_dataloader = DataLoader(test_dataset, ...)

for data_batch in test_dataloader:
    outputs = model(data_batch)
    acc = calculate_acc(outputs, data_batch)
```

下面是一段使用 PyTorch 进行模型推理的伪代码：

```python
model = ResNet()
model.load_state_dict(torch.load(CKPT_PATH))
model.eval()

for img in image_list:
    prediction = model(img)
```

可以从上面的三段代码看出，这三个任务的执行流程都可以归纳为构建模型、读取数据、循环迭代等步骤。上述代码都是以图像分类为例，但不论是图像分类还是目标检测或是图像分割，都脱离不了这套范式。
因此，我们将模型的训练、验证、测试的流程整合起来，形成了执行器。在执行器中，我们只需要准备好模型、数据等任务必须的模块或是这些模块的配置文件，执行器会自动完成任务流程的准备和执行。
通过使用执行器以及 MMEngine 中丰富的功能模块，用户不再需要手动搭建训练测试的流程，也不再需要去处理分布式与非分布式训练的区别，可以专注于算法和模型本身。

## 如何使用执行器

MMEngine 中默认的执行器支持执行模型的训练、测试以及推理。如果用户需要使用这几项功能中的某一项，那就需要准备好对应功能所依赖的模块。
用户可以手动构建这些模块的实例，也可以通过编写[配置文件](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/config.html) ，
由执行器自动从[注册器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/registry.html) 中构建所需要的模块。这两种使用方式中，我们更推荐后者。

### 手动构建模块来使用执行器

如上文所说，使用执行器的某一项功能时需要准备好对应功能所依赖的模块。以使用执行器的训练功能为例，用户需要准备[模型](TODO) 、[优化器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optimizer.html) 、
[参数调度器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/param_scheduler.html) 还有训练[数据集](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/basedataset.html) 。
在创建完符合上述文档规范的模块的对象后，就可以使用这些模块初始化执行器：

```python
# 准备训练任务所需要的模块
model = ResNet()
optimzier = SGD(model.parameters(), lr=0.01, momentum=0.9)
lr_scheduler = MultiStepLR(milestones=[80, 90], by_epoch=True)
train_dataset = ImageNetDataset()
train_dataloader = Dataloader(dataset=train_dataset, batch_size=32, num_workers=4)

# 训练相关参数设置
train_cfg = dict(by_epoch=True, max_epoch=100)

# 初始化执行器
runner = Runner(model=model, optimizer=optimzier, param_scheduler=lr_scheduler,
                train_dataloader=train_dataloader, train_cfg=train_cfg)
# 执行训练
runner.train()
```

上面的例子中，我们手动构建了 ResNet 分类模型和 ImageNet 数据集，以及训练所需要的优化器和学习率调度器，使用这些模块初始化了执行器，最后通过调用执行器的 `train` 函数进行模型训练。

再举一个模型测试的例子，模型的测试需要用户准备模型和训练好的权重路径、测试数据集以及[评测器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/evaluator.html) ：

```python
model = FasterRCNN()
test_dataset = CocoDataset()
test_dataloader = Dataloader(dataset=test_dataset, batch_size=2, num_workers=2)
evaluator = CocoEvaluator(metric='bbox')

# 初始化执行器
runner = Runner(model=model, test_dataloader=test_dataloader, evaluator=evaluator,
                load_checkpoint='./faster_rcnn.pth')

# 执行测试
runner.test()
```

这个例子中我们手动构建了一个 Faster R-CNN 检测模型，以及测试用的 COCO 数据集和对应的 COCO 评测器，并使用这些模块初始化执行器，最后通过调用执行器的 `test` 函数进行模型测试。

### 通过配置文件使用执行器

OpenMMLab 的开源项目普遍使用注册器 + 配置文件的方式来管理和构建模块，MMEngine 中的执行器也推荐使用配置文件进行构建。
下面是一个通过配置文件使用执行器的例子：

```python
from mmengine import Config, Runner

# 加载配置文件
config = Config.fromfile('configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py/')

# 通过配置文件初始化执行器
runner = Runner.build_from_cfg(config)

# 执行训练
runner.train()

# 执行测试
runner.test()
```

与手动构建模块来使用执行器不同的是，通过调用 Runner 类的 `build_from_cfg` 方法，执行器能够自动读取配置文件中的模块配置，从相应的注册器中构建所需要的模块，用户不再需要考虑训练和测试分别依赖哪些模块，也不需要为了切换训练的模型和数据而大量改动代码。

下面是一个典型的配置简单例子：

```python
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
optimizer = dict(type='SGD', lr=0.01)
# 参数调度器配置
param_scheduler = dict(type='MultiStepLR', milestones=[80, 90])
#评测器配置
evaluator = dict(type='Accuracy')

# 训练、验证、测试流程配置
train_cfg = dict(by_epoch=True, max_epochs=100)
validation_cfg = dict(interval=1)  # 每隔一个 epoch 进行一次验证
test_cfg = dict()

# 自定义钩子
custom_hooks = [...]

# 默认钩子
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # 计时器钩子
    checkpoint=dict(type='CheckpointHook', interval=1),  # 模型保存钩子
    logger=dict(type='TextLoggerHook'),  # 训练日志钩子
    optimizer=dict(type='OptimzierHook', grad_clip=False),  # 优化器钩子
    param_scheduler=dict(type='ParamSchedulerHook'))  # 参数调度器执行钩子

# 环境配置
env_cfg = dict(
    dist_params=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork')
)
# 系统日志配置
log_cfg = dict(log_level='INFO')
```

一个完整的配置文件主要由模型、数据、优化器、参数调度器、评测器等模块的配置，训练、验证、测试等流程的配置，还有执行流程过程中的各种钩子模块的配置，以及环境和日志等其他配置的字段组成。
通过配置文件构建的执行器是延迟初始化 (lazy-init) 的，只有当调用到训练或测试等执行函数时，才会根据配置文件去完整初始化所需要的模块。

## 进阶使用

MMEngine 中的默认执行器能够完成大部分的深度学习任务，但不可避免会存在无法满足的情况。有的用户希望能够对执行器进行更多自定义修改，因此，MMEngine 支持自定义模型的训练、验证以及测试的流程。
更进一步，如果默认执行器中依然有其他无法满足需求的部分，用户可以像自定义其他模块一样，通过继承重写的方式，实现自定义的执行器。执行器同样也可以通过注册器进行管理。

### 自定义执行流程

在 MMEngine 中，我们将任务的执行流程抽象成循环（Loop），因为大部分的深度学习任务执行流程都可以归纳为模型在一组或多组数据上进行循环迭代。
MMEngine 内提供了四种默认的循环：
- EpochBasedTrainLoop 基于轮次的训练循环
- IterBasedTrainLoop 基于迭代次数的训练循环
- ValLoop 标准的验证循环
- TestLoop 标准的测试循环

![Loop](https://user-images.githubusercontent.com/12907710/155972762-8ec29ec1-aa2a-42f8-9aee-ff4a56d7bdc0.jpg)

用户可以通过继承循环基类来实现自己的训练流程。循环基类需要提供两个输入：`runner` 执行器的实例和 `loader` 循环所需要迭代的迭代器。
用户如果有自定义的需求，也可以增加更多的输入参数。MMEngine 中同样提供了 LOOPS 注册器对循环类进行管理，用户可以向注册器内注册自定义的循环模块，
然后在配置文件的 `train_cfg`、`validation_cfg`、`test_cfg` 中增加 `type` 字段来指定使用何种循环。

```python
from mmengine.registry import LOOPS
from mmengine.runner.loop import BaseLoop

@LOOPS.register_module()
class CustomValLoop(BaseLoop):
    def __init__(self, runner, loader, evaluator, loader2):
        super().__init__(runner, loader, evaluator)
        self.loader2 = runner.build_dataloader(loader2)

    def run(self):
        self.runner.call_hooks('before_val')
        for idx, databatch in enumerate(self.loader):
            self.run_iter(idx, databatch)
        metric = self.evaluator.evaluate()
        for idx, databatch in enumerate(self.loader2):
            self.run_iter(idx, databatch)
        metric2 = self.evaluator.evaluate()

        ...

        self.runner.call_hooks('after_val')

```

上面的例子中实现了一个与默认验证循环不一样的自定义验证循环，它在两个不同的验证集上进行验证，并对两个验证结果进行进一步的处理。在实现了自定义的循环类之后，
只需要在配置文件的 `validation_cfg` 内设置 `type='CustomValLoop'`，并添加额外的配置即可。

```python
validation_cfg = dict(type='CustomValLoop', loader2=dict(dataset=dict(type='ValDataset2'), ...))
```

### 自定义执行器

如果自定义执行流程依然无法满足需求，用户同样可以实现自己的执行器。具体实现流程与其他模块无异：继承 MMEngine 中的 Runner，重写需要修改的函数，添加进 RUNNERS 注册器中，最后在配置文件中指定 `runner_type` 即可。

```python
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

@RUNNERS.register_module()
class CustomRunner(Runner):

    def setup_env(self):
        ...
```

上述例子实现了一个自定义的执行器，并重写了 `setup_env` 函数，然后添加进了 RUNNERS 注册器中，完成了这些步骤之后，便可以在配置文件中设置 `runner_type='CustomRunner'` 来构建自定义的执行器。
