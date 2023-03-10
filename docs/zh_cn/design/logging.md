# 日志系统

## 概述

[执行器（Runner）](./runner.md)在运行过程中会产生很多日志，例如加载的数据集信息、模型的初始化信息、训练过程中的学习率、损失等。为了让用户能够更加自由的获取这些日志信息，MMEngine 实现了[消息枢纽（MessageHub）](mmengine.logging.MessageHub)、[历史缓冲区（HistoryBuffer）](mmengine.logging.HistoryBuffer)、[日志处理器（LogProcessor）](mmengine.runner.LogProcessor) 和 [MMLogger](mmengine.logging.MMLogger) 来支持以下功能：

- 用户可以通过配置文件，根据个人偏好来选择日志统计方式，例如在终端输出整个训练过程中的平均损失而不是基于固定迭代次数平滑的损失
- 用户可以在任意组件中获取当前的训练状态，例如当前的迭代次数、训练轮次等
- 用户可以通过配置文件来控制是否保存分布式训练下的多进程日志

![image](https://user-images.githubusercontent.com/57566630/163441489-47999f3a-3259-44ab-949c-77a8a599faa5.png)

训练过程中的产生的损失、学习率等数据由历史缓冲区管理和封装，汇总后交给消息枢纽维护。日志处理器将消息枢纽中的数据进行格式化，最后通过[记录器钩子（LoggerHook）](mmengine.hooks.LoggerHook) 展示到各种可视化后端。**一般情况下用户无需感知数据处理流程，可以直接通过配置日志处理器来选择日志的统计方式**。在介绍 MMEngine 的日志系统的设计之前，可以先阅读[记录日志教程](../advanced_tutorials/logging.md) 了解日志系统的基本用法。

## 历史缓冲区（HistoryBuffer）

MMEngine 实现了历史数据存储的抽象类历史缓冲区（HistoryBuffer），用于存储训练日志的历史轨迹，如模型损失、优化器学习率、迭代时间等。通常情况下，历史缓冲区作为内部类，配合[消息枢纽（MessageHub）](mmengine.logging.MessageHub)、记录器钩子（LoggerHook ）和[日志处理器（LogProcessor）](mmengine.runner.LogProcessor) 实现了训练日志的可配置化。

用户也可以单独使用历史缓冲区来管理训练日志，能够非常简单的使用不同方法来统计训练日志。我们先来介绍如何单独使用历史缓冲区，在消息枢纽一节再进一步介绍二者的联动。

### 历史缓冲区初始化

历史缓冲区的初始化可以接受 `log_history` 和 `count_history` 两个参数。`log_history` 表示日志的历史轨迹，例如前三次迭代的 loss 为 0.3，0.2，0.1。我们就可以记 `log_history=[0.3, 0.2, 0.1]`。`count_history` 是一个比较抽象的概念，如果按照迭代次数来算，0.3，0.2，0.1 分别是三次迭代的结果，那么我们可以记 `count_history=[1, 1, 1]`，其中 “1” 表示一次迭代。如果按照 batch 来算，例如每次迭代的 `batch_size` 为 8，那么 `count_history=[8, 8, 8]`。`count_history` 只会在统计均值时用到，用于控制返回均值的粒度。就拿上面那个例子来说，`count_history=[1, 1, 1]` 时会统计每次迭代的平均 loss，而 `count_history=[8, 8, 8]` 则会统计每张图片的平均 loss。

```python
from mmengine.logging import HistoryBuffer

history_buffer = HistoryBuffer()  # 空初始化
log_history, count_history = history_buffer.data
# [] []
history_buffer = HistoryBuffer([1, 2, 3], [1, 2, 3])  # list 初始化
log_history, count_history = history_buffer.data
# [1 2 3] [1 2 3]
history_buffer = HistoryBuffer([1, 2, 3], [1, 2, 3], max_length=2)
# The length of history buffer(3) exceeds the max_length(2), the first few elements will be ignored.
log_history, count_history = history_buffer.data  # 最大长度为2,只能存储 [2, 3]
# [2 3] [2 3]
```

我们可以通过 `history_buffer.data` 来返回日志的历史轨迹。此外，我们可以为历史缓冲区设置最大队列长度，当历史缓冲区的长度大于最大队列长度时，会自动丢弃最早更新的数据。

### 更新历史缓冲区

我们可以通过 `update` 接口来更新历史缓冲区。update 接受两个参数，第一个参数用于更新 `log_history `，第二个参数用于更新 `count_history`。

```python
history_buffer = HistoryBuffer([1, 2, 3], [1, 1, 1])
history_buffer.update(4)  # 更新日志
log_history, count_history = history_buffer.data
# [1, 2, 3, 4] [1, 1, 1, 1]
history_buffer.update(5, 2)  # 更新日志
log_history, count_history = history_buffer.data
# [1, 2, 3, 4, 5] [1, 1, 1, 1, 2]
```

### 基本统计方法

历史缓冲区提供了基本的数据统计方法：

- `current()`：获取最新更新的数据。
- `mean(window_size=None)`：获取窗口内数据的均值，默认返回数据的全局均值
- `max(window_size=None)`：获取窗口内数据的最大值，默认返回全局最大值
- `min(window_size=None)`：获取窗口内数据的最小值，默认返回全局最小值

```python
history_buffer = HistoryBuffer([1, 2, 3], [1, 1, 1])
history_buffer.min(2)
# 2，从 [2, 3] 中统计最小值
history_buffer.min()
# 返回全局最小值 1

history_buffer.max(2)
# 3，从 [2, 3] 中统计最大值
history_buffer.min()
# 返回全局最大值 3
history_buffer.mean(2)
# 2.5，从 [2, 3] 中统计均值, (2 + 3) / (1 + 1)
history_buffer.mean()  # (1 + 2 + 3) / (1 + 1 + 1)
# 返回全局均值 2
history_buffer = HistoryBuffer([1, 2, 3], [2, 2, 2])  # 当 count 不为 1时
history_buffer.mean()  # (1 + 2 + 3) / (2 + 2 + 2)
# 返回均值 1
history_buffer = HistoryBuffer([1, 2, 3], [1, 1, 1])
history_buffer.update(4, 1)
history_buffer.current()
# 4
```

### 统计方法的统一入口

要想支持在配置文件中通过配置 'max'，'min' 等字段来选择日志的统计方式，那么 HistoryBuffer 就需要一个接口来接受 'min'，'max' 等统计方法字符串和相应参数，进而找到对应的统计方法，最后输出统计结果。`statistics(name, *args, **kwargs)` 接口就起到了这个作用。其中 name 是已注册的方法名（已经注册 `min`，`max` 等基本统计方法），`*arg` 和 `**kwarg` 用于接受对应方法的参数。

```python
history_buffer = HistoryBuffer([1, 2, 3], [1, 1, 1])
history_buffer.statistics('mean')
# 2 返回全局均值
history_buffer.statistics('mean', 2)
# 2.5 返回 [2, 3] 的均值
history_buffer.statistics('mean', 2, 3)  # 错误！传入了不匹配的参数
history_buffer.statistics('data')  # 错误！ data 方法未被注册，无法被调用
```

### 注册统计方法

为了保证历史缓冲区的可扩展性，用户可以通过 `register_statistics` 接口注册自定义的统计函数

```python
from mmengine.logging import HistoryBuffer
import numpy as np


@HistoryBuffer.register_statistics
def weighted_mean(self, window_size, weight):
    assert len(weight) == window_size
    return (self._log_history[-window_size:] * np.array(weight)).sum() / \
            self._count_history[-window_size:]


history_buffer = HistoryBuffer([1, 2], [1, 1])
history_buffer.statistics('weighted_mean', 2, [2, 1])  # get (2 * 1 + 1 * 2) / (1 + 1)
```

用户可以通过 `statistics` 接口，传入方法名和对应参数来调用被注册的函数。

### 使用样例

用户可以独立使用历史缓冲区来记录日志，通过简单的接口调用就能得到期望的统计接口。

```Python
logs = dict(lr=HistoryBuffer(), loss=HistoryBuffer())  # 字典配合 HistoryBuffer 记录不同字段的日志
max_iter = 10
log_interval = 5
for iter in range(1, max_iter+1):
    lr = iter / max_iter * 0.1  # 线性学习率变化
    loss = 1 / iter  # loss
    logs['lr'].update(lr, 1)
    logs['loss'].update(loss, 1)
    if iter % log_interval == 0:
        latest_lr = logs['lr'].statistics('current')  # 通过字符串来选择统计方法
        mean_loss = logs['loss'].statistics('mean', log_interval)
        print(f'lr:   {latest_lr}\n'  # 返回最近一次更新的学习率。
              f'loss: {mean_loss}')   # 平滑最新更新的 log_interval 个数据。
# lr:   0.05
# loss: 0.45666666666666667
# lr:   0.1
# loss: 0.12912698412698415
```

MMEngine 利用历史缓冲区的特性，结合消息枢纽，实现了训练日志的高度可定制化。

## 消息枢纽（MessageHub）

历史缓冲区（HistoryBuffer）可以十分简单地完成单个日志的更新和统计，而在模型训练过程中，日志的种类繁多，并且来自于不同的组件，因此如何完成日志的分发和收集是需要考虑的问题。 MMEngine 使用消息枢纽（MessageHub）来实现组件与组件、执行器与执行器之间的数据共享。消息枢纽继承自全局管理器（ManagerMixin），支持跨模块访问。

消息枢纽存储了两种含义的数据：

- 历史缓冲区字典：消息枢纽会收集各个模块更新的训练日志，如损失、学习率、迭代时间，并将其更新至内部的历史缓冲区字典中。历史缓冲区字典经[消息处理器（LogProcessor）](mmengine.runner.LogProcessor)处理后，会被输出到终端/保存到本地。如果用户需要记录自定义日志，可以往历史缓冲区字典中更新相应内容。
- 运行时信息字典：运行时信息字典用于存储迭代次数、训练轮次等运行时信息，方便 MMEngine 中所有组件共享这些信息。

```{note}
当用户想在终端输出自定义日志，或者想跨模块共享一些自定义数据时，才会用到消息枢纽。
```

为了方便用户理解消息枢纽在训练过程中更新信息以及分发信息的流程，我们通过几个例子来介绍消息枢纽的使用方法，以及如何使用消息枢纽向终端输出自定义日志。

### 更新/获取训练日志

历史缓冲区以字典的形式存储在消息枢纽中。当我们第一次调用 `update_scalar` 时，会初始化对应字段的历史缓冲区，后续的每次更新等价于调用对应字段历史缓冲区的 `update` 方法。同样的我们可以通过 `get_scalar` 来获取对应字段的历史缓冲区，并按需计算统计值。如果想获取消息枢纽的全部日志，可以访问其 `log_scalars` 属性。

```python
from mmengine import MessageHub

message_hub = MessageHub.get_instance('task')
message_hub.update_scalar('train/loss', 1, 1)
message_hub.get_scalar('train/loss').current()  # 1，最近一次更新值为 1
message_hub.update_scalar('train/loss', 3, 1)
message_hub.get_scalar('train/loss').mean()  # 2，均值为 (3 + 1) / (1 + 1)
message_hub.update_scalar('train/lr', 0.1, 1)

message_hub.update_scalars({'train/time': {'value': 0.1, 'count': 1},
                            'train/data_time': {'value': 0.1, 'count': 1}})

train_time = message_hub.get_scalar('train/time')  # 获取单个日志

log_dict = message_hub.log_scalars  # 返回存储全部 HistoryBuffer 的字典
lr_buffer, loss_buffer, time_buffer, data_time_buffer = (
    log_dict['train/lr'], log_dict['train/loss'], log_dict['train/time'],
    log_dict['train/data_time'])
```

```{note}
损失、学习率、迭代时间等训练日志在执行器和钩子中自动更新，无需用户维护。
```

```{note}
消息枢纽的历史缓冲区字典对 key 没有特殊要求，但是 MMEngine 约定历史缓冲区字典的 key 要有 train/val/test 的前缀，只有带前缀的日志会被输出当终端。
```

### 更新/获取运行时信息

运行时信息以字典的形式存储在消息枢纽中，能够存储任意数据类型，每次更新都会被覆盖。

```python
message_hub = MessageHub.get_instance('task')
message_hub.update_info('iter', 1)
message_hub.get_info('iter')  # 1
message_hub.update_info('iter', 2)
message_hub.get_info('iter')  # 2 覆盖上一次结果
```

### 消息枢纽的跨组件通讯

执行器运行过程中，各个组件会通过消息枢纽来分发、接受消息。[RuntimeInfoHook](mmengine.hooks.RuntimeInfoHook) 会汇总其他组件更新的学习率、损失等信息，将其导出到用户指定的输出端（Tensorboard，WandB 等）。由于上述流程较为复杂，这里用一个简单示例来模拟日志钩子和其他组件通讯的过程。

```python
from mmengine import MessageHub

class LogProcessor:
    # 汇总不同模块更新的消息，类似 LoggerHook
    def __init__(self, name):
        self.message_hub = MessageHub.get_instance(name)  # 获取 MessageHub

    def run(self):
        print(f"Learning rate is {self.message_hub.get_scalar('train/lr').current()}")
        print(f"loss is {self.message_hub.get_scalar('train/loss').current()}")
        print(f"meta is {self.message_hub.get_info('meta')}")


class LrUpdater:
    # 更新学习率
    def __init__(self, name):
        self.message_hub = MessageHub.get_instance(name)  # 获取 MessageHub

    def run(self):
        self.message_hub.update_scalar('train/lr', 0.001)
        # 更新学习率，以 HistoryBuffer 形式存储


class MetaUpdater:
    # 更新元信息
    def __init__(self, name):
        self.message_hub = MessageHub.get_instance(name)

    def run(self):
        self.message_hub.update_info(
            'meta',
            dict(experiment='retinanet_r50_caffe_fpn_1x_coco.py',
                 repo='mmdetection'))    # 更新元信息，每次更新会覆盖上一次的信息


class LossUpdater:
    # 更新损失函数
    def __init__(self, name):
        self.message_hub = MessageHub.get_instance(name)

    def run(self):
        self.message_hub.update_scalar('train/loss', 0.1)

class ToyRunner:
    # 组合个各个模块
    def __init__(self, name):
        self.message_hub = MessageHub.get_instance(name)  # 创建 MessageHub
        self.log_processor = LogProcessor(name)
        self.updaters = [LossUpdater(name),
                         MetaUpdater(name),
                         LrUpdater(name)]

    def run(self):
        for updater in self.updaters:
            updater.run()
        self.log_processor.run()

if __name__ == '__main__':
    task = ToyRunner('name')
    task.run()
    # Learning rate is 0.001
    # loss is 0.1
    # meta {'experiment': 'retinanet_r50_caffe_fpn_1x_coco.py', 'repo': 'mmdetection'}
```

### 添加自定义日志

我们可以在任意模块里更新消息枢纽的历史缓冲区字典，历史缓冲区字典中所有的合法字段经统计后最后显示到终端。

```{note}
更新历史缓冲区字典时，需要保证更新的日志名带有 train，val，test 前缀，否则日志不会在终端显示。
```

```python
class CustomModule:
    def __init__(self):
        self.message_hub = MessageHub.get_current_instance()

    def custom_method(self):
        self.message_hub.update_scalar('train/a', 100)
        self.message_hub.update_scalars({'train/b': 1, 'train/c': 2})
```

默认情况下，终端上额外显示 a、b、c 最后一次更新的结果。我们也可以通过配置[日志处理器](mmengine.runner.LogProcessor)来切换自定义日志的统计方式。

## 日志处理器（LogProcessor）

用户可以通过配置日志处理器（LogProcessor）来控制日志的统计方法及其参数。默认配置下，日志处理器会统计最近一次更新的学习率、基于迭代次数平滑的损失和迭代时间。用户可以在日志处理器中配置已知字段的统计方式。

### 最简配置

```python
log_processor = dict(
    window_size=10,
)
```

此时终端会输出每 10 次迭代的平均损失和平均迭代时间。假设此时终端的输出为

```bash
04/15 12:34:24 - mmengine - INFO - Iter [10/12]  , eta: 0:00:00, time: 0.003, data_time: 0.002, loss: 0.13
```

### 自定义的统计方式

我们可以通过配置 `custom_cfg` 列表来选择日志的统计方式。`custom_cfg` 中的每一个元素需要包括以下信息：

- `data_src`：日志的数据源，用户通过指定 `data_src` 来选择需要被重新统计的日志，一份数据源可以有多种统计方式。默认的日志源包括模型输出的损失字典的 `key`、学习率（`lr`）和迭代时间（`time`/`data_time`），一切经消息枢纽的 `update_scalar`/`update_scalars` 更新的日志均为可以配置的数据源（需要去掉 `train/`、`val/` 前缀）。（必填项）
- `method_name`：日志的统计方法，即历史缓冲区中的基本统计方法以及用户注册的自定义统计方法（必填项）
- `log_name`：日志被重新统计后的名字，如果不定义 `log_name`，新日志会覆盖旧日志（选填项）
- 其他参数：统计方法会用到的参数，其中 `window_size` 为特殊字段，可以为普通的整型、字符串 epoch 和字符串 global。LogProcessor 会实时解析这些参数，以返回基于 iteration、epoch 和全局平滑的统计结果（选填项）

1. 覆盖旧的统计方式

```python
log_processor = dict(
    window_size=10,
    by_epoch=True,
    custom_cfg=[
        dict(data_src='loss',
             method_name='mean',
             window_size=100)])
```

此时会无视日志处理器的默认窗口 10，用更大的窗口 100 去统计 loss 的均值，并将原有结果覆盖。

```bash
04/15 12:34:24 - mmengine - INFO - Iter [10/12]  , eta: 0:00:00, time: 0.003, data_time: 0.002, loss: 0.11
```

2. 新增统计方式，不覆盖

```python
log_processor = dict(
    window_size=10,
    by_epoch=True,
    custom_cfg=[
        dict(data_src='loss',
             log_name='loss_min',
             method_name='min',
             window_size=100)])
```

```bash
04/15 12:34:24 - mmengine - INFO - Iter [10/12]  , eta: 0:00:00, time: 0.003, data_time: 0.002, loss: 0.11, loss_min: 0.08
```

## MMLogger

为了能够导出层次分明、格式统一、且不受三方库日志系统干扰的日志，MMEngine 在 `logging` 模块的基础上实现了 `MMLogger`。`MMLogger` 继承自全局管理器（`ManagerMixin`），相比于 `logging.Logger`，`MMLogger` 能够在无法获取 `logger` 的名字（logger name）的情况下，拿到当前执行器的 `logger`。

### 创建 MMLogger

我们可以通过 `get_instance` 接口创建全局可获取的 `logger`，默认的日志格式如下

```python
logger = MMLogger.get_instance('mmengine', log_level='INFO')
logger.info("this is a test")
# 04/15 14:01:11 - mmengine - INFO - this is a test
```

`logger` 除了输出消息外，还会额外输出时间戳、logger 的名字和日志等级。对于 ERROR 等级的日志，我们会用红色高亮日志等级，并额外输出错误日志的代码位置

```python
logger = MMLogger.get_instance('mmengine', log_level='INFO')
logger.error('division by zero')
# 04/15 14:01:56 - mmengine - ERROR - /mnt/d/PythonCode/DeepLearning/OpenMMLab/mmengine/a.py - <module> - 4 - division by zero
```

### 导出日志

调用 `get_instance` 时，如果指定了 log_file，会将日志记录的信息以文本格式导出到本地。

```Python
logger = MMLogger.get_instance('mmengine', log_file='tmp.log', log_level='INFO')
logger.info("this is a test")
# 04/15 14:01:11 - mmengine - INFO - this is a test
```

`tmp/tmp.log`:

```text
04/15 14:01:11 - mmengine - INFO - this is a test
```

由于分布式情况下会创建多个日志文件，因此我们在预定的导出路径下，增加一级和导出文件同名的目录，用于存储所有进程的日志。上例中导出路径为 `tmp.log`，实际存储路径为 `tmp/tmp.log`。

### 分布式训练时导出日志

使用 pytorch 分布式训练时，我们可以通过配置 `distributed=True` 来导出分布式训练时各个进程的日志（默认关闭）。

```python
logger = MMLogger.get_instance('mmengine', log_file='tmp.log', distributed=True, log_level='INFO')
```

单机多卡，或者多机多卡但是共享存储的情况下，导出的分布式日志路径如下

```text
#  共享存储
work_dir/20230228_141908
├── 20230306_183634_${hostname}_device0_rank0.log
├── 20230306_183634_${hostname}_device1_rank1.log
├── 20230306_183634_${hostname}_device2_rank2.log
├── 20230306_183634_${hostname}_device3_rank3.log
├── 20230306_183634_${hostname}_device4_rank4.log
├── 20230306_183634_${hostname}_device5_rank5.log
├── 20230306_183634_${hostname}_device6_rank6.log
├── 20230306_183634_${hostname}_device7_rank7.log
...
├── 20230306_183634_${hostname}_device7_rank63.log
```

多机多卡，独立存储的情况：

```text
# 独立存储
# 设备0：
work_dir/20230228_141908
├── 20230306_183634_${hostname}_device0_rank0.log
├── 20230306_183634_${hostname}_device1_rank1.log
├── 20230306_183634_${hostname}_device2_rank2.log
├── 20230306_183634_${hostname}_device3_rank3.log
├── 20230306_183634_${hostname}_device4_rank4.log
├── 20230306_183634_${hostname}_device5_rank5.log
├── 20230306_183634_${hostname}_device6_rank6.log
├── 20230306_183634_${hostname}_device7_rank7.log

# 设备7：
work_dir/20230228_141908
├── 20230306_183634_${hostname}_device0_rank56.log
├── 20230306_183634_${hostname}_device1_rank57.log
├── 20230306_183634_${hostname}_device2_rank58.log
├── 20230306_183634_${hostname}_device3_rank59.log
├── 20230306_183634_${hostname}_device4_rank60.log
├── 20230306_183634_${hostname}_device5_rank61.log
├── 20230306_183634_${hostname}_device6_rank62.log
├── 20230306_183634_${hostname}_device7_rank63.log
```
