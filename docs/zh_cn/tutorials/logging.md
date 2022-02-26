# 记录日志 (logging)

## 概述

### 训练日志

训练日志指模型在训练/测试/推理迭代过程中的状态日志，包括学习率（lr）、损失（loss）、评价指标（metric） 等。[TensorBoard](https://www.tensorflow.org/tensorboard?hl=zh-cn) 、 [Wandb](https://wandb.ai/site) 等工具能将训练日志以图表的形式展示，便于我们观察模型的训练情况。由于不同类型的日志统计方式不同，且更新自不同的组件，在 **OpenMMLab1.0** 中用户难以通过修改配置文件来选择输出哪些日志，以及日志的统计方式。因此 MMEngine 重新设计了日志的存储方式以及 [runner](TODO) 中的数据流，以支持这一特性。

- **统一的日志存储格式**

不同类型的日志统计方式不同，损失一类的日志需要记录历史信息（用于平滑），而学习率、动量之类的却不需要。因此 **MMEnging** 在确保日志统计方式灵活性的前提下，抽象出了具有统一接口的日志类 `LogBuffer`，用户可以十分自然的使用 `LogBuffer` 来管理日志信息。

- **跨模块的日志传输**

在 **MMEngine** 中，各组件使用 `LogBuffer` 来存储训练日志，例如损失、学习率和迭代时间等。最后 [LoggerHook](TODO)  会将这些日志会汇总导出。上述流程涉及了不同组件的信息交互， 因此需要一套组件之间的消息传输方案。**MMEngine** 设计了 `MessageHub` 类实现跨模块通讯，让同一个 Runner 的不同组件能够轻松读写同一份日志。`MessageHub` 和 `LogBuffer`  的配合让用户可以通过更改配置文件，来决定哪些日志需要被记录，以何种方式被记录。

**MMEngine** 中 `MessageHub` 、 `LogBuffer` 与各组件之间的关系结构关系如下：

![message_hub关系图 (1)](https://user-images.githubusercontent.com/57566630/155832053-f2f052fb-8911-46d5-9c7d-c395a12579da.jpg)

可以看到 `MessaeHub` 除了记录训练记录日志（log_buffers）外，还会存储运行时信息（runtime_info）。运行时信息主要是 runner 的 meta 信息、迭代次数等。

### 系统日志

系统日志指模型训练过程中产生的所有日志，包括模型初始化方式、模型的迭代耗时和内存占用、程序抛出的警告异常等。系统日志用于监视程序的运行状态，一般会在终端显示。为了让系统日志有更加清晰的格式，并且能够以文本的形式导出到本地，**MMEngine** 在 `logging` 模块基础上，简化了配置流程，让用户可以十分简单的通过日志 `MMLogger` 获取功能强大的记录器（`logger`）。使用 `MMLogger` 获取的 logger 具备以下优点：

- 分布式训练时，能够保存所有进程的系统日志，以体现不同进程的训练状况
- 系统日志不受 **OpenMMLab** 算法库外的代码的日志影响，不会出现多重打印或日志格式不统一的情况
- error/warning 级别的日志需要输出对应代码在哪个文件的哪一行，便于用户调试，不同级别的日志有不同的色彩高亮

## 统一的日志存储格式（LogBuffer）

`LogBuffer` 用于存储、统计不同类型的日志，为更新/统计损失、迭代时间、学习率等日志提供了统一的接口，对外接口如下：

- `__init__(log_history=[], count_history=[], max_length=1000000)`: log_history，count_history 可以是 `list`，`np.ndarray`，或 `torch.Tensor`，用于初始化日志的历史信息（log_history）队列 和日志的历史计数（count_history）。`max_length` 为队列的最大长度。当日志的队列超过最大长度时，会舍弃最早更新的日志。

- `update(value, count=1)`: value 为需要被统计的日志信息，count 为 value 的累加次数，默认为 1。如果 value 已经是日志累加 n 次的结果（例如模型的迭代时间，实际上是 batch 张图片的累计耗时），需要另 `count=n`。
- `statistics('name', *args, **kwargs)`: 通过字符串来访问统计方法，传入参数必须和对应方法匹配。
- `register_statistics(method=None, name=None)`: 被 `register_statistics` 装饰的方法能被 `statistics()` 函数通过字符串访问。
- `mean(window_size=None)`: 返回最近更新的 window_size 个日志的均值，默认返回全局平均值，可以通过 `statistics` 方法访问。
- `min(window_size=None)`: 返回最近更新的 window_size 个日志的最小值，默认返回全局最小值，可以通过 `statistics` 方法访问。
- `max(window_size=None)`: 返回最近更新的 window_size 个日志的最大值，默认返回全局最大值，可以通过 `statistics` 方法访问。
- `current()`: 返回最近一次更新的日志，可以通过 `statistics` 方法访问。
- `data`: 返回日志记录历史记录。

这里简单介绍如何使用 `LogBuffer` 记录日志。

### LogBuffer 初始化

```python
log_buffer = LogBuffer()  # 空初始化
log_history, count_history = log_buffer.data
# [] []
log_buffer = LogBuffer([1, 2, 3], [1, 2, 3])  # list 初始化
log_history, count_history = log_buffer.data
# [1 2 3] [1 2 3]
log_buffer = LogBuffer([1, 2, 3], [1, 2, 3], , max_length=2)
log_history, count_history = log_buffer.data  # 最大长度为2,只能存储 [2, 3]
# [2 3] [2 3]

```

### LogBuffer 更新

```python
log_buffer = LogBuffer([1, 2, 3], [1, 1, 1])
log_buffer.update(3, 1)  # 更新日志
log_history, count_history = log_buffer.data
# [1, 2, 3, 4] [1, 1, 1, 1]
```

### 统计最大最小值

```python
log_buffer = LogBuffer([1, 2, 3], [1, 1, 1])
log_buffer.min(2)
# 2，从 [2, 3] 中统计最小值
log_buffer.min()
# 返回全局最小值 1

log_buffer.max(2)
# 3，从 [2, 3] 中统计最大值
log_buffer.min()
# 返回全局最大值 3
```

### 统计均值

```python
log_buffer = LogBuffer([1, 2, 3], [1, 1, 1])
log_buffer.mean(2)
# 2.5，从 [2, 3] 中统计均值, (2 + 3) / (1 + 1)
log_buffer.mean()  # (1 + 2 + 3) / (1 + 1 + 1)
# 返回全局均值 2
log_buffer = LogBuffer([1, 2, 3], [2, 2, 2])  # 当 count 不为 1时
log_buffer.mean()  # (1 + 2 + 3) / (2 + 2 + 2)
# 返回均值 1
```

### 统计最近更新的值

```python
log_buffer = LogBuffer([1, 2, 3], [1, 1, 1])
log_buffer.update(4, 1)
log_buffer.current()
# 4
```

### 使用 statistics 接口选择统计方法

为了让用户可以通过配置文件来选择日志的统计方式，`LogBuffer` 提供了 `statistics` 接口，允许用户通过字符串来选择方法。需要注意，在调用 `statistics(name, *args, **kwargs)` 时，需要保证 name 是已注册的方法名，并且参数和方法相匹配。

```python
log_buffer = LogBuffer([1, 2, 3], [1, 1, 1])
log_buffer.statistics('mean')
# 2 返回全局均值
log_buffer.statistics('mean', 2)
# 2.5 返回 [2, 3] 的均值
log_buffer.statistics('mean', 2, 3)  # 错误！传入了不匹配的参数
log_buffer.statistics('data')  # 错误！ data 方法未被注册，无法被调用

```

### 使用 LogBuffer 统计训练日志

```Python
logs = dict(lr=LogBuffer(), loss=LogBuffer())  # 字典配合 LogBuffer 记录不同字段的日志
max_iter = 100
log_interval = 20
for iter in range(max_iter):
    lr = iter / max_iter * 0.1  # 线性学习率变化
    loss = 1 / iter  # loss
    logs['lr'].update('lr', 1)
    logs['loss'].update('lr', 1)
    if iter % log_interval == 0:
        latest_lr = logs['lr'].statistics('current')  # 通过字符串来选择统计方法
        mean_loss = logs['loss'].statistics('mean', log_interval)
        print(f'lr:    {latest_lr}'   # 平滑最新更新的 log_interval 个数据。
                f'loss: {mean_loss}')  # 返回最近一次更新的学习率。

```

### 自定义统计方式

如果用户想使用自定义的日志统计方式，可以使用 `LogBuffer.register_statistics` 来装饰自己实现的函数。

```Python
@LogBuffer.register_statistcs()
def custom_method(self, *args, *kwargs):
    ...

log_buffer = LogBuffer()
custom_log = log_buffer.statistics('custom_method')  #  使用 statistics 接口调用自定方法。
```

## 全局可访问基类（BaseGlobalAccessible）

考虑到 runner 中存在多个有全局访问需求的类，例如 logger，[ComposedWriter](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/visualizer.md)，和 MessageHub（后文提出），因此设计了 `BaseGlobalAccessible` 基类。继承自 `BaseGlobalAccessible` 的子类，通过 `create_instance` 方法创建的实例，在全局可以通过 `get_instance` 接口获取。

![BaseGlobalAccessible_UML](https://user-images.githubusercontent.com/57566630/155836961-4c27416a-746b-4e5a-acb8-4a4ae2ed6c54.jpg)

继承自 `BaseGlobalAccessible` 的子类，可以通过 `create_instance(name, *args, **kwargs)` 来创建全局实例，在程序的任何位置通过 `get_instance(name,  current=False)` 来获取对应实例。

### 定义子类

`BaseGlobalAccessible` 子类的构造函数的必须接收 name 参数，且会调用 `BaseGlobalAccessible` 的构造函数，这样能改保每个实例有 name 属性，以及 name 与对应实例一一对应。

```python
class GlobalAccessible(BaseGlobalAccessible):
    def __init__(self, name, *args, **kwargs):  # 必须接收 name 参数
        super().__init__(name)  #  调用父类构造函数
        ...
```

### 创建实例

不通过 `create_instance` 创建的子类实例只是普通实例，不具备全局访问特性，无法通过 `get_instance` 获取。`create_instance` 不能创建重复 name 的全局实例。

```python
instance_local = GlobalAccessible('local')
instance_local = GlobalAccessible.get_instance('local')
# 错误，local 不是通过 create_instance创建，无法获取
instance_local = GlobalAccessible.create_instance('global')
instance_local = GlobalAccessible.get_instance('global')   # 能够获取 global
instance_local = GlobalAccessible.create_instance('global') # 错误，全局实例不允许重复创建
```

### 获取实例

- 调用 `get_instance(name)`，如果传入 name 会返回对应的子类实例，不传入则会根据 current 参数 返回根实例(调用 create_instance 每个子类都会初始一个 root_instance)或者最近被创建的子类实例。

```python
instance = GlobalAccessible.create_instance()
instance.instance_name # root 不传参，默认返回 root
instance = GlobalAccessible.create_instance('task1')
instance.instance_name # task1，返回 task1
instance = GlobalAccessible.get_instance(current=True)
instance.instance_name # task1  返回 task1,最近被创建
instance = GlobalAccessible.create_instance('task2')
instance = GlobalAccessible.get_instance(current=True)
instance.instance_name # task2 返回 task2,最近被创建
```

## 消息枢纽（MessageHub）

`LogBuffer` 可以十分简单的完成单个日志的更新和统计，而在模型训练过程中，日志的种类繁多，并且来自于不同的组件，因此如何完成日志的分发和收集是需要考虑的问题。 **MMEngine**  使用全局可访问类 `MessageHub` 来实现组件与组件、`runner` 与 `runner` 之间的互联互通。`MessageHub` 不仅会管理各个模块分发的 `LogBuffer`，还会管理一些运行时日志，例如 runner 的 meta 信息，迭代次数等。运行时信息每次更新都会被覆盖。

`MessageHub` 类继承自 `BaseGlobalAccessible` ，其主要接口如下：

<img src="https://user-images.githubusercontent.com/57566630/155838812-ecde80fd-be61-45db-bc9e-59bfcbc67e90.jpg" alt="MessageHub_UML" style="zoom:50%;" />

- `update_log(key, value, count=1)`: 更新指定字段的`LogBuffer` 。value，count 对应 `LogBuffer.update` 接口的入参。
- `update_info(key, value)`: 更新运行时信息并覆盖。
- `get_log(key)`: 获取指定字段的日志。
- `get_info(key)`: 获取指定字段的运行时信息。
- `log_buffers`: 返回所有日志
- `runtime_info`: 返回所有运行时信息。

### 更新/获取日志

`Logbuffer` 以 `dict` 的形式存储在 `MessageHub` 中。当我们第一次调用 `update_log` 时，会初始化对应字段的 `LogBuffer`，后续每次更新等价于对应字段的 `LogBuffer` 调用 `update` 方法。同样的我们可以通过 `get_log` 来获取对应字段的 `LogBuffer`，并按需计算统计值。如果想获取 `message_hub` 的全部日志，可以访问其 `log_buffers` 属性。

```python
message_hub = MessageHub.create_instance('task')
message_hub.update_log('loss', 1, 1)
message_hub.get_log('loss').current() # 1，最近一次更新值为 1
message_hub.update_log('loss', 3, 1)
message_hub.get_log('loss').mean()  # 2，均值为 (3 + 1) / (1 +1)
message_hub.update_log('lr', 0.1, 1)

log_dict = message_hub.log_buffers  # 返回存储全部 LogBuffer 的字典
lr_buffer, loss_buffer = log_dict['lr'], log_dict['loss']
```

### 更新/获取运行时信息

运行时信息 以 dict 的形式存储在 `MessageHub` 中，支持任意数据类型，每次更新都会覆盖。

```python
message_hub = MessageHub.create_instance('task')
message_hub.update_info('meta', dict(task=task))  # 更新 meta
message_hub.get_info('meta')  # {'task'='task'} 获取 meta
message_hub.update_info('meta', dict(task=task1))  # 覆盖 meta
message_hub.get_info('meta')  # {'task'='task1'} 之前的信息被覆盖

runtime_dict = message_hub.rumtime_info  # 返回存储全部 LogBuffer 的字典
meta = log_dict['meta']
```

### 消息枢纽的跨模块通讯

```python
class Receiver:
    # 汇总不同模块更新的消息
    def __init__(self, name):
        self.message_hub = MessageHub.get_instance(name)  # 获取 MessaeHub

    def run(self):
        print(f"Learning rate is {self.message_hub.get_log('lr').current()}")
        print(f"Learning rate is {self.message_hub.get_log('loss').current()}")
        print(f"Learning rate is {self.message_hub.get_info('meta')}")


class LrUpdater:
    # 更新学习率
    def __init__(self, name):
        self.message_hub = MessageHub.get_instance(name)  # 获取 MessaeHub

    def run(self):
        self.message_hub.update_log('lr', 0.001)  # 更新学习率，以 LogBuffer 形式存储


class MetaUpdater:
    # 更新 meta 信息
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
        self.message_hub.update_log('loss', 0.1)


class Task:
    # 组合个各个模块
    def __init__(self, name):
        self.message_hub = MessageHub.create_instance(name)  # 创建 MessageHub
        self.receiver = Receiver(name)
        self.updaters = [LossUpdater(name),
                         MetaUpdater(name),
                         LrUpdater(name)]

    def run(self):
        for updater in self.updaters:
            updater.run()
        self.receiver.run()

if __name__ == '__main__':
    task = Task('name')
    task.run()
    # Learning rate is 0.001
    # Learning rate is 0.1
    # Learning rate is {'experiment': 'retinanet_r50_caffe_fpn_1x_coco.py', 'repo': 'mmdetection'}

```

## 记录器（MMLogger）

为了能够导出层次分明、格式统一的系统日志，**MMEnging** 在 `logging` 模块的基础上设计了 MMLogger，其继承于 `BaseGlobalAccessible` 和 `logging.Logger`。由 `MMLogger.get_intance` 获取的 logger 具备统一的日志格式，且不会继承 `logging.root` ，受到三方库的日志影响。

![MMLogger_UML 图](https://user-images.githubusercontent.com/57566630/155837144-a492c2c4-3859-43e5-8c5d-eee62c3c37a4.jpg)

`MMLogger` 在构造函数中完成了 logger 的配置，除了`BaseGlobalAccessible` 和 `logging.Logger` 的基类接口外，没有提供额外的接口。`MMLogger` 全局获取的方式和 MessageHub 相同，此处不再赘述，我们主要介绍通过 `MMLogger.create_instance` 获取的 logger 具备哪些功能。

### 日志格式

```python
logger = MMLogger.create_instance('mmengine')
logger.info("this is a test")
# 2022-02-20 22:26:38,860 - mmengine - INFO - this is a test
```

logger 除了输出消息外，还会额外输出时间戳、logger 名字和日志等级。对于 ERROR 等级的日志，我们会用红色高亮日志等级，并额外输出日志的代码位置

```python
logger = MMLogger.create_instance('mmengine')
logger.error('division by zero')

#2022-02-20 22:49:11,317 - mmengine - ERROR - Found error in file: /path/to/test_logger.py function:div line: 111 “division by zero”
```

### 导出日志

调用 create_instance 时，如果指定了 log_file，会将日志记录的信息以文本格式导出到本地。

```Python
logger = get_logger('mmengine', log_file='tmp.log')
logger.info("this is a test")
# 2022-02-20 22:26:38,860 - mmengine - INFO - this is a test
```

`tmp.log`:

```text
2022-02-20 22:26:38,860 - mmengine - INFO - this is a test
```

### 分布式训练时导出日志

在使用 runner 分布式训练时，不同的进程会导出不同的日志，日志文件名为 runner 实例化时的时间戳。

```Python
./work_dir
├── rank1_20220228_121221.log
├── rank2_20220228_121221.log
├── rank3_20220228_121221.log
├── rank4_20220228_121221.log
├── rank5_20220228_121221.log
├── rank6_20220228_121221.log
├── rank7_20220228_121221.log
└── 20220228_121221.log
```
