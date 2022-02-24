# 记录日志 (logging)

## 概述

### 训练日志

训练日志指模型在训练/测试/推理迭代过程中的一系列状态日志，包括学习率（lr），损失（loss），评价指标（metric） 等。[TensorBoard ](https://www.tensorflow.org/tensorboard?hl=zh-cn) 、 [Wandb](https://wandb.ai/site) 等工具能将训练日志以图表的形式展示，便于我们观察模型的收敛情况。

- **统一的日志存储格式**

考虑不同类型的日志统计方式不同，损失（loss）一类的日志需要记录历史信息（用于平滑），而学习率，动量之类的却不需要。因此 **MMEnging** 在确保统计方式灵活性的前提下，抽象出了具有统一接口的日志类 `LogBuffer`，用户可以十分自然的使用 `LogBuffer` 来管理日志信息。

- **跨模块的日志传输**

在 **MMEngine** 中，不同组件使用 `LogBuffer` 来管理日志，例如在 [Scheduler](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/param_scheduler.md) 中记录学习率，[IterHook](TODO) 中记录迭代时间，最后将这些汇总到 [LoggerHook](TODO) 用于导出日志。上述流程涉及了不同组件的信息交互， 因此需要一套组件之间的消息传输方案。**MMEngine** 提出 `MessageHub` 类实现跨模块通讯，让同一个 Runner 的不同组件能够轻松读写同一份日志。`MessageHub` 和 `LogBuffer`  的配合让用户可以通过更改配置文件，来决定哪些日志需要被记录，以何种方式被记录。 

**MMEngine** 中 `MessageHub` 、 `LogBuffer` 与各组件之间的关系结构关系如下：

![结构关系](https://user-images.githubusercontent.com/57566630/155488205-5778b9aa-b4e3-499f-815b-e6c0d3e9277e.jpg)

可以看到 `MessaeHub` 除了记录日志（log_buffers）外，还会存储运行时信息（runtime）。运行时信息主要记录 runner 的 meta 信息、迭代次数等，运行时信息不需要历史记录，每次更新都会被覆盖。

### 系统日志

系统日志指 [runner](TODO) 从构建开始，到训练、测试、推理、乃至 runner 生命周期结束过程中，其内部模块产生的所有日志信息。系统日志用于监视模型训练的状态（迭代时间、内存占用是否正常），三方库抛出的警告或异常。当然系统日志也会记录部分训练日志，但是只会在终端显示，并且以文本格式保存在本地。

**MMEngine** 在 `logging` 模块基础上，简化了配置流程，让用户可以十分简单的获取功能强大的记录器（`logger`）管理模块 `MMLogger`。`MMLogger` 具备以下优点

- 分布式训练时，由 `MMLogger` 创建的 `logger` 能够保存所有进程的系统日志，以体现不同进程的训练状况
- 系统日志不受 **OpenMMLab** 算法库外的代码的日志影响，不会出现多重打印或日志格式不统一的情况
- error/warning 级别的日志需要输出对应代码在哪个文件的哪一行，便于用户调试，不同级别的日志有不同的色彩高亮

## 日志存储（LogBuffer）的使用

`LogBuffer` 

`LogBuffer ` 的对外接口如下：

- `__init__(log_history=[], count_history=[], max_length=1000000)`: log_history，count_history 可以是 `list`，`np.ndarray`，或 `torch.Tensor`，用于初始化日志的历史信息（log_history）队列 和日志的历史计数（count_history）。`max_length` 为队列的最大长度。当日志的队列超过最大长度时，会舍弃最早更新的日志。 

- `update(value, count=1)`: value 为需要被统计的日志信息，count 为 value 的累加次数，默认为 1。如果 value 已经是日志累加 n 次的结果（例如模型的迭代时间，实际上是 batch 张图片的累计耗时），需要另 `count=n`。
- `statistics('name', *args, **kwargs)`: 通过字符串来访问统计方法，传入参数必须和对应方法匹配。
- `register_statistics(method=None, name=None)`: 被 `register_statistics` 装饰的方法能被 `statistics()` 函数通过字符串访问。 
- `mean(window_size=None)`: 返回最近更新的 window_size 个日志的均值，默认返回全局平均值，可以通过 `statistics` 访问。
- `min(window_size=None)`: 返回最近更新的 window_size 个日志的最小值，默认返回全局最小值，可以通过 `statistics` 访问。
- `max(window_size=None)`: 返回最近更新的 window_size 个日志的最大值，默认返回全局最大值，可以通过 `statistics` 访问。
- `current()`: 返回最近一次更新的日志，可以通过 `statistics` 访问。
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

## 消息枢纽 （MessageHub）的使用

`LogBuffer` 可以十分简单的完成单个日志的更新和统计，而在模型训练过程中，日志的种类繁多，并且来自于不同的组件，因此如何完成日志的分发和收集是需要考虑的问题。 MMEngine  使用 `MessageHub` 来实现组件与组件、`runner` 与 `runner` 之间的互联互通。`MessageHub` 不仅会管理各个模块分发的 `LogBuffer`，还会管理一些运行时日志，例如 runner 的 meta 信息，迭代次数等。运行时信息每次更新都会被覆盖。

`MessageHub` 类的主要接口如下：

- `get_message_hub(name='', current=False)`：当不指定name，且 `current=False`时，该接口会获取 `root_message_hub`；当不指定 name，`current=True` 时，会返回最近一次被创建的`message_hub`。当指定 name 参数时，会返回对应 name 的 `message_hub`。
- `update_log(key, value, count=1)`: 更新指定字段的日志。value，count 对应 `LogBuffer.update` 接口的入参。
- `update_runtime(key, value)`: 更新运行时信息并覆盖。
- `get_log(key)`: 获取指定字段的日志。
- `get_runtime(key)`: 获取指定字段的运行时信息。
- `log_buffers`: 返回所有日志
- `caches`: 返回所有运行时信息。

### 消息枢纽的创建与访问

简单介绍如何访问 `root_message_hub`，如何通过 name 创建/访问对应 runner 的 `message_hub`，以及如何访问最近被创建的 `latest_message_hub`。

```Python
root_message_hub = MessageHub.get_message_hub()  # 不传参，默认返回 root_message_hub
task1_message_hub1 = MessageHub.get_message_hub('task1')  # 创建 task1_message_hub
latest_message_hub = MessageHub.get_message_hub_latest()  # task1_message_hub 是最近创建，因此返回 task1_message_hub
task2_message_hub2 = MessageHub.get_message_hub('task2')  # 创建 task2_message_hub2
latest_message_hub = MessageHub.get_message_hub_latest()  # task2_message_hub2 是最近创建，因此返回 task2_message_hub2
message_hub = MessageHub.get_message_hub('task1')  # 指定 task_name，访问之前创建的 task1_message_hub
```

### 消息枢纽的分发与记录

不同组件通过 name 来访问同一个 `message_hub`

```python
class Receiver:
    def __init__(self, name):
        self.message_hub = MessageHub.get_message_hub(name) # 获取 Task 中被创建的 message_hub

    def run(self):
        print(f"Learning rate is {self.message_hub.get_log('lr').current()}")
        # 接收数据，验证 Receiver 能够接收到 Dispatcher 中更新的日志。

class Dispatcher:
    def __init__(self, name):
        self.message_hub = MessageHub.get_message_hub(name)  # 获取 Task 中被创建的 message_hub

    def run(self):
        self.message_hub.update_log('lr', 0.001)  # 更新数据


class Task:
    def __init__(self, name):
        self.message_hub = MessageHub.get_message_hub(name)  # 创建制定 name 的 message_hub
        self.receiver = Receiver(name)
        self.dispatcher = Dispatcher(name)

    def run(self):
        self.dispatcher.run()
        self.receiver.run()
 
if __name__ == '__main__':
        task = Task('name')
    	task.run()
        # Learning rate is 0.001  

```

不显示指定 `name`，不同组件通过 `latest=True` 来访问同一个 message_hub。



## 记录系统日志

系统日志用于监视模型的训练状态，而记录器（`logger`）决定了系统日志的输出方式。为了能够导出格式统一、且能监听多进程训练状况的日志，MMEngine 在 `logging` 模块基础上，简化了配置流程，让用户可以十分简单的获取功能强大的记录器。

### 获取记录器

- `get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w')`：通过 `name `来获取对应名字的 `logger`，记录日志时 `name` 会出现在日志抬头。当 `log_file` 不为空时， `logger` 在终端输出日志的同时，会将日志保存到本地。log_level 用于控制日志等级，默认为 `logging.INFO`，`file_mode` 控制导出文件的读写权限，默认为 `w`。

我们可以通过 `get_logger` 来获取 `logger`，并将日志保存到本地。

```Python
logger = get_logger('mmengine', log_file='tmp.log')
logger.info("this is a test")
# 2022-02-20 22:26:38,860 - mmengine - INFO - this is a test
```

tmp.log

```text
2022-02-20 22:26:38,860 - mmengine - INFO - this is a test
```

### 日志等级不同，格式不同

对于 `ERROR` 级别的日志，我们不仅需要输出日志的内容，还需要输出错误发生的位置：

```python
logger = get_logger('mmengine', log_file='tmp.log')
logger.error('division by zero')

#2022-02-20 22:49:11,317 - mmengine - ERROR - division by zero Found error in file: /path/to/test_logger.py function:div line: #111
```

### 分布式训练时导出日志

分布式训练时，不同的进程会导出不同的日志

```Python
./work_dir
├── rank1_tmp.log
├── rank2_tmp.log
├── rank3_tmp.log
├── rank4_tmp.log
├── rank5_tmp.log
├── rank6_tmp.log
├── rank7_tmp.log
└── tmp.log
```
