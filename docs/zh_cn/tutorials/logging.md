# 记录日志 (logging)

在模型训练过程中，我们往往需要记录一些日志信息，例如迭代次数（`iter`），学习率（`lr`），损失（`loss`）等。常见的做法是将需要统计的日志存储到列表，然后根据个人偏好平滑、显示、保存数据。MMEnging 在保证记录方式灵活性的前提下，抽象出了具有统一接口的日志类，用户可以十分自然的使用日志类来管理日志信息。在 **OpenMMLab2.0** 中，用户可以通过更改配置文件，来决定哪些日志需要被记录，以何种方式被记录。

## 概述

我们将模型训练过程中的日志分成两类

- 训练日志

  训练日志用于监视模型的收敛情况，观察损失收敛是否正常，学习率变化是否符合预期，模型训练的数据是否正确，推理的结果是否变好等。[TensorBoard ](https://www.tensorflow.org/tensorboard?hl=zh-cn) 、 [Wandb](https://wandb.ai/site) 等工具会将训练日志以图表的形式展示，便于用户观察。

- 系统日志

  系统日志用于监视模型训练的状态，观察每次迭代的时间，三方库抛出的警告或异常。当然系统日志也会记录部分训练日志，但是只会在终端显示，或者以文本格式保存在本地。

然而训练日志和系统日志可能来自于 [Runner](TODO) 的不同组件，例如需要从 [Scheduler](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/param_scheduler.md) 中获取学习率，[Hook](TODO) 中获取迭代时间、训练图片，[Model](TODO) 中获取特征图信息，因此需要一套组件之间的消息传输方案。MMEngine 通过 `MessageHub` 类实现跨模块通讯，让同一个 Runner 的不同组件能够轻松访问/修改同一份日志。最后 `LoggerHook` 汇总日志信息，将训练日志、系统日志分别输出到终端/本地/网页。

考虑到损失（`loss`）一类的日志需要额外记录历史信息（用于平滑），而学习率，迭代次数之类的却不需要。因此 MMEngine 设计了 `LogBuffer` 类以统一日志的存储方式。`LogBuffer` 除了提供一些基础接口来统计日志，如`latest`，`moving_mean` ，还支持用字符串来调用对应的统计方法。这一特性让用户能够在配置文件中选择日志的统计方式。

MMEngine 中 `MessageHub` 与 `LogBuffer` 结构关系如下：

![结构关系](https://user-images.githubusercontent.com/57566630/154812484-25247662-242f-4b94-bc29-db0d42d6c181.png)

可以看到 `MessaeHub` 除了记录日志外，还会缓存（caches）信息。缓存主要用来记录临时变量，例如某次迭代时的特征图，缓存信息没有历史记录，每次更新都会被覆盖。

## 日志存储（LogBuffer）的使用

`LogBuffer ` 的主要接口如下：

- `update(value, count)`：将 `value ` 和 `count` 更新到日志队列（队列长度有上限），`value` 为日志统计 `count` 次的累加值。以统计训练时间为例，每个 `Iter`（一个 `batch_size`）的训练耗时为 `iter_time`。如果我们想统计每个 `Iter` 的平均耗时，需要调用 `update(value=iter_time, count=1)`（含义为 1 个 iter 的耗时为 `iter_time`），如果需要统计每张图片的平均耗时，则调用 `update(value=iter_time, count=batch_size)` （含义为 `batch_size ` 张图片的耗时为 `iter_time`）。`LogBuffer` 的滑动平均接口会根据 `count ` 的值返回不同含义的日志。

- `moving_mean(window_size=None)`：平滑窗口内的日志，即 `sum(values[-window_size:]) / sum(counts[-window_size:]) `，默认返回全局平均值。
- `min(window_size=None)`：返回窗口内日志的最小值，默认返回全局最小值。
- `max(window_size=None)`：返回窗口内日志的最大值，默认返回全局最大值。
- `latest()`：返回最近一次更新的日志。
- `excute('name', *args, **kwargs)`：通过字符串来访问方法。
- `data()`：返回日志。

这里简单介绍如何使用 `LogBuffer` 记录日志。

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
        latest_lr = logs['lr'].latest()
        mean_loss = logs['loss'].moving_mean(log_interval)
        print(f'lr:   {latest_lr}'   # 平滑最新更新的 log_interval 个数据。
              f'loss: {mean_loss}')  # 返回最近一次更新的学习率。
    
```

为了让用户能够通过配置文件来指定日志的统计方式， `LogBuffer` 支持用字符串来访问方法。

```python
for iter in range(max_iter):
		...
    if iter % log_interval == 0:
        latest_lr = logs['lr'].excute('latest')							# 通过字符串来访问方法
        mean_loss = logs['loss'].excute('moving_mean', log_interval)
        ...
```

如果用户想使用自定义的日志统计方式，可以继承 `LogBuffer`类，并使用  `MethodRegister` 来装饰自定义方法，让其可以被 `excute` 函数调用。

```Python
@LOG_
class CustomLogBuffer(LogBuffer)
	@MethodRegister
	def custom_method(self, *args, *kwargs):
		...

log_buffer = CustomLogBuffer()
custom_log = log_buffer.excute('custom_method')  #  被 MethodRegister 修饰后，可以通过字符串调用对应方法。
```

## 消息枢纽 （MessageHub）的使用

`LogBuffer` 可以十分简单的完成单个日志的更新和统计，而在模型训练过程中，日志的种类繁多，并且来自于不同的组件，因此如何完成日志的分发和收集是需要考虑的问题。 MMEngine  使用 `MessageHub` 来实现组件与组件之间、`runner` 与 `runner` 之间的互联互通。`MessageHub` 不仅会管理各个模块分发的 `LogBuffer`，还会管理一些缓存，例如模型的特征图，缓存信息每次更新都会被覆盖。

`MessageHub` 的主要接口如下：

- `get_message_hub(name='', log_cfg=dict(type='LogBuffer'))`：当不指定 `name` 时，获取汇总所有 `runner` 消息的 `root_message_hub`。当指定 `name=task_name` 时，则返回对应 runner 的 `message_hub`（task 和 runner 之间的关系见：[runner](TODO)）。`log_cfg` 用于配置自定义的 `LogBuffer ` 类型，默认使用 MMEngine 的 `LogBuffer`。
- `get_message_hub_latest()`：获取最近被创建的 `message_hub`。考虑到有些组件无法获取到对应 runner 的 `task_name`，可以通过 `get_message_hub_latest` 来获取当前组件隶属 runner 的 `message_hub`。
- `update_log(key, value, count=1)`：更新指定字段的日志。`value，count` 即 `LogBuffer` 的 `update` 接口的入参。
- `update_cache(key, value)`： 更新缓存并覆盖。
- `get_log(key)`: 获取指定字段的日志。
- `get_cache(key)`: 获取指定字段的缓存。
- `log_buffers`: 返回所有日志
- `caches`：返回所有缓存。

### 消息枢纽的创建与访问

简单介绍如何访问 `root_message_hub`，如果通过 `task_name` 访问对应 runner 的 `message_hub`，以及如何访问最近被创建的 `latest_message_hub`。

```Python
root_message_hub = MessageHub.get_message_hub()  # 不传参，默认返回 root_message_hub
task1_message_hub1 = MessageHub.get_message_hub('task1')  # 创建 task1_message_hub
latest_message_hub = MessageHub.get_message_hub_latest()  # task1_message_hub 是最近创建，因此返回 task1_message_hub
task2_message_hub2 = MessageHub.get_message_hub('task2')  # 创建 task2_message_hub2
latest_message_hub = MessageHub.get_message_hub_latest()  # task2_message_hub2 是最近创建，因此返回 task2_message_hub2
message_hub = MessageHub.get_message_hub('task1')  # 指定 task_name，访问之前创建的 task1_message_hub
```

### 消息枢纽的分发与记录

这边以 [IterTimerHook](TODO) 和 用户自定义的 `CustomModels` 为例，介绍 MMEngine 如实使用  `MessageHub` 实现模块之间的信息交互。

- 消息分发

```python
class IterTimerHook(Hook):
    def before_run(self, runner):
        # 对于能获取到 task 信息的组件，根据 task 获取对应 runner 的 message_hub
        self.message_hub = MessageHub.get_message_hub(runner.task)
        
    def before_epoch(self, runner):
        self.t = time.time()
        
    def before_iter(self, runner):
        # 使用 update_log 更新日志信息
        self.message_hub.update_log 更新日志信息('data_time', time.time() - self.t)

    def after_iter(self, runner):
        # 使用 update_log 更新日志信息
        self.message_hub.update_log('time', time.time() - self.t)
        self.t = time.time()
        
class CustomModels(nn.Module):
    def __init__():
        ...
        # model 模块无法获取 task 信息，直接使用 get_message_hub_latest() 获取当前 runner 的 message_hub
        self.message_hub = MessageHub.get_message_hub_latest()
    
    def forward(img):
        feat = self.model(x)
        # 更新缓存。
        self.message_hub.update_cache('custom_feat', feat)
```

- 消息汇总

[LoggerHook](TODO) 需要将各组件分发的日志进行汇总，通过 [Writer]([mmengine/visualizer.md at main · open-mmlab/mmengine (github.com)](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/visualizer.md)) 写到指定端。这边简单介绍 `LoggerHook` 是如何使用 `MessageHub` 及其成员 `LogBuffer`，灵活的选择需要被记录的字段（`momentum`，`lr` 等）及其统计方式（`latest`，`moving_mean` 等）。

```python
# LoggerHook 模块收集日志：
class LoggerHook(Hook):
    # 默认的平滑方式
    def __init__(...
                 interval=20,
        		 custom_keys=[dict(log_key='momentum',
                                   type='latest', 
                                   method='latest')]
                 smooth_method='moving_mean',
                 ...
                 ):
        ...
        # 额外记录用户感兴趣的日志，用户需要提供日志的字段以及平滑方式
        self.interval = interval
        self.custom_keys = custom_keys
        self.smooth_method = smooth_method
        
    def log(self, runner):
        ...
        log_dict = OrderedDict()
        log_buffers = self.message.log_buffers
        log_dict['lr'] = log_buffers['lr'].latest() # 对于统计方式固定的字段，直接调用对应方法，例如字段为 `lr` 时，使用 `latest` 接口
        for key, log_buffer in log_buffers:
            # 根据用户配置的 `smooth_method`，对数据进行平滑，默认 窗口大小为 self.interval。
            log_dict[key]  = log_buffer[key].excute(self.smooth_method,
                                              self.interval)
        # 根据用户给出的平滑尺度和平滑方法来统计自定义字段的日志。
        for item in self.custom_keys:
            key = item['log_key']
            method = item['method']
            method_type = item['type']
            if method_type == 'latest':
                log_dict[key] = log_buffer[key].excute('latest')
            elif method_type == 'global_smooth'  # 全局平滑 ，窗口大小为日志的总长度，不需要传参，平滑方式由用户指定。
                log_dict[key] = log_buffer[key].excute(method)
            elif method_type == 'epoch_smooth'	 # 窗口大小为 runner.inner_iter
                log_dict[key] = log_buffer[key].excute(method,
                                                       runner.inner_iter)
            elif method_type == 'iter_smooth'
                log_dict[key] = log_buffer[key].excute(method,
                                                       self.interval)
        ...
```

如果需要可视化模型的特征图， `VisualizerHook` 也可以通过 `MessageHub` 来获取 `CustomModels ` 中分发的 `custom_feat`，实现特征可视化。

```python
class CustomVisualizerHook(Hook):
    def __init__(...
                custom_keys=)
	def after_train_epoch(runner):
        ...
        custom_feat = self.message.get_cache('custom_feat')
        self.visualizer.draw_featmap(custom_feat)
        ...
```

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

