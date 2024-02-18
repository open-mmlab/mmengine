# Logging

## Overview

[Runner](./runner.md) produces amounts of logs during execution. These logs include dataset information, model initialization, learning rates, losses, etc. In order to make these logs easily accessed by users, MMEngine designs [MessageHub](mmengine.logging.MessageHub), [HistoryBuffer](mmengine.logging.HistoryBuffer), [LogProcessor](mmengine.runner.LogProcessor) and [MMLogger](mmengine.logging.MMLogger), which enable:

- Configure statistical methods in config files. For example, losses can be globally averaged or smoothed by a sliding window.
- Query training states (iterations, epochs, etc.) in any module
- Configure whether save the multi-process log or not during distributed training.

![image](https://user-images.githubusercontent.com/57566630/163441489-47999f3a-3259-44ab-949c-77a8a599faa5.png)

Each scalar (losses, learning rates, etc.) during training is encapsulated by HistoryBuffer, managed by MessageHub in key-value pairs, formatted by LogProcessor and then exported to various visualization backends by [LoggerHook](mmengine.hooks.LoggerHook). **In most cases, statistical methods of these scalars can be configured through the LogProcessor without understanding the data flow.**  Before diving into the design of the logging system, please read through [logging tutorial](../advanced_tutorials/logging.md) first for familiarizing basic use cases.

## HistoryBuffer

`HistoryBuffer` records the history of the corresponding scalar such as losses, learning rates, and iteration time in an array. As an internal class, it works with [MessageHub](mmengine.logging.MessageHub), LoggerHook and [LogProcessor](mmengine.runner.LogProcessor) to make training log configurable. Meanwhile, HistoryBuffer can also be used alone, which enables users to manage their training logs and do various statistics in an easy manner.

We will first introduce the usage of HistoryBuffer in the following section. The association between HistoryBuffer and MessageHub will be introduced later in the MessageHub section.

### HistoryBuffer Initialization

HistoryBuffer accepts `log_history`, `count_history` and `max_length` for initialization.

- `log_history` records the history of the scaler. For example, if the loss in the previous 3 iterations is 0.3, 0.2, 0.1 respectively, there will be `log_history=[0.3, 0.2, 0.1]`.
- `count_history` controls the statistical granularity and will be used when counting the average. Take the above example, if we count the average loss across iterations, we have `count_history=[1, 1, 1]`. Instead, if we count the average loss across images with `batch_size=8`, then we have `count_history=[8, 8, 8]`.
- `max_length` controls the maximum length of the history. If the length of `log_history` and `count_history` exceeds `max_length`, the earliest elements will be removed.

Besides, we can access the history of the data through `history_buffer.data`.

```python
from mmengine.logging import HistoryBuffer

history_buffer = HistoryBuffer()  # Default initialization
log_history, count_history = history_buffer.data
# [] []
history_buffer = HistoryBuffer([1, 2, 3], [1, 2, 3])  # Init with lists
log_history, count_history = history_buffer.data
# [1 2 3] [1 2 3]
history_buffer = HistoryBuffer([1, 2, 3], [1, 2, 3], max_length=2)
# The length of history buffer(3) exceeds the max_length(2), the first few elements will be ignored.
log_history, count_history = history_buffer.data
# [2 3] [2 3]
```

### HistoryBuffer Update

We can update the `log_history` and `count_history` through `HistoryBuffer.update(log_history, count_history)`.

```python
history_buffer = HistoryBuffer([1, 2, 3], [1, 1, 1])
history_buffer.update(4)  # count default to 1
log_history, count_history = history_buffer.data
# [1, 2, 3, 4] [1, 1, 1, 1]
history_buffer.update(5, 2)
log_history, count_history = history_buffer.data
# [1, 2, 3, 4, 5] [1, 1, 1, 1, 2]
```

### Basic Statistical Methods

HistoryBuffer provides some basic statistical methods:

- `current()`: Get the latest data.
- `mean(window_size=None)`: Count the mean value of the previous `window_size` data. Defaults to None, as global mean.
- `max(window_size=None)`: Count the max value of the previous `window_size` data. Defaults to None, as global maximum.
- `min(window_size=None)`: Count the min value of the previous `window_size` data. Defaults to None, as global minimum.

```python
history_buffer = HistoryBuffer([1, 2, 3], [1, 1, 1])
history_buffer.min(2)
# 2, the minimum in [2, 3]
history_buffer.min()
# 1, the global minimum

history_buffer.max(2)
# 3, the maximum in [2, 3]
history_buffer.min()
# 3, the global maximum
history_buffer.mean(2)
# 2.5, the mean value in [2, 3], (2 + 3) / (1 + 1)
history_buffer.mean()
# 2, the global mean, (1 + 2 + 3) / (1 + 1 + 1)
history_buffer = HistoryBuffer([1, 2, 3], [2, 2, 2])  # Cases when counts are not 1
history_buffer.mean()
# 1, (1 + 2 + 3) / (2 + 2 + 2)
history_buffer = HistoryBuffer([1, 2, 3], [1, 1, 1])
history_buffer.update(4, 1)
history_buffer.current()
# 4
```

### Statistical Methods Invoking

Statistical methods can be accessed through `HistoryBuffer.statistics` with method name and arguments. The `name` parameter should be a registered method name (i.e. built-in methods like `min` and `max`), while arguments should be the corresponding method's arguments.

```python
history_buffer = HistoryBuffer([1, 2, 3], [1, 1, 1])
history_buffer.statistics('mean')
# 2, as global mean
history_buffer.statistics('mean', 2)
# 2.5, as the mean of [2, 3]
history_buffer.statistics('mean', 2, 3)
# Error! mismatch arguments given to `mean(window_size)`
history_buffer.statistics('data')
# Error! `data` method not registered
```

### Statistical Methods Registration

Custom statistical methods can be registered through `@HistoryBuffer.register_statistics`.

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

### Use Cases

```Python
logs = dict(lr=HistoryBuffer(), loss=HistoryBuffer())  # different keys for different logs
max_iter = 10
log_interval = 5
for iter in range(1, max_iter+1):
    lr = iter / max_iter * 0.1  # linear scaling of lr
    loss = 1 / iter  # loss
    logs['lr'].update(lr, 1)
    logs['loss'].update(loss, 1)
    if iter % log_interval == 0:
        latest_lr = logs['lr'].statistics('current')  # select statistical methods by name
        mean_loss = logs['loss'].statistics('mean', log_interval)  # mean loss of the latest `log_interval` iterations
        print(f'lr:   {latest_lr}\n'
              f'loss: {mean_loss}')
# lr:   0.05
# loss: 0.45666666666666667
# lr:   0.1
# loss: 0.12912698412698415
```

## MessageHub

As shown above, HistoryBuffer can easily handle the update and statistics of a single variable. However, there are multiple variables to log during training, each potentially coming from a different module. This makes it an issue to collect and distribute different variables. To address this issue, we provide MessageHub in MMEngine. It is derived from [ManagerMixin](../advanced_tutorials/manager_mixin.md) and thus can be accessed globally. It can be used to simplify the sharing of data across modules.

MessageHub stores data into 2 internal dictionaries, each has its own definition:

- `log_scalars`: Scalars including losses, learning rates and iteration time are collected from different modules and stored into the HistoryBuffer with corresponding key in this dict. Values in this dict will be formatted by [LogProcessor](mmengine.runner.LogProcessor) and then output to terminal or saved locally. If you want to customize your logging info, you can add new keys to this dict and update in the subsequent training steps.
- `runtime_info`: Some runtime information including epochs and iterations are stored in this dict. This dict makes it easy to share some necessary information across modules.

```{note}
You may need to use MessageHub only if you want to add extra data to logs or share custom data across modules.
```

The following examples show the usage of MessageHub, including scalars update, data sharing and log customization.

### Update & get training log

HistoryBuffers are stored in MessageHub's `log_scalars` dictionary as values. You can call `update_scalars` method to update the HistoryBuffer with the given key. On first call with an unseen key, a HistoryBuffer will be initialized. In the subsequent calls with the same key, the corresponding HistoryBuffer's `update` method will be invoked. You can get values or statistics of a HistoryBuffer by specifying a key in `get_scalar` method. You can also get full logs by directly accessing the `log_scalars` attribute of a MessageHub.

```python
from mmengine import MessageHub

message_hub = MessageHub.get_instance('task')
message_hub.update_scalar('train/loss', 1, 1)
message_hub.get_scalar('train/loss').current()  # 1, the latest updated train/loss
message_hub.update_scalar('train/loss', 3, 1)
message_hub.get_scalar('train/loss').mean()  # 2, the mean calculated as (1 + 3) / (1 + 1)
message_hub.update_scalar('train/lr', 0.1, 1)

message_hub.update_scalars({'train/time': {'value': 0.1, 'count': 1},
                            'train/data_time': {'value': 0.1, 'count': 1}})

train_time = message_hub.get_scalar('train/time')  # 1

log_dict = message_hub.log_scalars  # return the whole dict
lr_buffer, loss_buffer, time_buffer, data_time_buffer = (
    log_dict['train/lr'], log_dict['train/loss'], log_dict['train/time'],
    log_dict['train/data_time'])
```

```{note}
Losses, learning rates and iteration time are automatically updated by runner and hooks. You are not supposed to manually update them.
```

```{note}
MessageHub has no special requirements for keys in `log_scalars`. However, MMEngine will only output a scalar to logs if it has a key prfixed with train/val/test.
```

### Update & get runtime info

Runtime information is stored in `runtime_info` dict. The dict accepts data in any data types. Different from HistoryBuffer, the value will be overwritten on every update.

```python
message_hub = MessageHub.get_instance('task')
message_hub.update_info('iter', 1)
message_hub.get_info('iter')  # 1
message_hub.update_info('iter', 2)
message_hub.get_info('iter')  # 2, overwritten by the above command
```

### Share MessageHub across modules

During the execution of a runner, different modules receive and post data through MessageHub. Then, [RuntimeInfoHook](mmengine.hooks.RuntimeInfoHook) gathers data such as losses and learning rates before exporting them to user defined backends (Tensorboard, WandB, etc). Following is an example to show the communication between logger hook and other modules.

```python
from mmengine import MessageHub

class LogProcessor:
    # gather data from other modules. similar to logger hook
    def __init__(self, name):
        self.message_hub = MessageHub.get_instance(name)  # access MessageHub

    def run(self):
        print(f"Learning rate is {self.message_hub.get_scalar('train/lr').current()}")
        print(f"loss is {self.message_hub.get_scalar('train/loss').current()}")
        print(f"meta is {self.message_hub.get_info('meta')}")


class LrUpdater:
    # update the learning rate
    def __init__(self, name):
        self.message_hub = MessageHub.get_instance(name)  # access MessageHub

    def run(self):
        self.message_hub.update_scalar('train/lr', 0.001)
        # update the learning rate, saved as HistoryBuffer


class MetaUpdater:
    # update meta information
    def __init__(self, name):
        self.message_hub = MessageHub.get_instance(name)

    def run(self):
        self.message_hub.update_info(
            'meta',
            dict(experiment='retinanet_r50_caffe_fpn_1x_coco.py',
                 repo='mmdetection'))    # meta info will be overwritten on every update


class LossUpdater:
    # update losses
    def __init__(self, name):
        self.message_hub = MessageHub.get_instance(name)

    def run(self):
        self.message_hub.update_scalar('train/loss', 0.1)

class ToyRunner:
    # compose of different modules
    def __init__(self, name):
        self.message_hub = MessageHub.get_instance(name)  # this will create a global MessageHub instance
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

### Add custom logs

Users can update scalars in MessageHub anywhere in any module. All data in `log_scalars` with valid keys are exported to user defined backends after statistical methods.

```{note}
Only those data in `log_scalars` with keys prefixed with train/val/test are exported.
```

```python
class CustomModule:
    def __init__(self):
        self.message_hub = MessageHub.get_current_instance()

    def custom_method(self):
        self.message_hub.update_scalar('train/a', 100)
        self.message_hub.update_scalars({'train/b': 1, 'train/c': 2})
```

By default, the latest value of the custom data(a, b and c) are exported. Users can also configure the [LogProcessor](mmengine.runner.LogProcessor) to switch between statistical methods.

## LogProcessor

Users can configure the LogProcessor to specify the statistical methods and extra arguments. By default, learning rates are displayed by the latest value, while losses and iteration time are counted with an iteration-based smooth method.

### Minimum example

```python
log_processor = dict(
    window_size=10
)
```

In this configuration, losses and iteration time will be averaged in the latest 10 iterations. The output might be:

```bash
04/15 12:34:24 - mmengine - INFO - Iter [10/12]  , eta: 0:00:00, time: 0.003, data_time: 0.002, loss: 0.13
```

### Custom statistical methods

Users can configure the `custom_cfg` list to specify the statistical method. Each element in `custom_cfg` must be a dict consisting of the following keys:

- `data_src`: Required argument representing the data source of the log. A data source may have multiple statistical methods. Default sources, which are automatically added to logs, include all keys in loss dict(i.e. `loss`), learning rate(`lr`) and iteration time(`time` & `data_time`). Besides, all scalars updated by MessageHub's `update_scalar`/`update_scalars` methods with valid keys are configurable data sources, but be aware that the prefix('train/', 'val/', 'test/') should be removed.
- `method_name`: Required argument representing the statistical method. It supports both built-in methods and custom methods.
- `log_name`: Optional argument representing the output name after statistics. If not specified, the new log will overwrite the old one.
- Other arguments: Extra arguments needed by your specified method. `window_size` is a special key, which can be either an int, 'epoch' or 'global'. LogProcessor will parse these arguments and return statistical result based on iteration/epoch/global smooth.

1. Overwrite the old statistical method

```python
log_processor = dict(
    window_size=10,
    by_epoch=True,
    custom_cfg=[
        dict(data_src='loss',
             method_name='mean',
             window_size=100)])
```

In this configuration, LogProcessor will overwrite the default window size 10 by a larger window size 100 and output the mean value to 'loss' field in logs.

```bash
04/15 12:34:24 - mmengine - INFO - Iter [10/12]  , eta: 0:00:00, time: 0.003, data_time: 0.002, loss: 0.11
```

2. New statistical method without overwriting

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

In order to export logs with clear hierarchies, unified formats and less disturbation from third-party logging systems, MMengine implements a `MMLogger` class based on `logging`. It is derived from ManagerMixin. Compared with `logging.logger`, it enables accessing logger in current runner without knowing the logger name.

### Instantiate MMLogger

Users can create a global logger by calling `get_instance`. The default log format is shown as below

```python
logger = MMLogger.get_instance('mmengine', log_level='INFO')
logger.info("this is a test")
# 04/15 14:01:11 - mmengine - INFO - this is a test
```

Apart from user defined messages, the logger will also export timestamps, logger name and log level. ERROR messages are treated specially with red highlight and extra information like error locations.

```python
logger = MMLogger.get_instance('mmengine', log_level='INFO')
logger.error('division by zero')
# 04/15 14:01:56 - mmengine - ERROR - /mnt/d/PythonCode/DeepLearning/OpenMMLab/mmengine/a.py - <module> - 4 - division by zero
```

### Export logs

When `get_instance` is invoked with log_file argument, logs will be additionally exported to local storage in text format.

```Python
logger = MMLogger.get_instance('mmengine', log_file='tmp.log', log_level='INFO')
logger.info("this is a test")
# 04/15 14:01:11 - mmengine - INFO - this is a test
```

`tmp/tmp.log`:

```text
04/15 14:01:11 - mmengine - INFO - this is a test
```

Since distributed applications will create multiple log files, we add a directory with the same name to the exported log file name. Logs from different processes are all saved in this directory. Therefore, the actual log file path in the above example is `tmp/tmp.log`.

### Export logs in distributed training

When training with pytorch distributed methods, users can set `distributed=True` or `log_level='DEBUG'` in config file to export multiple logs from all processes. If not specified, only master process will export log file.

```python
logger = MMLogger.get_instance('mmengine', log_file='tmp.log', distributed=True, log_level='INFO')
# or
# logger = MMLogger.get_instance('mmengine', log_file='tmp.log', log_level='DEBUG')
```

In the case of multiple processes in a single node, or multiple processes in multiple nodes with shared storage, the exported log files have the following hierarchy

```text
#  shared storage case
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

In the case of multiple processes in multiple nodes without storage, logs are organized as follows

```text
# without shared storage
# node 0:
work_dir/20230228_141908
├── 20230306_183634_${hostname}_device0_rank0.log
├── 20230306_183634_${hostname}_device1_rank1.log
├── 20230306_183634_${hostname}_device2_rank2.log
├── 20230306_183634_${hostname}_device3_rank3.log
├── 20230306_183634_${hostname}_device4_rank4.log
├── 20230306_183634_${hostname}_device5_rank5.log
├── 20230306_183634_${hostname}_device6_rank6.log
├── 20230306_183634_${hostname}_device7_rank7.log

# node 7:
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
