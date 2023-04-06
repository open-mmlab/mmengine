# 记录日志

[执行器（Runner）](../tutorials/runner.md)在运行过程中会产生很多日志，例如损失、迭代时间、学习率等。MMEngine 实现了一套灵活的日志系统让我们能够在配置执行器时，选择不同类型日志的统计方式；在代码的任意位置，新增需要被统计的日志。

## 灵活的日志统计方式

我们可以通过在构建执行器时候配置[日志处理器](mmengine.runner.LogProcessor)，来灵活地选择日志统计方式。如果不为执行器配置日志处理器，则会按照日志处理器的默认参数构建实例，效果等价于：

```python
log_processor = dict(window_size=10, by_epoch=True, custom_cfg=None, num_digits=4)
```

其输出的日志格式如下：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mmengine.runner import Runner
from mmengine.model import BaseModel

train_dataset = [(torch.ones(1, 1), torch.ones(1, 1))] * 50
train_dataloader = DataLoader(train_dataset, batch_size=2)


class ToyModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, img, label, mode):
        feat = self.linear(img)
        loss1 = (feat - label).pow(2)
        loss2 = (feat - label).abs()
        return dict(loss1=loss1, loss2=loss2)

runner = Runner(
    model=ToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=1),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01))
)
runner.train()
```

```
08/21 02:58:41 - mmengine - INFO - Epoch(train) [1][10/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0019  data_time: 0.0004  loss1: 0.8381  loss2: 0.9007  loss: 1.7388
08/21 02:58:41 - mmengine - INFO - Epoch(train) [1][20/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0029  data_time: 0.0010  loss1: 0.1978  loss2: 0.4312  loss: 0.6290
```

以训练阶段为例，日志处理器默认会按照以下方式统计执行器输出的日志：

- 日志前缀：
  - Epoch 模式（`by_epoch=True`）：`Epoch(train) [{当前epoch次数}][{当前迭代次数}/{Dataloader 总长度}]`
  - Iter 模式（`by_epoch=False`）： `Iter(train) [{当前迭代次数}/{总迭代次数}]`
- 学习率（`lr`）：统计最近一次迭代，参数更新的学习率
- 时间
  - 迭代时间（`time`）：最近 `window_size`（日志处理器参数） 次迭代，处理一个 batch 数据（包括数据加载和模型前向推理）的平均时间
  - 数据时间（`data_time`）：最近 `window_size` 次迭代，加载一个 batch 数据的平均时间
  - 剩余时间（`eta`）：根据总迭代次数和历次迭代时间计算出来的总剩余时间，剩余时间随着迭代次数增加逐渐趋于稳定
- 损失：模型前向推理得到的各种字段的损失，默认统计最近 `window_size` 次迭代的平均损失。

默认情况下，`window_size=10`，日志处理器会统计最近 10 次迭代，损失、迭代时间、数据时间的均值。

默认情况下，所有日志的有效位数（`num_digits` 参数）为 4。

默认情况下，输出所有自定义日志最近一次迭代的值。

基于上述规则，代码示例中的日志处理器会输出 `loss1` 和 `loss2` 每 10 次迭代的均值。如果我们想统计 `loss1` 从第一次迭代开始至今的全局均值，可以这样配置：

```python
runner = Runner(
    model=ToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=1),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)),
    log_processor=dict(  # 配置日志处理器
        custom_cfg=[
            dict(data_src='loss1',  # 原日志名：loss1
                 method_name='mean',  # 统计方法：均值统计
                 window_size='global')])  # 统计窗口：全局
)
runner.train()
```

```
08/21 02:58:49 - mmengine - INFO - Epoch(train) [1][10/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0026  data_time: 0.0007  loss1: 0.7381  loss2: 0.8446  loss: 1.5827
08/21 02:58:49 - mmengine - INFO - Epoch(train) [1][20/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0030  data_time: 0.0012  loss1: 0.4521  loss2: 0.3939  loss: 0.5600
```

```{note}
log_processor 默认输出 `by_epoch=True` 格式的日志。日志格式需要和 `train_cfg` 中的 `by_epoch` 参数保持一致，例如我们想按迭代次数输出日志，就需要另 `log_processor` 和 `train_cfg` 的 `by_epoch=False`。
```

其中 `data_src` 为原日志名，`mean` 为统计方法，`global` 为统计方法的参数。这样的话，日志中统计的 `loss1` 就是全局均值。我们可以在日志处理器中配置以下统计方法：

<table class="docutils">
<thead>
<tr>
    <th>统计方法</th>
    <th>参数</th>
    <th>功能</th>
</tr>
<tr>
    <td>mean</td>
    <td>window_size</td>
    <td>统计窗口内日志的均值</td>
</tr>
<tr>
    <td>min</td>
    <td>window_size</td>
    <td>统计窗口内日志的最小值</td>
</tr>
<tr>
    <td>max</td>
    <td>window_size</td>
    <td>统计窗口内日志的最大值</td>
</tr>
<tr>
    <td>current</td>
    <td>/</td>
    <td>返回最近一次更新的日志</td>
</tr>
</thead>
</table>

其中 `window_size` 的值可以是：

- 数字：表示统计窗口的大小
- `global`：统计全局的最大、最小和均值
- `epoch`：统计一个 epoch 内的最大、最小和均值

当然我们也可以选择自定义的统计方法，详细步骤见[日志设计](../design/logging.md)。

如果我们既想统计窗口为 10 的 `loss1` 的局部均值，又想统计 `loss1` 的全局均值，则需要额外指定 `log_name`：

```python
runner = Runner(
    model=ToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=1),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)),
    log_processor=dict(
        custom_cfg=[
            # log_name 表示 loss1 重新统计后的日志名
            dict(data_src='loss1', log_name='loss1_global', method_name='mean', window_size='global')])
)
runner.train()
```

```
08/21 18:39:32 - mmengine - INFO - Epoch(train) [1][10/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0016  data_time: 0.0004  loss1: 0.1512  loss2: 0.3751  loss: 0.5264  loss1_global: 0.1512
08/21 18:39:32 - mmengine - INFO - Epoch(train) [1][20/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0051  data_time: 0.0036  loss1: 0.0113  loss2: 0.0856  loss: 0.0970  loss1_global: 0.0813
```

类似地，我们也可以统计 `loss1` 的局部最大值和全局最大值：

```python
runner = Runner(
    model=ToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=1),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)),
    log_processor=dict(custom_cfg=[
        # 统计 loss1 的局部最大值，统计窗口为 10，并在日志中重命名为 loss1_local_max
        dict(data_src='loss1',
             log_name='loss1_local_max',
             window_size=10,
             method_name='max'),
        # 统计 loss1 的全局最大值，并在日志中重命名为 loss1_local_max
        dict(
            data_src='loss1',
            log_name='loss1_global_max',
            method_name='max',
            window_size='global')
    ]))
runner.train()
```

```
08/21 03:17:26 - mmengine - INFO - Epoch(train) [1][10/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0021  data_time: 0.0006  loss1: 1.8495  loss2: 1.3427  loss: 3.1922  loss1_local_max: 2.8872  loss1_global_max: 2.8872
08/21 03:17:26 - mmengine - INFO - Epoch(train) [1][20/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0024  data_time: 0.0010  loss1: 0.5464  loss2: 0.7251  loss: 1.2715  loss1_local_max: 2.8872  loss1_global_max: 2.8872
```

更多配置规则见[日志处理器文档](mmengine.runner.LogProcessor)

## 自定义统计内容

除了 MMEngine 默认的日志统计类型，如损失、迭代时间、学习率，用户也可以自行添加日志的统计内容。例如我们想统计损失的中间结果，可以这样做：

```python
from mmengine.logging import MessageHub


class ToyModel(BaseModel):

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, img, label, mode):
        feat = self.linear(img)
        loss_tmp = (feat - label).abs()
        loss = loss_tmp.pow(2)

        message_hub = MessageHub.get_current_instance()
        # 在日志中额外统计 `loss_tmp`
        message_hub.update_scalar('train/loss_tmp', loss_tmp.sum())
        return dict(loss=loss)


runner = Runner(
    model=ToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=1),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)),
    log_processor=dict(
        custom_cfg=[
        # 统计 loss_tmp 的局部均值
            dict(
                data_src='loss_tmp',
                window_size=10,
                method_name='mean')
        ]
    )
)
runner.train()
```

```
08/21 03:40:31 - mmengine - INFO - Epoch(train) [1][10/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0026  data_time: 0.0008  loss_tmp: 0.0097  loss: 0.0000
08/21 03:40:31 - mmengine - INFO - Epoch(train) [1][20/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0028  data_time: 0.0013  loss_tmp: 0.0065  loss: 0.0000
```

通过调用[消息枢纽](mmengine.logging.MessageHub)的接口实现自定义日志的统计，具体步骤如下：

1. 调用 `get_current_instance` 接口获取执行器的消息枢纽。
2. 调用 `update_scalar` 接口更新日志内容，其中第一个参数为日志的名称，日志名称以 `train/`，`val/`，`test/` 前缀打头，用于区分训练状态，然后才是实际的日志名，如上例中的 `train/loss_tmp`,这样统计的日志中就会出现 `loss_tmp`。
3. 配置日志处理器，以均值的方式统计 `loss_tmp`。如果不配置，日志里显示 `loss_tmp` 最近一次更新的值。

## 输出调试日志

初始化执行器（Runner）时，将 `log_level` 设置成 `debug`。这样终端上就会额外输出日志等级为 `debug` 的日志

```python
runner = Runner(
    model=ToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    log_level='DEBUG',
    train_cfg=dict(by_epoch=True, max_epochs=1),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)))
runner.train()
```

```
08/21 18:16:22 - mmengine - DEBUG - Get class `LocalVisBackend` from "vis_backend" registry in "mmengine"
08/21 18:16:22 - mmengine - DEBUG - An `LocalVisBackend` instance is built from registry, its implementation can be found in mmengine.visualization.vis_backend
08/21 18:16:22 - mmengine - DEBUG - Get class `RuntimeInfoHook` from "hook" registry in "mmengine"
08/21 18:16:22 - mmengine - DEBUG - An `RuntimeInfoHook` instance is built from registry, its implementation can be found in mmengine.hooks.runtime_info_hook
08/21 18:16:22 - mmengine - DEBUG - Get class `IterTimerHook` from "hook" registry in "mmengine"
...
```

此外，分布式训练时，`DEBUG` 模式还会分进程存储日志。单机多卡，或者多机多卡但是共享存储的情况下，导出的分布式日志路径如下

```text
# 共享存储
./tmp
├── tmp.log
├── tmp_rank1.log
├── tmp_rank2.log
├── tmp_rank3.log
├── tmp_rank4.log
├── tmp_rank5.log
├── tmp_rank6.log
└── tmp_rank7.log
...
└── tmp_rank63.log
```

多机多卡，独立存储的情况：

```text
# 独立存储
# 设备 0：
work_dir/
└── exp_name_logs
    ├── exp_name.log
    ├── exp_name_rank1.log
    ├── exp_name_rank2.log
    ├── exp_name_rank3.log
    ...
    └── exp_name_rank7.log

# 设备 7：
work_dir/
└── exp_name_logs
    ├── exp_name_rank56.log
    ├── exp_name_rank57.log
    ├── exp_name_rank58.log
    ...
    └── exp_name_rank63.log
```

如果想要更加深入的了解 MMEngine 的日志系统，可以参考[日志系统设计](../design/logging.md)。
