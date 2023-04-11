# Logging

[Runner](../tutorials/runner.md) will produce a lot of logs during the running process, such as loss, iteration time, learning rate, etc. MMEngine implements a flexible logging system that allows us to choose different types of log statistical methods when configuring the runner. It could help us set/get the recorded log at any location in the code.

## Flexible Logging System

Logging system is configured by passing a [LogProcessor](mmengine.runner.LogProcessor) to the runner. If no log processor is passed, the runner will use the default log processor, which is equivalent to:

```python
log_processor = dict(window_size=10, by_epoch=True, custom_cfg=None, num_digits=4)
```

The format of the output log is as follows:

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

LogProcessor will output the log in the following format:

- The prefix of the log:
  - epoch mode(`by_epoch=True`): `Epoch(train) [{current_epoch}/{current_iteration}]/{dataloader_length}`
  - iteration mode(`by_epoch=False`): `Iter(train) [{current_iteration}/{max_iteration}]`)
- Learning rate (`lr`): The learning rate of the last iteration.
- Time:
  - `time`: The averaged time for inference of the last `window_size` iterations.
  - `data_time`: The averaged time for loading data of the last `window_size` iterations.
  - `eta`: The estimated time of arrival to finish the training.
- Loss: The averaged loss output by model of the last `window_size` iterations.

```{note}
`window_size=10` by default.

The significant digits(`num_digits`) of the log is 4 by default.

Output the value of all custom logs at the last iteration by default.
```

```{warning}
log_processor outputs the epoch based log by default(`by_epoch=True`). To get an expected log matched with the `train_cfg`, we should set the same value for `by_epoch` in `train_cfg` and `log_processor`.
```

Based on the rules above, the code snippet will count the average value of the `loss1` and the `loss2` every 10 iterations.

If we want to count the global average value of `loss1`, we can set `custom_cfg` like this:

```python
runner = Runner(
    model=ToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=1),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)),
    log_processor=dict(
        custom_cfg=[
            dict(data_src='loss1',  # original loss name: loss1
                 method_name='mean',  # statistical method: mean
                 window_size='global')])  # window_size: global
)
runner.train()
```

```
08/21 02:58:49 - mmengine - INFO - Epoch(train) [1][10/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0026  data_time: 0.0007  loss1: 0.7381  loss2: 0.8446  loss: 1.5827
08/21 02:58:49 - mmengine - INFO - Epoch(train) [1][20/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0030  data_time: 0.0012  loss1: 0.4521  loss2: 0.3939  loss: 0.5600
```

`data_src` means the original loss name, `method_name` means the statistic method, `window_size` means the window size of the statistic method. Since we want to count the global average value of `loss1`, we set `window_size` to `global`.

Currently, MMEngine supports the following statistical methods:

<table class="docutils">
<thead>
<tr>
    <th>statistic method</th>
    <th>arguments</th>
    <th>function</th>
</tr>
<tr>
    <td>mean</td>
    <td>window_size</td>
    <td>statistic the average log of the last `window_size`</td>
</tr>
<tr>
    <td>min</td>
    <td>window_size</td>
    <td>statistic the minimum log of the last `window_size`</td>
</tr>
<tr>
    <td>max</td>
    <td>window_size</td>
    <td>statistic the maximum log of the last `window_size`</td>
</tr>
<tr>
    <td>current</td>
    <td>/</td>
    <td>statistic the latest</td>
</tr>
</thead>
</table>

`window_size` mentioned above could be:

- int number: The window size of the statistic method.
- `global`: Equivalent to `window_size=cur_iteration`.
- `epoch`: Equivalent to `window_size=len(dataloader)`.

If we want to statistic the average value of `loss1` of the last 10 iterations, and also want to statistic the global average value of `loss1`. We need to set `log_name` additionally:

```python
runner = Runner(
    model=ToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=1),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)),
    log_processor=dict(
        custom_cfg=[
            # log_name means the second name of loss1
            dict(data_src='loss1', log_name='loss1_global', method_name='mean', window_size='global')])
)
runner.train()
```

```
08/21 18:39:32 - mmengine - INFO - Epoch(train) [1][10/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0016  data_time: 0.0004  loss1: 0.1512  loss2: 0.3751  loss: 0.5264  loss1_global: 0.1512
08/21 18:39:32 - mmengine - INFO - Epoch(train) [1][20/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0051  data_time: 0.0036  loss1: 0.0113  loss2: 0.0856  loss: 0.0970  loss1_global: 0.0813
```

Similarly, we can also statistic the global/local maximum value of `loss` at the same time.

```python
runner = Runner(
    model=ToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=1),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)),
    log_processor=dict(custom_cfg=[
        # statistic loss1 with the local maximum value
        dict(data_src='loss1',
             log_name='loss1_local_max',
             window_size=10,
             method_name='max'),
        # statistic loss1 with the global maximum value
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

More examples can be found in [log_processor](mmengine.runner.LogProcessor).

## Customize log

The logging system could not only log the `loss`, `lr`, .etc but also collect and output the custom log. For example, if we want to statistic the intermediate `loss`:

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
        # update the intermediate `loss_tmp` in the message hub
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
        # statistic the loss_tmp with the averaged value
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

The custom log will be recorded by updating the [messagehub](mmengine.logging.MessageHub):

1. Calling `MessageHub.get_current_instance()` to get the message of runner
2. Calling `MessageHub.update_scalar` to update the custom log. The first argument means the log name with the mode prefix(`train/val/test`). The output log will only retain the log name without the mode prefix.
3. Configure statistic method of `loss_tmp` in `log_processor`. If it is not configured, only the latest value of `loss_tmp` will be logged.

## Export the debug log

Set `log_level=DEBUG` for runner, and the debug log will be exported to the `work_dir`:

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

Besides, logs of different ranks will be saved in `debug` mode if you are training your model with the shared storage. The hierarchy of the log is as follows:

```text
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

The log of Multiple machine with independent storage:

```text
# device: 0:
work_dir/
└── exp_name_logs
    ├── exp_name.log
    ├── exp_name_rank1.log
    ├── exp_name_rank2.log
    ├── exp_name_rank3.log
    ...
    └── exp_name_rank7.log

# device: 7:
work_dir/
└── exp_name_logs
    ├── exp_name_rank56.log
    ├── exp_name_rank57.log
    ├── exp_name_rank58.log
    ...
    └── exp_name_rank63.log
```
