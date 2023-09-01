# 调试技巧

## 设置数据集的长度

在调试代码的过程中，有时需要训练几个 epoch，例如调试验证过程或者权重的保存是否符合期望。然而如果数据集太大，需要花费较长时间才能训完一个 epoch，这种情况下可以设置数据集的长度。注意，只有继承自 [BaseDataset](mmengine.dataset.BaseDataset) 的 Dataset 才支持这个功能，`BaseDataset` 的用法可阅读 [数据集基类（BASEDATASET）](../advanced_tutorials/basedataset.md)。

以 `MMPretrain` 为例（参考[文档](https://mmpretrain.readthedocs.io/zh_CN/latest/get_started.html)安装 MMPretrain）。

启动训练命令

```bash
python tools/train.py configs/resnet/resnet18_8xb16_cifar10.py
```

下面是训练的部分日志，其中 `3125` 表示需要迭代的次数。

```
02/20 14:43:11 - mmengine - INFO - Epoch(train)   [1][ 100/3125]  lr: 1.0000e-01  eta: 6:12:01  time: 0.0149  data_time: 0.0003  memory: 214  loss: 2.0611
02/20 14:43:13 - mmengine - INFO - Epoch(train)   [1][ 200/3125]  lr: 1.0000e-01  eta: 4:23:08  time: 0.0154  data_time: 0.0003  memory: 214  loss: 2.0963
02/20 14:43:14 - mmengine - INFO - Epoch(train)   [1][ 300/3125]  lr: 1.0000e-01  eta: 3:46:27  time: 0.0146  data_time: 0.0003  memory: 214  loss: 1.9858
```

关掉训练，然后修改 [configs/_base_/datasets/cifar10_bs16.py](https://github.com/open-mmlab/mmpretrain/blob/main/configs/_base_/datasets/cifar100_bs16.py) 中的 `dataset` 字段，设置 `indices=5000`。

```python
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/cifar10',
        test_mode=False,
        indices=5000,  # 设置 indices=5000，表示每个 epoch 只迭代 5000 个样本
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
```

重新启动训练

```bash
python tools/train.py configs/resnet/resnet18_8xb16_cifar10.py
```

可以看到，迭代次数变成了 `313`，相比原先，这样能够更快跑完一个 epoch。

```
02/20 14:44:58 - mmengine - INFO - Epoch(train)   [1][100/313]  lr: 1.0000e-01  eta: 0:31:09  time: 0.0154  data_time: 0.0004  memory: 214  loss: 2.1852
02/20 14:44:59 - mmengine - INFO - Epoch(train)   [1][200/313]  lr: 1.0000e-01  eta: 0:23:18  time: 0.0143  data_time: 0.0002  memory: 214  loss: 2.0424
02/20 14:45:01 - mmengine - INFO - Epoch(train)   [1][300/313]  lr: 1.0000e-01  eta: 0:20:39  time: 0.0143  data_time: 0.0003  memory: 214  loss: 1.814
```

## 固定训练的迭代次数（基于 epoch 的训练）

在调试代码的过程中，有时需要训练几个 epoch，例如调试验证过程或者权重的保存是否符合期望。然而如果数据集太大，需要花费较长时间才能训完一个 epoch，在这种情况下，可以配置 dataloader 的 `num_batch_per_epoch` 参数。

```{note}
`num_batch_per_epoch` 参数不是 PyTorch 中 dataloader 的原生参数，而是 MMEngine 为了实现此功能而额外添加的参数。
```

我们以[15 分钟上手 MMEngine](../get_started/15_minutes.md) 中定义的模型为例。通过在 `train_dataloader` 和 `val_dataloader` 中设置 `num_batch_per_epoch=5`，便可实现一个 epoch 只迭代 5 次。

```python
train_dataloader = dict(
    batch_size=32,
    dataset=train_set,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    num_batch_per_epoch=5)
val_dataloader = dict(
    batch_size=32,
    dataset=valid_set,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    num_batch_per_epoch=5)
runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
    launcher=args.launcher,
)
runner.train()
```

可以看到，迭代次数变成了 `5`，相比原先，这样能够更快跑完一个 epoch。

```
08/18 20:27:22 - mmengine - INFO - Epoch(train) [1][5/5]  lr: 1.0000e-03  eta: 0:00:02  time: 0.4566  data_time: 0.0074  memory: 477  loss: 6.7576
08/18 20:27:22 - mmengine - INFO - Saving checkpoint at 1 epochs
08/18 20:27:22 - mmengine - WARNING - `save_param_scheduler` is True but `self.param_schedulers` is None, so skip saving parameter schedulers
08/18 20:27:23 - mmengine - INFO - Epoch(val) [1][5/5]    accuracy: 7.5000  data_time: 0.0044  time: 0.0146
08/18 20:27:23 - mmengine - INFO - Exp name: 20230818_202715
08/18 20:27:23 - mmengine - INFO - Epoch(train) [2][5/5]  lr: 1.0000e-03  eta: 0:00:00  time: 0.2501  data_time: 0.0077  memory: 477  loss: 5.3044
08/18 20:27:23 - mmengine - INFO - Saving checkpoint at 2 epochs
08/18 20:27:24 - mmengine - INFO - Epoch(val) [2][5/5]    accuracy: 12.5000  data_time: 0.0058  time: 0.0175
```

## 检查不参与 loss 计算的参数

使用多卡训练时，当模型的参数参与了前向计算，但没有参与 loss 的计算，程序会抛出下面的错误：

```
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by
making sure all `forward` function outputs participate in calculating loss.
```

我们以[15 分钟上手 MMEngine](../get_started/15_minutes.md) 中定义的模型为例：

```python
class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels
```

将其修改为下面的代码：

```python
class MMResNet50(BaseModel):

    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()
        self.param = nn.Parameter(torch.ones(1))

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        # self.param 参与了前向计算，但 y 没有参与 loss 的计算
        y = self.param + x
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels
```

使用两张卡启动训练

```bash
torchrun --nproc-per-node 2 examples/distributed_training.py --launcher pytorch
```

程序会抛出上面提到的错误。

我们可以通过设置 `find_unused_parameters=True` 来解决这个问题，

```python
cfg = dict(
    model_wrapper_cfg=dict(
        type='MMDistributedDataParallel', find_unused_parameters=True)
)
runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
    launcher=args.launcher,
    cfg=cfg,
)
runner.train()
```

重新启动训练，可以看到程序可以正常训练并打印日志。

但是，设置 `find_unused_parameters=True` 会让程序变慢，因此，我们希望找出这些参数并分析它们没有参与 loss 计算的原因。

可以通过设置 `detect_anomalous_params=True` 来打印未被使用的参数。

```python
cfg = dict(
    model_wrapper_cfg=dict(
        type='MMDistributedDataParallel',
        find_unused_parameters=True,
        detect_anomalous_params=True),
)
```

重新启动训练，可以看到日志中打印了未参与 loss 计算的参数。

```
08/03 15:04:42 - mmengine - ERROR - mmengine/logging/logger.py - print_log - 323 - module.param with shape torch.Size([1]) is not in the computational graph
```

在找到这些参数后，我们可以分析为什么这些参数没有参与 loss 的计算。

```{important}
只应在调试时设置 `find_unused_parameters=True` 和 `detect_anomalous_params=True`。
```
