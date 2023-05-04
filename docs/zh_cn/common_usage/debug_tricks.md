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
