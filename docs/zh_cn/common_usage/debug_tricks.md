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

## 增加cfg参数

在调试代码的过程中，有时需要训练几个 epoch，例如调试验证过程或者权重的保存是否符合期望。然而如果数据集太大，需要花费较长时间才能训完一个 epoch，这种情况下可以增加cfg参数。
以 `MMEngine` 为例（参考[文档](https://mmengine.readthedocs.io/zh_CN/latest/get_started/installation.html)安装 MMEngine）。

训练脚本示例

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD
from torch.utils.data import DataLoader

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import FlexibleRunner


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


class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        return dict(accuracy=100 * total_correct / total_size)


norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
train_dataloader = DataLoader(batch_size=32,
                              shuffle=True,
                              dataset=torchvision.datasets.CIFAR10(
                                  'data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(**norm_cfg)
                                  ])))

val_dataloader = DataLoader(batch_size=32,
                            shuffle=False,
                            dataset=torchvision.datasets.CIFAR10(
                                'data/cifar10',
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(**norm_cfg)
                                ])))

runner = FlexibleRunner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1, num_batch_per_epoch=10),
    val_dataloader=val_dataloader,
    val_cfg=dict(num_batch_per_epoch=10),
    val_evaluator=dict(type=Accuracy)
)
runner.train()
```

通过在`train_cfg`和`val_cfg`新增`num_batch_per_epoch`参数实现快速调试。

运行训练脚本。可以看到跑完每个epoch跑完10个batch就结束了。相比原来，调试更加快速和灵活。

```
08/07 16:35:45 - mmengine - INFO - Epoch(train) [1][  10/1563]  lr: 1.0000e-03  eta: 1:15:03  time: 0.5770  data_time: 0.0075  memory: 477  loss: 5.0847
08/07 16:35:45 - mmengine - INFO - Saving checkpoint at 1 epochs
08/07 16:35:46 - mmengine - INFO - Epoch(val) [1][ 10/313]    eta: 0:00:03  time: 0.0131  data_time: 0.0037  memory: 477
08/07 16:35:46 - mmengine - INFO - Epoch(val) [1][313/313]    accuracy: 13.0682  data_time: 0.0038  time: 0.0130
08/07 16:35:46 - mmengine - INFO - Epoch(train) [2][  10/1563]  lr: 1.0000e-03  eta: 0:38:13  time: 0.0360  data_time: 0.0066  memory: 477  loss: 2.7406
08/07 16:35:46 - mmengine - INFO - Saving checkpoint at 2 epochs
08/07 16:35:47 - mmengine - INFO - Epoch(val) [2][ 10/313]    eta: 0:00:03  time: 0.0104  data_time: 0.0036  memory: 477
08/07 16:35:47 - mmengine - INFO - Epoch(val) [2][313/313]    accuracy: 12.5000  data_time: 0.0036  time: 0.0117
```
