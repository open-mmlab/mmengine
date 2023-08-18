# Debug Tricks

## Set the Dataset's length

During the process of debugging code, sometimes it is necessary to train for several epochs, such as debugging the validation process or checking whether the checkpoint saving meets expectations. However, if the dataset is too large, it may take a long time to complete one epoch, in which case the length of the dataset can be set. Note that only datasets inherited from [BaseDataset](mmengine.dataset.BaseDataset) support this feature, and the usage of BaseDataset can be found in the [BaseDataset](../advanced_tutorials/basedataset.md).

Take MMPretrain as an example (Refer to the [documentation](https://mmpretrain.readthedocs.io/en/latest/get_started.html) for installing MMPretrain).

Launch training

```bash
python tools/train.py configs/resnet/resnet18_8xb16_cifar10.py
```

Here is part of the training log, where `3125` represents the number of iterations to be performed.

```
02/20 14:43:11 - mmengine - INFO - Epoch(train)   [1][ 100/3125]  lr: 1.0000e-01  eta: 6:12:01  time: 0.0149  data_time: 0.0003  memory: 214  loss: 2.0611
02/20 14:43:13 - mmengine - INFO - Epoch(train)   [1][ 200/3125]  lr: 1.0000e-01  eta: 4:23:08  time: 0.0154  data_time: 0.0003  memory: 214  loss: 2.0963
02/20 14:43:14 - mmengine - INFO - Epoch(train)   [1][ 300/3125]  lr: 1.0000e-01  eta: 3:46:27  time: 0.0146  data_time: 0.0003  memory: 214  loss: 1.9858
```

Turn off the training and set `indices` as `5000` in the `dataset` field in [configs/base/datasets/cifar10_bs16.py](https://github.com/open-mmlab/mmpretrain/blob/main/configs/_base_/datasets/cifar100_bs16.py).

```python
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/cifar10',
        test_mode=False,
        indices=5000,  # set indices=5000，represent every epoch only iterator 5000 samples
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
```

Launch training again

```bash
python tools/train.py configs/resnet/resnet18_8xb16_cifar10.py
```

As we can see, the number of iterations has changed to `313`. Compared to before, this can complete an epoch faster.

```
02/20 14:44:58 - mmengine - INFO - Epoch(train)   [1][100/313]  lr: 1.0000e-01  eta: 0:31:09  time: 0.0154  data_time: 0.0004  memory: 214  loss: 2.1852
02/20 14:44:59 - mmengine - INFO - Epoch(train)   [1][200/313]  lr: 1.0000e-01  eta: 0:23:18  time: 0.0143  data_time: 0.0002  memory: 214  loss: 2.0424
02/20 14:45:01 - mmengine - INFO - Epoch(train)   [1][300/313]  lr: 1.0000e-01  eta: 0:20:39  time: 0.0143  data_time: 0.0003  memory: 214  loss: 1.814
```

## Add cfg parameter

During the process of debugging code, sometimes it is necessary to train for several epochs, such as debugging the validation process or checking whether the checkpoint saving meets expectations. However, if the dataset is too large, it may take a long time to complete one epoch, in which case the cfg parameter can be added.
Take `MMEngine` as an example（Refer to the [documentation](https://mmengine.readthedocs.io/zh_CN/latest/get_started/installation.html)for installing MMEngine）。

Example of a training script

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

Fast debugging is achieved by adding the `num_batch_per_epoch` parameter to `train_dataloader` and `val_dataloader`.

Run the training script. You can see that after running each epoch run 5 batch is over. Compared to the original, debugging is faster and more flexible.

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
