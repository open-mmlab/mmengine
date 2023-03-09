# Debug Tricks

## Set the Dataset's length

During the process of debugging code, sometimes it is necessary to train for several epochs, such as debugging the validation process or checking whether the checkpoint saving meets expectations. However, if the dataset is too large, it may take a long time to complete one epoch, in which case the length of the dataset can be set. Note that only datasets inherited from [BaseDataset](mmengine.dataset.BaseDataset) support this feature, and the usage of BaseDataset can be found in the [BaseDataset](../advanced_tutorials/basedataset.md).

Take MMClassification as an example (Refer to the [documentation](https://mmclassification.readthedocs.io/en/dev-1.x/get_started.html) for installing MMClassification).

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

Turn off the training and set `indices` as `5000` in the `dataset` field in [configs/base/datasets/cifar10_bs16.py](https://github.com/open-mmlab/mmclassification/blob/dev-1.x/configs/_base_/datasets/cifar10_bs16.py).

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
