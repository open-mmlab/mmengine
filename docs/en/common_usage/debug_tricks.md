# Debug Tricks

## Set the Dataset's length

Sometimes it's necessary to train several epochs during debug code, for example, verify in the debug mode or save the weight is meet expectation. However, if the Dataset is too large to take long time to train an epoch completed, in this situation can set the length of the Dataset. Attention, only inherited from [BaseDataset](mmengine.dataset.BaseDataset) can support this function, Please read usage of [BaseDataset](../advanced_tutorials/basedataset.md).

For example: Using `MMClassification` Refer [Document](https://mmclassification.readthedocs.io/en/dev-1.x/get_started.html) to install `MMClassification`.

Start training

```bash
python tools/train.py configs/resnet/resnet18_8xb16_cifar10.py
```

Following is part of the training log, `3125` represents the number of iterators.

```
02/20 14:43:11 - mmengine - INFO - Epoch(train)   [1][ 100/3125]  lr: 1.0000e-01  eta: 6:12:01  time: 0.0149  data_time: 0.0003  memory: 214  loss: 2.0611
02/20 14:43:13 - mmengine - INFO - Epoch(train)   [1][ 200/3125]  lr: 1.0000e-01  eta: 4:23:08  time: 0.0154  data_time: 0.0003  memory: 214  loss: 2.0963
02/20 14:43:14 - mmengine - INFO - Epoch(train)   [1][ 300/3125]  lr: 1.0000e-01  eta: 3:46:27  time: 0.0146  data_time: 0.0003  memory: 214  loss: 1.9858
```

Turn of the training. modify `dataset` field in file [configs/base/datasets/cifar10_bs16.py](https://github.com/open-mmlab/mmclassification/blob/dev-1.x/configs/_base_/datasets/cifar10_bs16.py), set `indices=5000`.

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

Reopen training

```bash
python tools/train.py configs/resnet/resnet18_8xb16_cifar10.py
```

Accord to following. The number of iterator has been changed to `313`, which can finished an epoch faster than before.

```
02/20 14:44:58 - mmengine - INFO - Epoch(train)   [1][100/313]  lr: 1.0000e-01  eta: 0:31:09  time: 0.0154  data_time: 0.0004  memory: 214  loss: 2.1852
02/20 14:44:59 - mmengine - INFO - Epoch(train)   [1][200/313]  lr: 1.0000e-01  eta: 0:23:18  time: 0.0143  data_time: 0.0002  memory: 214  loss: 2.0424
02/20 14:45:01 - mmengine - INFO - Epoch(train)   [1][300/313]  lr: 1.0000e-01  eta: 0:20:39  time: 0.0143  data_time: 0.0003  memory: 214  loss: 1.814
```
