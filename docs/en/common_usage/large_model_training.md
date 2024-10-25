# Training Big Models

When training large models, significant resources are required. A single GPU memory is often insufficient to meet the training needs. As a result, techniques for training large models have been developed, and one typical approach is [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/#zero-overview). DeepSpeed ZeRO supports optimizer, gradient, and parameter sharding.

To provide more flexibility in supporting large model training techniques, starting from MMEngine v0.8.0, we have introduced a new runner called [FlexibleRunner](mmengine.runner.FlexibleRunner) and multiple abstract [Strategies](../api/strategy).

```{warning}
The new FlexibleRunner and Strategy are still in the experimental stage, and their interfaces may change in future versions.
```

The following example code is excerpted from [examples/distributed_training_with_flexible_runner.py](https://github.com/open-mmlab/mmengine/blob/main/examples/distributed_training_with_flexible_runner.py).

## DeepSpeed

[DeepSpeed](https://github.com/microsoft/DeepSpeed/tree/master) is an open-source distributed framework based on PyTorch, developed by Microsoft. It supports training strategies such as `ZeRO`, `3D-Parallelism`, `DeepSpeed-MoE`, and `ZeRO-Infinity`.

Starting from MMEngine v0.8.0, MMEngine supports training models using DeepSpeed.

To use DeepSpeed, you need to install it first by running the following command:

```bash
pip install deepspeed
```

After installing DeepSpeed, you need to configure the `strategy` and `optim_wrapper` parameters of FlexibleRunner as follows:

- strategy: Set `type='DeepSpeedStrategy'` and configure other parameters. See [DeepSpeedStrategy](mmengine._strategy.DeepSpeedStrategy) for more details.
- optim_wrapper: Set `type='DeepSpeedOptimWrapper'` and configure other parameters. See [DeepSpeedOptimWrapper](mmengine._strategy.deepspeed.DeepSpeedOptimWrapper) for more details.

Here is an example configuration related to DeepSpeed:

```python
from mmengine.runner._flexible_runner import FlexibleRunner

# set `type='DeepSpeedStrategy'` and configure other parameters
strategy = dict(
    type='DeepSpeedStrategy',
    fp16=dict(
        enabled=True,
        fp16_master_weights_and_grads=False,
        loss_scale=0,
        loss_scale_window=500,
        hysteresis=2,
        min_loss_scale=1,
        initial_scale_power=15,
    ),
    inputs_to_half=[0],
    zero_optimization=dict(
        stage=3,
        allgather_partitions=True,
        reduce_scatter=True,
        allgather_bucket_size=50000000,
        reduce_bucket_size=50000000,
        overlap_comm=True,
        contiguous_gradients=True,
        cpu_offload=False),
)

# set `type='DeepSpeedOptimWrapper'` and configure other parameters
optim_wrapper = dict(
    type='DeepSpeedOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3))

# construct FlexibleRunner
runner = FlexibleRunner(
    model=MMResNet50(),
    work_dir='./work_dirs',
    strategy=strategy,
    train_dataloader=train_dataloader,
    optim_wrapper=optim_wrapper,
    param_scheduler=dict(type='LinearLR'),
    train_cfg=dict(by_epoch=True, max_epochs=10, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy))

# start training
runner.train()
```

Using two GPUs to launch distributed training:

```bash
torchrun --nproc-per-node 2 examples/distributed_training_with_flexible_runner.py --use-deepspeed
```

<details>
<summary>training log</summary>

```
07/03 13:04:17 - mmengine - INFO - Epoch(train)  [1][ 10/196]  lr: 3.3333e-04  eta: 0:13:14  time: 0.4073  data_time: 0.0335  memory: 970  loss: 6.1887
07/03 13:04:19 - mmengine - INFO - Epoch(train)  [1][ 20/196]  lr: 3.3333e-04  eta: 0:09:39  time: 0.1904  data_time: 0.0327  memory: 970  loss: 2.5746
07/03 13:04:21 - mmengine - INFO - Epoch(train)  [1][ 30/196]  lr: 3.3333e-04  eta: 0:08:32  time: 0.1993  data_time: 0.0342  memory: 970  loss: 2.4180
07/03 13:04:23 - mmengine - INFO - Epoch(train)  [1][ 40/196]  lr: 3.3333e-04  eta: 0:08:01  time: 0.2052  data_time: 0.0368  memory: 970  loss: 2.3682
07/03 13:04:25 - mmengine - INFO - Epoch(train)  [1][ 50/196]  lr: 3.3333e-04  eta: 0:07:39  time: 0.2013  data_time: 0.0356  memory: 970  loss: 2.3025
07/03 13:04:27 - mmengine - INFO - Epoch(train)  [1][ 60/196]  lr: 3.3333e-04  eta: 0:07:25  time: 0.2025  data_time: 0.0353  memory: 970  loss: 2.2078
07/03 13:04:29 - mmengine - INFO - Epoch(train)  [1][ 70/196]  lr: 3.3333e-04  eta: 0:07:13  time: 0.1999  data_time: 0.0352  memory: 970  loss: 2.2045
07/03 13:04:31 - mmengine - INFO - Epoch(train)  [1][ 80/196]  lr: 3.3333e-04  eta: 0:07:04  time: 0.2013  data_time: 0.0350  memory: 970  loss: 2.1709
07/03 13:04:33 - mmengine - INFO - Epoch(train)  [1][ 90/196]  lr: 3.3333e-04  eta: 0:06:56  time: 0.1975  data_time: 0.0341  memory: 970  loss: 2.2070
07/03 13:04:35 - mmengine - INFO - Epoch(train)  [1][100/196]  lr: 3.3333e-04  eta: 0:06:49  time: 0.1993  data_time: 0.0347  memory: 970  loss: 2.0891
07/03 13:04:37 - mmengine - INFO - Epoch(train)  [1][110/196]  lr: 3.3333e-04  eta: 0:06:44  time: 0.1995  data_time: 0.0357  memory: 970  loss: 2.0700
07/03 13:04:39 - mmengine - INFO - Epoch(train)  [1][120/196]  lr: 3.3333e-04  eta: 0:06:38  time: 0.1966  data_time: 0.0342  memory: 970  loss: 1.9983
07/03 13:04:41 - mmengine - INFO - Epoch(train)  [1][130/196]  lr: 3.3333e-04  eta: 0:06:37  time: 0.2216  data_time: 0.0341  memory: 970  loss: 1.9409
07/03 13:04:43 - mmengine - INFO - Epoch(train)  [1][140/196]  lr: 3.3333e-04  eta: 0:06:32  time: 0.1944  data_time: 0.0336  memory: 970  loss: 1.9800
07/03 13:04:45 - mmengine - INFO - Epoch(train)  [1][150/196]  lr: 3.3333e-04  eta: 0:06:27  time: 0.1946  data_time: 0.0338  memory: 970  loss: 1.9356
07/03 13:04:47 - mmengine - INFO - Epoch(train)  [1][160/196]  lr: 3.3333e-04  eta: 0:06:22  time: 0.1937  data_time: 0.0333  memory: 970  loss: 1.8145
07/03 13:04:49 - mmengine - INFO - Epoch(train)  [1][170/196]  lr: 3.3333e-04  eta: 0:06:18  time: 0.1941  data_time: 0.0335  memory: 970  loss: 1.8525
07/03 13:04:51 - mmengine - INFO - Epoch(train)  [1][180/196]  lr: 3.3333e-04  eta: 0:06:17  time: 0.2204  data_time: 0.0341  memory: 970  loss: 1.7637
07/03 13:04:53 - mmengine - INFO - Epoch(train)  [1][190/196]  lr: 3.3333e-04  eta: 0:06:14  time: 0.1998  data_time: 0.0345  memory: 970  loss: 1.7523
```

</details>

## FullyShardedDataParallel (FSDP)

PyTorch has supported training with FullyShardedDataParallel (FSDP) since version v1.11. However, due to its evolving interface, we only support PyTorch versions 2.0.0 and above.

To use FSDP, you need to configure the `'strategy'` parameter of FlexibleRunner by specifying `type='FSDPStrategy'` and configuring the parameters. For detailed information about it, you can refer to [FSDPStrategy](mmengine._strategy.FSDPStrategy).

Here is an example configuration related to FSDP:

```python
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
size_based_auto_wrap_policy = partial(
    size_based_auto_wrap_policy, min_num_params=1e7)

# set `type='FSDPStrategy'` and configure other parameters
strategy = dict(
    type='FSDPStrategy',
    model_wrapper=dict(auto_wrap_policy=size_based_auto_wrap_policy))

# set `type='AmpOptimWrapper'` and configure other parameters
optim_wrapper = dict(
    type='AmpOptimWrapper', optimizer=dict(type='AdamW', lr=1e-3))

# construct FlexibleRunner
runner = FlexibleRunner(
    model=MMResNet50(),
    work_dir='./work_dirs',
    strategy=strategy,
    train_dataloader=train_dataloader,
    optim_wrapper=optim_wrapper,
    param_scheduler=dict(type='LinearLR'),
    train_cfg=dict(by_epoch=True, max_epochs=10, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy))

# start training
runner.train()
```

Using two GPUs to launch distributed training:

```bash
torchrun --nproc-per-node 2 examples/distributed_training_with_flexible_runner.py --use-fsdp
```

<details>
<summary>training log</summary>

```
07/03 13:05:37 - mmengine - INFO - Epoch(train)  [1][ 10/196]  lr: 3.3333e-04  eta: 0:08:28  time: 0.2606  data_time: 0.0330  memory: 954  loss: 6.1265
07/03 13:05:38 - mmengine - INFO - Epoch(train)  [1][ 20/196]  lr: 3.3333e-04  eta: 0:05:18  time: 0.0673  data_time: 0.0325  memory: 954  loss: 2.5584
07/03 13:05:39 - mmengine - INFO - Epoch(train)  [1][ 30/196]  lr: 3.3333e-04  eta: 0:04:13  time: 0.0666  data_time: 0.0320  memory: 954  loss: 2.4816
07/03 13:05:39 - mmengine - INFO - Epoch(train)  [1][ 40/196]  lr: 3.3333e-04  eta: 0:03:41  time: 0.0666  data_time: 0.0321  memory: 954  loss: 2.3695
07/03 13:05:40 - mmengine - INFO - Epoch(train)  [1][ 50/196]  lr: 3.3333e-04  eta: 0:03:21  time: 0.0671  data_time: 0.0324  memory: 954  loss: 2.3208
07/03 13:05:41 - mmengine - INFO - Epoch(train)  [1][ 60/196]  lr: 3.3333e-04  eta: 0:03:08  time: 0.0667  data_time: 0.0320  memory: 954  loss: 2.2431
07/03 13:05:41 - mmengine - INFO - Epoch(train)  [1][ 70/196]  lr: 3.3333e-04  eta: 0:02:58  time: 0.0667  data_time: 0.0320  memory: 954  loss: 2.1873
07/03 13:05:42 - mmengine - INFO - Epoch(train)  [1][ 80/196]  lr: 3.3333e-04  eta: 0:02:51  time: 0.0669  data_time: 0.0320  memory: 954  loss: 2.2006
07/03 13:05:43 - mmengine - INFO - Epoch(train)  [1][ 90/196]  lr: 3.3333e-04  eta: 0:02:45  time: 0.0671  data_time: 0.0324  memory: 954  loss: 2.1547
07/03 13:05:43 - mmengine - INFO - Epoch(train)  [1][100/196]  lr: 3.3333e-04  eta: 0:02:40  time: 0.0667  data_time: 0.0321  memory: 954  loss: 2.1361
07/03 13:05:44 - mmengine - INFO - Epoch(train)  [1][110/196]  lr: 3.3333e-04  eta: 0:02:36  time: 0.0668  data_time: 0.0320  memory: 954  loss: 2.0405
07/03 13:05:45 - mmengine - INFO - Epoch(train)  [1][120/196]  lr: 3.3333e-04  eta: 0:02:32  time: 0.0669  data_time: 0.0320  memory: 954  loss: 2.0228
07/03 13:05:45 - mmengine - INFO - Epoch(train)  [1][130/196]  lr: 3.3333e-04  eta: 0:02:29  time: 0.0670  data_time: 0.0324  memory: 954  loss: 2.0375
07/03 13:05:46 - mmengine - INFO - Epoch(train)  [1][140/196]  lr: 3.3333e-04  eta: 0:02:26  time: 0.0664  data_time: 0.0320  memory: 954  loss: 1.9926
07/03 13:05:47 - mmengine - INFO - Epoch(train)  [1][150/196]  lr: 3.3333e-04  eta: 0:02:24  time: 0.0668  data_time: 0.0320  memory: 954  loss: 1.9820
07/03 13:05:47 - mmengine - INFO - Epoch(train)  [1][160/196]  lr: 3.3333e-04  eta: 0:02:22  time: 0.0674  data_time: 0.0325  memory: 954  loss: 1.9728
07/03 13:05:48 - mmengine - INFO - Epoch(train)  [1][170/196]  lr: 3.3333e-04  eta: 0:02:20  time: 0.0666  data_time: 0.0320  memory: 954  loss: 1.9359
07/03 13:05:49 - mmengine - INFO - Epoch(train)  [1][180/196]  lr: 3.3333e-04  eta: 0:02:18  time: 0.0667  data_time: 0.0321  memory: 954  loss: 1.9488
07/03 13:05:49 - mmengine - INFO - Epoch(train)  [1][190/196]  lr: 3.3333e-04  eta: 0:02:16  time: 0.0671  data_time: 0.0323  memory: 954  loss: 1.9023\
```

</details>

## ColossalAI

[ColossalAI](https://colossalai.org/) is a comprehensive large-scale model training system that utilizes efficient parallelization techniques. Starting from MMEngine v0.9.0, it supports training models using optimization strategies from the ZeRO series in ColossalAI.

Install ColossalAI with a version greater than v0.3.1. This version requirement is due to a [bug](https://github.com/hpcaitech/ColossalAI/issues/4393) in v0.3.1 that causes some program blocking, which has been fixed in later versions. If the highest available version of ColossalAI is still v0.3.1, it is recommended to install ColossalAI from the source code on the main branch.

```{note}
Note that if you encounter compilation errors like `nvcc fatal: Unsupported gpu architecture 'compute_90'` and your PyTorch version is higher than 2.0, you need to git clone the source code and follow the modifications in this [PR](https://github.com/hpcaitech/ColossalAI/pull/4357) before proceeding with the installation.
```

```bash
pip install git+https://github.com/hpcaitech/ColossalAI
```

If the latest version of ColossalAI is higher than v0.3.1, you can directly install it using pip:

```bash
pip install colossalai
```

Once ColossalAI is installed, configure the `strategy` and `optim_wrapper` parameters for FlexibleRunner:

- `strategy`: Specify `type='ColossalAIStrategy'` and configure the parameters. Detailed parameter descriptions can be found in [ColossalAIStrategy](mmengine._strategy.ColossalAIStrategy).
- `optim_wrapper`: Default to no `type` parameter or specify `type=ColossalAIOptimWrapper`. It is recommended to choose `HybridAdam` as the optimizer type. Other configurable types are listed in [ColossalAIOptimWrapper](mmengine._strategy.ColossalAIOptimWrapper).

Here's the configuration related to ColossalAI:

```python
from mmengine.runner._flexible_runner import FlexibleRunner

strategy = dict(type='ColossalAIStrategy')
optim_wrapper = dict(optimizer=dict(type='HybridAdam', lr=1e-3))

# Initialize FlexibleRunner
runner = FlexibleRunner(
    model=MMResNet50(),
    work_dir='./work_dirs',
    strategy=strategy,
    train_dataloader=train_dataloader,
    optim_wrapper=optim_wrapper,
    param_scheduler=dict(type='LinearLR'),
    train_cfg=dict(by_epoch=True, max_epochs=10, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy))

# Start training
runner.train()
```

To initiate distributed training using two GPUs:

```bash
torchrun --nproc-per-node 2 examples/distributed_training_with_flexible_runner.py --use-colossalai
```

<details>
<summary>Training Logs</summary>

```
08/18 11:56:34 - mmengine - INFO - Epoch(train) [1][ 10/196] lr: 3.3333e-04 eta: 0:10:31 time: 0.3238  data_time: 0.0344  memory: 597  loss: 3.8766
08/18 11:56:35 - mmengine - INFO - Epoch(train)  [1][ 20/196]  lr: 3.3333e-04  eta: 0:06:56  time: 0.1057  data_time: 0.0338  memory: 597  loss: 2.3797
08/18 11:56:36 - mmengine - INFO - Epoch(train)  [1][ 30/196]  lr: 3.3333e-04  eta: 0:05:45  time: 0.1068  data_time: 0.0342  memory: 597  loss: 2.3219
08/18 11:56:37 - mmengine - INFO - Epoch(train)  [1][ 40/196]  lr: 3.3333e-04  eta: 0:05:08  time: 0.1059  data_time: 0.0337  memory: 597  loss: 2.2641
08/18 11:56:38 - mmengine - INFO - Epoch(train)  [1][ 50/196]  lr: 3.3333e-04  eta: 0:04:45  time: 0.1062  data_time: 0.0338  memory: 597  loss: 2.2250
08/18 11:56:40 - mmengine - INFO - Epoch(train)  [1][ 60/196]  lr: 3.3333e-04  eta: 0:04:31  time: 0.1097  data_time: 0.0339  memory: 597  loss: 2.1672
08/18 11:56:41 - mmengine - INFO - Epoch(train)  [1][ 70/196]  lr: 3.3333e-04  eta: 0:04:21  time: 0.1096  data_time: 0.0340  memory: 597  loss: 2.1688
08/18 11:56:42 - mmengine - INFO - Epoch(train)  [1][ 80/196]  lr: 3.3333e-04  eta: 0:04:13  time: 0.1098  data_time: 0.0338  memory: 597  loss: 2.1781
08/18 11:56:43 - mmengine - INFO - Epoch(train)  [1][ 90/196]  lr: 3.3333e-04  eta: 0:04:06  time: 0.1097  data_time: 0.0338  memory: 597  loss: 2.0938
08/18 11:56:44 - mmengine - INFO - Epoch(train)  [1][100/196]  lr: 3.3333e-04  eta: 0:04:01  time: 0.1097  data_time: 0.0339  memory: 597  loss: 2.1078
08/18 11:56:45 - mmengine - INFO - Epoch(train)  [1][110/196]  lr: 3.3333e-04  eta: 0:04:01  time: 0.1395  data_time: 0.0340  memory: 597  loss: 2.0141
08/18 11:56:46 - mmengine - INFO - Epoch(train)  [1][120/196]  lr: 3.3333
```
