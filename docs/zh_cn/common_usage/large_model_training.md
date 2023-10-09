# 大模型训练

在训练大模型时，需要庞大的资源。单卡显存通常不能满足训练的需要，因此出现了大模型训练技术，其中典型的一种是 [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/#zero-overview)。DeepSpedd ZeRO 支持切分优化器、梯度以及参数。

为了更加灵活地支持大模型训练技术，从 MMEngine v0.8.0 开始，我们提供了新的执行器 [FlexibleRunner](mmengine.runner.FlexibleRunner) 和多个抽象策略 [Strategy](../api/strategy)。

```{warning}
新的执行器 FlexibleRunner 和 Strategy 还处于实验性阶段，在将来的版本中，它们的接口有可能会发生变化。
```

下面的示例代码摘自 [examples/distributed_training_with_flexible_runner.py](https://github.com/open-mmlab/mmengine/blob/main/examples/distributed_training_with_flexible_runner.py)。

## DeepSpeed

[DeepSpeed](https://github.com/microsoft/DeepSpeed/tree/master) 是微软开源的基于 PyTorch 的分布式框架，其支持了 `ZeRO`, `3D-Parallelism`, `DeepSpeed-MoE`, `ZeRO-Infinity` 等训练策略。
MMEngine 自 v0.8.0 开始支持使用 DeepSpeed 进行模型的训练。

使用 DeepSpeed 前需安装 deepspeed：

```bash
pip install deepspeed
```

安装好 deepspeed 后，需配置 FlexibleRunner 的 strategy 和 optim_wrapper 参数：

- strategy：指定 `type='DeepSpeedStrategy'` 并配置参数。参数的详细介绍可阅读 [DeepSpeedStrategy](mmengine._strategy.DeepSpeedStrategy)。
- optim_wrapper：指定 `type='DeepSpeedOptimWrapper'` 并配置参数。参数的详细介绍可阅读 [DeepSpeedOptimWrapper](mmengine._strategy.deepspeed.DeepSpeedOptimWrapper)。

下面是 DeepSpeed 相关的配置：

```python
from mmengine.runner._flexible_runner import FlexibleRunner

# 指定 DeepSpeedStrategy 并配置参数
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

# 指定 DeepSpeedOptimWrapper 并配置参数
optim_wrapper = dict(
    type='DeepSpeedOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3))

# 初始化 FlexibleRunner
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

# 开始训练
runner.train()
```

使用两张卡启动分布式训练：

```bash
torchrun --nproc-per-node 2 examples/distributed_training_with_flexible_runner.py --use-deepspeed
```

<details>
<summary>训练日志</summary>

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

PyTorch 从 v1.11 版本开始支持 [FullyShardedDataParallel](https://pytorch.org/docs/stable/fsdp.html) 训练，但由于其接口一直处于变动中，我们只支持 PyTorch v2.0.0 及以上的版本。

使用 FSDP 需配置 FlexibleRunner 的 strategy 参数：指定 `type='FSDPStrategy'` 并配置参数。参数的详细介绍可阅读 [FSDPStrategy](mmengine._strategy.FSDPStrategy)。

下面是 FSDP 相关的配置：

```python
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
size_based_auto_wrap_policy = partial(
    size_based_auto_wrap_policy, min_num_params=1e7)

# 指定 FSDPStrategy 并配置参数
strategy = dict(
    type='FSDPStrategy',
    model_wrapper=dict(auto_wrap_policy=size_based_auto_wrap_policy))

# 指定 AmpOptimWrapper 并配置参数
optim_wrapper = dict(
    type='AmpOptimWrapper', optimizer=dict(type='AdamW', lr=1e-3))

# 初始化 FlexibleRunner
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

# 开始训练
runner.train()
```

使用两张卡启动分布式训练：

```bash
torchrun --nproc-per-node 2 examples/distributed_training_with_flexible_runner.py --use-fsdp
```

<details>
<summary>训练日志</summary>

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

[ColossalAI](https://colossalai.org/) 是一个具有高效并行化技术的综合大规模模型训练系统。MMEngine 自 v0.9.0 开始，支持使用 ColossalAI 中的 ZeRO 系列优化策略训练模型。

安装版本大于 v0.3.1 的 ColossalAI。这个版本限制是由于 v0.3.1 存在一些程序阻塞的 [Bug](https://github.com/hpcaitech/ColossalAI/issues/4393)，而该 Bug 在之后的版本中已经修复。如果目前 ColossalAI 的最高版本仍为 v0.3.1，建议从源码安装主分支的 ColossalAI。

```{note}
需要注意的是，如果你的 PyTorch 版本高于 2.0，并遇到了 `nvcc fatal : Unsupported gpu architecture 'compute_90'` 类似的编译错误，则需要 git clone 源码，参考该 [PR](https://github.com/hpcaitech/ColossalAI/pull/4357) 进行修改源码，再进行安装
```

```bash
pip install git+https://github.com/hpcaitech/ColossalAI
```

如果 ColossalAI 的最新版本大于 v0.3.1，可以直接使用 pip 安装：

```bash
pip install colossalai
```

安装好 ColossalAI 后，需配置 FlexibleRunner 的 strategy 和 optim_wrapper 参数：

- strategy：指定 `type='ColossalAIStrategy'` 并配置参数。参数的详细介绍可阅读 [ColossalAIStrategy](mmengine._strategy.ColossalAIStrategy)。
- optim_wrapper：缺省 `type` 参数，或指定 `type=ColossalAIOptimWrapper`，优化器类型建议选择 `HybridAdam`。其他可配置类型可阅读 [ColossalAIOptimWrapper](mmengine._strategy.ColossalAIOptimWrapper)。

下面是 ColossalAI 相关的配置：

```python
from mmengine.runner._flexible_runner import FlexibleRunner

strategy = dict(type='ColossalAIStrategy')
optim_wrapper = dict(optimizer=dict(type='HybridAdam', lr=1e-3))

# 初始化 FlexibleRunner
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

# 开始训练
runner.train()
```

使用两张卡启动分布式训练：

```bash
torchrun --nproc-per-node 2 examples/distributed_training_with_flexible_runner.py --use-colossalai
```

<details>
<summary>训练日志</summary>

```
08/18 11:56:34 - mmengine - INFO - Epoch(train)  [1][ 10/196]  lr: 3.3333e-04  eta: 0:10:31  time: 0.3238  data_time: 0.0344  memory: 597  loss: 3.8766
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
08/18 11:56:46 - mmengine - INFO - Epoch(train)  [1][120/196]  lr: 3.3333e-04  eta: 0:03:56  time: 0.1090  data_time: 0.0338  memory: 597  loss: 2.0273
08/18 11:56:48 - mmengine - INFO - Epoch(train)  [1][130/196]  lr: 3.3333e-04  eta: 0:03:52  time: 0.1096  data_time: 0.0339  memory: 597  loss: 2.0086
08/18 11:56:49 - mmengine - INFO - Epoch(train)  [1][140/196]  lr: 3.3333e-04  eta: 0:03:49  time: 0.1096  data_time: 0.0339  memory: 597  loss: 1.9180
08/18 11:56:50 - mmengine - INFO - Epoch(train)  [1][150/196]  lr: 3.3333e-04  eta: 0:03:46  time: 0.1092  data_time: 0.0339  memory: 597  loss: 1.9578
08/18 11:56:51 - mmengine - INFO - Epoch(train)  [1][160/196]  lr: 3.3333e-04  eta: 0:03:43  time: 0.1097  data_time: 0.0339  memory: 597  loss: 1.9375
08/18 11:56:52 - mmengine - INFO - Epoch(train)  [1][170/196]  lr: 3.3333e-04  eta: 0:03:40  time: 0.1092  data_time: 0.0339  memory: 597  loss: 1.9312
08/18 11:56:53 - mmengine - INFO - Epoch(train)  [1][180/196]  lr: 3.3333e-04  eta: 0:03:37  time: 0.1070  data_time: 0.0339  memory: 597  loss: 1.9078
```

</details>
