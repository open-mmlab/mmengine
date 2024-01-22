# 分布式训练

MMEngine 支持 CPU、单卡、单机多卡以及多机多卡的训练。当环境中有多张显卡时，我们可以使用以下命令开启单机多卡或者多机多卡的方式从而缩短模型的训练时间。

## 启动训练

### 单机多卡

假设当前机器有 8 张显卡，可以使用以下命令开启多卡训练

```bash
python -m torch.distributed.launch --nproc_per_node=8 examples/distributed_training.py --launcher pytorch
```

如果需要指定显卡的编号，可以设置 `CUDA_VISIBLE_DEVICES` 环境变量，例如使用第 0 和第 3 张卡

```bash
CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 examples/distributed_training.py --launcher pytorch
```

### 多机多卡

假设有 2 台机器，每台机器有 8 张卡。

第一台机器运行以下命令

```bash
python -m torch.distributed.launch \
    --nnodes 2 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 29500 \
    --nproc_per_node=8 \
    examples/distributed_training.py --launcher pytorch
```

第 2 台机器运行以下命令

```bash
python -m torch.distributed.launch \
    --nnodes 2 \
    --node_rank 1 \
    --master_addr "ip_of_the_first_machine" \
    --master_port 29500 \
    --nproc_per_node=8 \
    examples/distributed_training.py --launcher pytorch
```

如果在 slurm 集群运行 MMEngine，只需运行以下命令即可开启 2 机 16 卡的训练

```bash
srun -p mm_dev \
    --job-name=test \
    --gres=gpu:8 \
    --ntasks=16 \
    --ntasks-per-node=8 \
    --cpus-per-task=5 \
    --kill-on-bad-exit=1 \
    python examples/distributed_training.py --launcher="slurm"
```

## 定制化分布式训练

当用户从单卡训练切换到多卡训练时，无需做任何改动，[Runner](mmengine.runner.Runner.wrap_model) 会默认使用 [MMDistributedDataParallel](mmengine.model.MMDistributedDataParallel) 封装 model，从而支持多卡训练。
如果你希望给 `MMDistributedDataParallel` 传入更多的参数或者使用自定义的 `CustomDistributedDataParallel`，你可以设置 `model_wrapper_cfg`。

### 往 MMDistributedDataParallel 传入更多的参数

例如设置 `find_unused_parameters` 为 `True`：

```python
cfg = dict(
    model_wrapper_cfg=dict(
        type='MMDistributedDataParallel', find_unused_parameters=True)
)
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
    cfg=cfg,
)
runner.train()
```

### 使用自定义的 CustomDistributedDataParallel

```python
from mmengine.registry import MODEL_WRAPPERS

@MODEL_WRAPPERS.register_module()
class CustomDistributedDataParallel(DistributedDataParallel):
    pass


cfg = dict(model_wrapper_cfg=dict(type='CustomDistributedDataParallel'))
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
    cfg=cfg,
)
runner.train()
```
