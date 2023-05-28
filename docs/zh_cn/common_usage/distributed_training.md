# 分布式训练

MMEngine 支持 CPU、单卡、单机多卡以及多机多卡的训练。当环境中有多张显卡时，我们可以使用以下命令开启单机多卡或者多机多卡的方式从而缩短模型的训练时间。

## 单机多卡

假设当前机器有 8 张显卡，可以使用以下命令开启多卡训练

```bash
python -m torch.distributed.launch --nproc_per_node=8 examples/distributed_training.py --launcher pytorch
```

如果需要指定显卡的编号，可以设置 `CUDA_VISIBLE_DEVICES` 环境变量，例如使用第 0 和第 3 张卡

```bash
CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 examples/distributed_training.py --launcher pytorch
```

## 多机多卡

假设有 2 台机器，每台机器有 8 张卡。

第一台机器运行以下命令

```bash
python -m torch.distributed.launch \
    --nnodes 8 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 29500 \
    --nproc_per_node=8 \
    examples/distributed_training.py --launcher pytorch
```

第 2 台机器运行以下命令

```bash
python -m torch.distributed.launch \
    --nnodes 8 \
    --node_rank 1 \
    --master_addr 127.0.0.1 \
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
