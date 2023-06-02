# Distributed Training

MMEngine supports training models with CPU, single GPU, multiple GPUs in single machine and multiple machines. When multiple GPUs are available in the environment, we can use the following command to enable multiple GPUs in single machine or multiple machines to shorten the training time of the model.

## multiple GPUs in single machine

Assuming the current machine has 8 GPUs, you can enable multiple GPUs training with the following command:

```bash
python -m torch.distributed.launch --nproc_per_node=8 examples/distributed_training.py --launcher pytorch
```

If you need to specify the GPU index, you can set the `CUDA_VISIBLE_DEVICES` environment variable, e.g. use the 0th and 3rd GPU.

```bash
CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 examples/distributed_training.py --launcher pytorch
```

## multiple machines

Assume that there are 2 machines connected with ethernet, you can simply run following commands.

On the first machine:

```bash
python -m torch.distributed.launch \
    --nnodes 8 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 29500 \
    --nproc_per_node=8 \
    examples/distributed_training.py --launcher pytorch
```

On the second machine:

```bash
python -m torch.distributed.launch \
    --nnodes 8 \
    --node_rank 1 \
    --master_addr 127.0.0.1 \
    --master_port 29500 \
    --nproc_per_node=8 \
    examples/distributed_training.py --launcher pytorch
```

If you are running MMEngine in a slurm cluster, simply run the following command to enable training for 2 machines and 16 GPUs.

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
