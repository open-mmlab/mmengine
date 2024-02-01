# Distributed Training

MMEngine supports training models with CPU, single GPU, multiple GPUs in single machine and multiple machines. When multiple GPUs are available in the environment, we can use the following command to enable multiple GPUs in single machine or multiple machines to shorten the training time of the model.

## Launch Training

### multiple GPUs in single machine

Assuming the current machine has 8 GPUs, you can enable multiple GPUs training with the following command:

```bash
python -m torch.distributed.launch --nproc_per_node=8 examples/distributed_training.py --launcher pytorch
```

If you need to specify the GPU index, you can set the `CUDA_VISIBLE_DEVICES` environment variable, e.g. use the 0th and 3rd GPU.

```bash
CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 examples/distributed_training.py --launcher pytorch
```

### multiple machines

Assume that there are 2 machines connected with ethernet, you can simply run following commands.

On the first machine:

```bash
python -m torch.distributed.launch \
    --nnodes 2 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 29500 \
    --nproc_per_node=8 \
    examples/distributed_training.py --launcher pytorch
```

On the second machine:

```bash
python -m torch.distributed.launch \
    --nnodes 2 \
    --node_rank 1 \
    --master_addr "ip_of_the_first_machine" \
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

## Customize Distributed Training

When users switch from `single GPU` training to `multiple GPUs` training, no changes need to be made. [Runner](mmengine.runner.Runner.wrap_model) will use [MMDistributedDataParallel](mmengine.model.MMDistributedDataParallel) by default to wrap the model, thereby supporting multiple GPUs training.

If you want to pass more parameters to MMDistributedDataParallel or use your own `CustomDistributedDataParallel`, you can set `model_wrapper_cfg`.

### Pass More Parameters to MMDistributedDataParallel

For example, setting `find_unused_parameters` to `True`:

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

### Use a Customized CustomDistributedDataParallel

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
