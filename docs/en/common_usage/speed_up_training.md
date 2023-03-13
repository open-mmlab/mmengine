# Speed up Training

## Distributed Training

MMEngine supports training models with CPU, single GPU, multiple GPUs in single machine and multiple machines. When multiple GPUs are available in the environment, we can use the following command to enable multiple GPUs in single machine or multiple machines to shorten the training time of the model.

- multiple GPUs in single machine

  Assuming the current machine has 8 GPUs, you can enable multiple GPUs training with the following command:

  ```bash
  python -m torch.distributed.launch --nproc_per_node=8 examples/train.py --launcher pytorch
  ```

  If you need to specify the GPU index, you can set the `CUDA_VISIBLE_DEVICES` environment variable, e.g. use the 0th and 3rd GPU.

  ```bash
  CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 examples/train.py --launcher pytorch
  ```

- multiple machines

  Assume that there are 2 machines connected with ethernet, you can simply run following commands.

  On the first machine:

  ```bash
  python -m torch.distributed.launch \
      --nnodes 8 \
      --node_rank 0 \
      --master_addr 127.0.0.1 \
      --master_port 29500 \
      --nproc_per_node=8 \
      examples/train.py --launcher pytorch
  ```

  On the second machine:

  ```bash
  python -m torch.distributed.launch \
      --nnodes 8 \
      --node_rank 1 \
      --master_addr 127.0.0.1 \
      --master_port 29500 \
      --nproc_per_node=8 \
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
      python examples/train.py --launcher="slurm"
  ```

## Mixed Precision Training

Nvidia introduced the Tensor Core unit into the Volta and Turing architectures to support FP32 and FP16 mixed precision computing. They further support BF16 in Ampere architectures. With automatic mixed precision training enabled, some operators operate at FP16/BF16 and the rest operate at FP32, which reduces training time and storage requirements without changing the model or degrading its training precision, thus supporting training with larger batch sizes, larger models, and larger input sizes.

[PyTorch officially supports amp from 1.6](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/). If you are interested in the implementation of automatic mixing precision, you can refer to [Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html).

MMEngine provides the wrapper [AmpOptimWrapper](mmengine.optim.AmpOptimWrapper) for auto-mixing precision training, just set `type='AmpOptimWrapper'` in ` optim_wrapper` to enable auto-mixing precision training, no other code changes are needed.

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    optim_wrapper=dict(
        type='AmpOptimWrapper',
        # If you want to use bfloat16, uncomment the following line
        # dtype='bfloat16',  # valid values: ('float16', 'bfloat16', None)
        optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
)
runner.train()
```

```{warning}
Up till PyTorch 1.13, `torch.bfloat16` performance on `Convolution` is bad unless manually set environment variable `TORCH_CUDNN_V8_API_ENABLED=1`. More context at [PyTorch issue](https://github.com/pytorch/pytorch/issues/57707#issuecomment-1166656767)
```

## Model Compilation

PyTorch introduced [torch.compile](https://pytorch.org/docs/2.0/dynamo/get-started.html) in its 2.0 release. It compiles your model to speedup trainning & validation. This feature can be enabled since MMEngine v0.7.0, by passing to `Runner` an extra `cfg` dict with `compile` keyword:

```python
runner = Runner(
    model=ResNet18(),
    ...  # other arguments you want
    cfg=dict(compile=True)
)
```

For advanced usage, you can also change compile options as illustrated in [torch.compile API Documentation](https://pytorch.org/docs/2.0/generated/torch.compile.html#torch-compile). For example:

```python
compile_options = dict(backend='inductor', mode='max-autotune')
runner = Runner(
    model=ResNet18(),
    ...  # other arguments you want
    cfg=dict(compile=compile_options)
)
```

This feature is only available for PyTorch >= 2.0.0.

```{warning}
`torch.compile` is still under development by PyTorch team. Some models may fail compilation. If you encounter errors during compilation, you can refer to [PyTorch Dynamo FAQ](https://pytorch.org/docs/2.0/dynamo/faq.html) for quick fix, or [TorchDynamo Troubleshooting](https://pytorch.org/docs/2.0/dynamo/troubleshooting.html) to post an issue in PyTorch.
```
