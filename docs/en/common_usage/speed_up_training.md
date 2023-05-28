# Speed up Training

## Distributed Training

```{warning}
The usage of distributed had been moved to [Distributed Training](./distributed_training.md).
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

## Using faster Optimizers

If Ascend devices are used, you can use the Ascend optimizers to shorten the training time of the model. The optimizers supported by Ascend devices are as follows:

- NpuFusedAdadelta
- NpuFusedAdam
- NpuFusedAdamP
- NpuFusedAdamW
- NpuFusedBertAdam
- NpuFusedLamb
- NpuFusedRMSprop
- NpuFusedRMSpropTF
- NpuFusedSGD

The usage is the same as native optimizers, and you can refer to [Using Optimizers](../tutorials/optim_wrapper.md#configure-the-optimwapper-in-runner) for more information.
