# 加速训练

## 分布式训练

```{warning}
内容已被迁移至 [分布式训练](./distributed_training.md)。
```

## 混合精度训练

Nvidia 在 Volta 和 Turing 架构中引入 Tensor Core 单元，来支持 FP32 和 FP16 混合精度计算。在 Ampere 架构中，他们进一步支持了 BF16 计算。开启自动混合精度训练后，部分算子的操作精度是 FP16/BF16，其余算子的操作精度是 FP32。这样在不改变模型、不降低模型训练精度的前提下，可以缩短训练时间，降低存储需求，因而能支持更大的 batch size、更大模型和尺寸更大的输入的训练。

[PyTorch 从 1.6 开始官方支持 amp](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/)。如果你对自动混合精度的实现感兴趣，可以阅读 [torch.cuda.amp: 自动混合精度详解](https://zhuanlan.zhihu.com/p/348554267)。

MMEngine 提供自动混合精度的封装 [AmpOptimWrapper](mmengine.optim.AmpOptimWrapper) ，只需在 `optim_wrapper` 设置 `type='AmpOptimWrapper'` 即可开启自动混合精度训练，无需对代码做其他修改。

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    optim_wrapper=dict(
        type='AmpOptimWrapper',
        # 如果你想要使用 BF16，请取消下面一行的代码注释
        # dtype='bfloat16',  # 可用值： ('float16', 'bfloat16', None)
        optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
)
runner.train()
```

```{warning}
截止到 PyTorch 1.13 版本，在 `Convolution` 中直接使用 `torch.bfloat16` 性能低下，必须手动设置环境变量 `TORCH_CUDNN_V8_API_ENABLED=1` 以启用 CuDNN 版本的 BF16 Convolution。相关讨论见 [PyTorch Issue](https://github.com/pytorch/pytorch/issues/57707#issuecomment-1166656767)
```

## 模型编译

PyTorch 2.0 版本引入了 [torch.compile](https://pytorch.org/docs/2.0/dynamo/get-started.html) 新特性，通过对模型进行编译来加速训练、验证。MMEngine 从 v0.7.0 版本开始支持这一特性，你可以通过向 `Runner` 的 `cfg` 参数传入一个带有 `compile` 关键词的字典来开启模型编译：

```python
runner = Runner(
    model=ResNet18(),
    ...  # 你的其他 Runner 配置参数
    cfg=dict(compile=True)
)
```

此外，你也可以传入更多的编译配置选项，所有编译配置选项可以参考 [torch.compile API 文档](https://pytorch.org/docs/2.0/generated/torch.compile.html#torch-compile)

```python
compile_options = dict(backend='inductor', mode='max-autotune')
runner = Runner(
    model=ResNet18(),
    ...  # 你的其他 Runner 配置参数
    cfg=dict(compile=compile_options)
)
```

这一特性只有在你安装 PyTorch >= 2.0.0 版本时才可用。

```{warning}
`torch.compile` 目前仍然由 PyTorch 团队持续开发中，一些模型可能会编译失败。如果遇到了类似问题，你可以查阅 [PyTorch Dynamo FAQ](https://pytorch.org/docs/2.0/dynamo/faq.html) 解决常见问题，或参考 [TorchDynamo Troubleshooting](https://pytorch.org/docs/2.0/dynamo/troubleshooting.html) 向 PyTorch 提 issue.
```

## 使用更快的优化器

如果使用了昇腾的设备，可以使用昇腾的优化器从而缩短模型的训练时间。昇腾设备支持的优化器如下

- NpuFusedAdadelta
- NpuFusedAdam
- NpuFusedAdamP
- NpuFusedAdamW
- NpuFusedBertAdam
- NpuFusedLamb
- NpuFusedRMSprop
- NpuFusedRMSpropTF
- NpuFusedSGD

使用方式同原生优化器一样，可参考[优化器的使用](../tutorials/optim_wrapper.md#在执行器中配置优化器封装)。
