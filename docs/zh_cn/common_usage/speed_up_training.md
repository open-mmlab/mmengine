# 加速训练

## 分布式训练

MMEngine 支持 CPU、单卡、单机多卡以及多机多卡的训练。当环境中有多张显卡时，我们可以使用以下命令开启单机多卡或者多机多卡的方式从而缩短模型的训练时间。

- 单机多卡

  假设当前机器有 8 张显卡，可以使用以下命令开启多卡训练

  ```bash
  python -m torch.distributed.launch --nproc_per_node=8 examples/train.py --launcher pytorch
  ```

  如果需要指定显卡的编号，可以设置 `CUDA_VISIBLE_DEVICES` 环境变量，例如使用第 0 和第 3 张卡

  ```bash
  CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 examples/train.py --launcher pytorch
  ```

- 多机多卡

  假设有 2 台机器，每台机器有 8 张卡。

  第一台机器运行以下命令

  ```bash
  python -m torch.distributed.launch \
      --nnodes 8 \
      --node_rank 0 \
      --master_addr 127.0.0.1 \
      --master_port 29500 \
      --nproc_per_node=8 \
      examples/train.py --launcher pytorch
  ```

  第 2 台机器运行以下命令

  ```bash
  python -m torch.distributed.launch \
      --nnodes 8 \
      --node_rank 1 \
      --master_addr 127.0.0.1 \
      --master_port 29500 \
      --nproc_per_node=8 \
      examples/train.py --launcher pytorch
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
      python examples/train.py --launcher="slurm"
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
