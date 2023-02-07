# 实验管理

MMEngine 集成了 [TensorBoard](https://www.tensorflow.org/tensorboard?hl=zh-cn) 和 [WandB](https://docs.wandb.ai/v/zh-hans/) 实验管理工具，你可以很方便地

- 跟踪和可视化损失及准确率等指标
- 可视化模型结构图
- 可视化特征图

下面基于[15 分钟上手 MMENGINE](../get_started/15_minutes.md)中的例子介绍如何一行配置实验管理工具。

## TensorBoard

`Runner` 新增 `visualizer` 参数并设置 `vis_backends` 为 `TensorboardVisBackend`。

```python
runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
    visualizer=dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')]),
)
runner.train()
```

## WandB

使用 WandB 前需安装依赖库 wandb 并登录至 wandb。

- 安装

  ```bash
  pip install wandb
  ```

- 登录

  ```bash
  wandb login
  ```

接下来只需修改 `Runner` 输入参数 `visualizer` 中的 `vis_backends` 为 `WandbVisBackend` 即可。

```python
runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
    visualizer=dict(type='Visualizer', vis_backends=[dict(type='WandbVisBackend')]),
)
runner.train()
```

![image](https://user-images.githubusercontent.com/58739961/217226120-0c45267c-c45f-4fce-bdd5-a99c8c393006.png)

可以点击 [WandbVisBackend API](mmengine.visualization.WandbVisBackend) 查看 `WandbVisBackend` 可配置的参数。例如 `init_kwargs`，该参数会传给 [wandb.init](https://docs.wandb.ai/ref/python/init) 方法。

```python
runner = Runner(
    ...
    visualizer=dict(
        type='Visualizer',
        vis_backends=[
            dict(
                type='WandbVisBackend',
                init_kwargs=dict(project='toy-example')
            ),
        ],
    ),
    ...
)
runner.train()
```
