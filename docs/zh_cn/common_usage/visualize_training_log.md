# 可视化训练日志

MMEngine 集成了 [TensorBoard](https://www.tensorflow.org/tensorboard?hl=zh-cn)、[Weights & Biases (WandB)](https://docs.wandb.ai/)、[MLflow](https://mlflow.org/docs/latest/index.html) 、[ClearML](https://clear.ml/docs/latest/docs)、[Neptune](https://docs.neptune.ai/)、[DVCLive](https://dvc.org/doc/dvclive) 和 [Aim](https://aimstack.readthedocs.io/en/latest/overview.html) 实验管理工具，你可以很方便地跟踪和可视化损失及准确率等指标。

下面基于 [15 分钟上手 MMENGINE](../get_started/15_minutes.md) 中的例子介绍如何一行配置实验管理工具。

## TensorBoard

设置 `Runner` 初始化参数中的 `visualizer`，并将 `vis_backends` 设置为 [TensorboardVisBackend](mmengine.visualization.TensorboardVisBackend)。

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

使用 WandB 前需安装依赖库 `wandb` 并登录至 wandb。

```bash
pip install wandb
wandb login
```

设置 `Runner` 初始化参数中的 `visualizer`，并将 `vis_backends` 设置为 [WandbVisBackend](mmengine.visualization.WandbVisBackend)。

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

## MLflow (WIP)

## ClearML

使用 ClearML 前需安装依赖库 `clearml` 并参考 [Connect ClearML SDK to the Server](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps#connect-clearml-sdk-to-the-server) 进行配置。

```bash
pip install clearml
clearml-init
```

设置 `Runner` 初始化参数中的 `visualizer`，并将 `vis_backends` 设置为 [ClearMLVisBackend](mmengine.visualization.ClearMLVisBackend)。

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
    visualizer=dict(type='Visualizer', vis_backends=[dict(type='ClearMLVisBackend')]),
)
runner.train()
```

![image](https://github.com/open-mmlab/mmengine/assets/58739961/d68e1dd2-9e82-40fb-ad81-00a647549adc)

## Neptune

使用 Neptune 前需先安装依赖库 `neptune` 并登录 [Neptune.AI](https://docs.neptune.ai/) 进行配置。

```bash
pip install neptune
```

设置 `Runner` 初始化参数中的 `visualizer`，并将 `vis_backends` 设置为 [NeptuneVisBackend](mmengine.visualization.NeptuneVisBackend)。

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
    visualizer=dict(type='Visualizer', vis_backends=[dict(type='NeptuneVisBackend')]),
)
runner.train()
```

![image](https://github.com/open-mmlab/mmengine/assets/58739961/9122e2ac-cc4f-43b2-bad3-ae33faa64043)

请注意：若未提供 `project` 和 `api_token` ，neptune 将被设置成离线模式，产生的文件将保存到本地 `.neptune` 文件下。
推荐在初始化时提供 `project` 和 `api_token` ，具体方法如下所示：

```python
runner = Runner(
    ...
    visualizer=dict(
        type='Visualizer',
        vis_backends=[
            dict(
                type='NeptuneVisBackend',
                init_kwargs=dict(project='workspace-name/project-name',
                                 api_token='your api token')
            ),
        ],
    ),
    ...
)
runner.train()
```

更多初始化配置参数可点击 [neptune.init_run API](https://docs.neptune.ai/api/neptune/#init_run) 查询。

## DVCLive

使用 DVCLive 前需先安装依赖库 `dvclive` 并参考 [iterative.ai](https://dvc.org/doc/start) 进行配置。常见的配置方式如下：

```bash
pip install dvclive
cd ${WORK_DIR}
git init
dvc init
git commit -m "DVC init"
```

设置 `Runner` 初始化参数中的 `visualizer`，并将 `vis_backends` 设置为 [DVCLiveVisBackend](mmengine.visualization.DVCLiveVisBackend)。

```python
runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir_dvc',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
    visualizer=dict(type='Visualizer', vis_backends=[dict(type='DVCLiveVisBackend')]),
)
runner.train()
```

```{note}
推荐将 `work_dir` 设置为 `work_dirs`。否则，你在 OpenMMLab 仓库中运行试验时，DVC 会给出警告 `WARNING:dvclive:Error in cache: bad DVC file name 'work_dirs\xxx.dvc' is git-ignored`。
```

打开 `work_dir_dvc` 下面的 `report.html` 文件，即可看到如下图的可视化效果。

![image](https://github.com/open-mmlab/mmengine/assets/58739961/47d85520-9a4a-4143-a449-12ed7347cc63)

你还可以安装 VSCode 扩展 [DVC](https://marketplace.visualstudio.com/items?itemName=Iterative.dvc) 进行可视化。

更多初始化配置参数可点击 [DVCLive API Reference](https://dvc.org/doc/dvclive/live) 查询。

## Aim

使用 Aim 前需先安装依赖库 `aim`。

```bash
pip install aim
```

设置 `Runner` 初始化参数中的 `visualizer`，并将 `vis_backends` 设置为 [AimVisBackend](mmengine.visualization.AimVisBackend)。

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
    visualizer=dict(type='Visualizer', vis_backends=[dict(type='AimVisBackend')]),
)
runner.train()
```

在终端中输入

```bash
aim up
```

或者在 Jupyter Notebook 中输入

```bash
%load_ext aim
%aim up
```

即可启动 Aim UI，界面如下图所示。

![image](https://github.com/open-mmlab/mmengine/assets/58739961/2fc6cdd8-1de7-4125-a20a-c95c1a8bdb1b)

初始化配置参数可点击 [Aim SDK Reference](https://aimstack.readthedocs.io/en/latest/refs/sdk.html#module-aim.sdk.run) 查询。
