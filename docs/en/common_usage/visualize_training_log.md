# Visualize Training Logs

MMEngine integrates experiment management tools such as [TensorBoard](https://www.tensorflow.org/tensorboard), [Weights & Biases (WandB)](https://docs.wandb.ai/), [MLflow](https://mlflow.org/docs/latest/index.html), [ClearML](https://clear.ml/docs/latest/docs), [Neptune](https://docs.neptune.ai/), [DVCLive](https://dvc.org/doc/dvclive) and [Aim](https://aimstack.readthedocs.io/en/latest/overview.html), making it easy to track and visualize metrics like loss and accuracy.

Below, we'll show you how to configure an experiment management tool in just one line, based on the example from [15 minutes to get started with MMEngine](../get_started/15_minutes.md).

## TensorBoard

Configure the `visualizer` in the initialization parameters of the Runner, and set `vis_backends` to [TensorboardVisBackend](mmengine.visualization.TensorboardVisBackend).

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

Before using WandB, you need to install the `wandb` dependency library and log in to WandB.

```bash
pip install wandb
wandb login
```

Configure the `visualizer` in the initialization parameters of the Runner, and set `vis_backends` to [WandbVisBackend](mmengine.visualization.WandbVisBackend).

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

You can click on [WandbVisBackend API](mmengine.visualization.WandbVisBackend) to view the configurable parameters for `WandbVisBackend`. For example, the `init_kwargs` parameter will be passed to the [wandb.init](https://docs.wandb.ai/ref/python/init) method.

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

Before using ClearML, you need to install the `clearml` dependency library and refer to [Connect ClearML SDK to the Server](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps#connect-clearml-sdk-to-the-server) for configuration.

```bash
pip install clearml
clearml-init
```

Configure the `visualizer` in the initialization parameters of the Runner, and set `vis_backends` to [ClearMLVisBackend](mmengine.visualization.ClearMLVisBackend).

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

Before using Neptune, you need to install `neptune` dependency library and refer to [Neptune.AI](https://docs.neptune.ai/) for configuration.

```bash
pip install neptune
```

Configure the `Runner` in the initialization parameters of the Runner, and set `vis_backends` to [NeptuneVisBackend](mmengine.visualization.NeptuneVisBackend).

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

Please note: If the `project` and `api_token` are not specified, neptune will be set to offline mode and the generated files will be saved to the local `.neptune` file.
It is recommended to specify the `project` and `api_token` during initialization as shown below.

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

More initialization configuration parameters are available at [neptune.init_run API](https://docs.neptune.ai/api/neptune/#init_run).

## DVCLive

Before using DVCLive, you need to install `dvclive` dependency library and refer to [iterative.ai](https://dvc.org/doc/start) for configuration. Common configurations are as follows:

```bash
pip install dvclive
cd ${WORK_DIR}
git init
dvc init
git commit -m "DVC init"
```

Configure the `Runner` in the initialization parameters of the Runner, and set `vis_backends` to [DVCLiveVisBackend](mmengine.visualization.DVCLiveVisBackend).

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
Recommend not to set `work_dir` as `work_dirs`. Or DVC will give a warning `WARNING:dvclive:Error in cache: bad DVC file name 'work_dirs\xxx.dvc' is git-ignored` if you run experiments in a OpenMMLab's repo.
```

Open the `report.html` file under `work_dir_dvc`, and you will see the visualization as shown in the following image.

![image](https://github.com/open-mmlab/mmengine/assets/58739961/47d85520-9a4a-4143-a449-12ed7347cc63)

You can also configure a VSCode extension of [DVC](https://marketplace.visualstudio.com/items?itemName=Iterative.dvc) to visualize the training process.

More initialization configuration parameters are available at [DVCLive API Reference](https://dvc.org/doc/dvclive/live).

## Aim

Before using Aim, you need to install `aim` dependency library.

```bash
pip install aim
```

Configure the `Runner` in the initialization parameters of the Runner, and set `vis_backends` to [AimVisBackend](mmengine.visualization.AimVisBackend).

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

In the terminal, use the following command,

```bash
aim up
```

or in the Jupyter Notebook, use the following command,

```bash
%load_ext aim
%aim up
```

to launch the Aim UI as shown below.

![image](https://github.com/open-mmlab/mmengine/assets/58739961/2fc6cdd8-1de7-4125-a20a-c95c1a8bdb1b)

Initialization configuration parameters are available at [Aim SDK Reference](https://aimstack.readthedocs.io/en/latest/refs/sdk.html#module-aim.sdk.run).
