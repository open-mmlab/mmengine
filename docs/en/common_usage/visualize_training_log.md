# Visualize Training Logs

MMEngine integrates experiment management tools such as [TensorBoard](https://www.tensorflow.org/tensorboard), [Weights & Biases (WandB)](https://docs.wandb.ai/), [MLflow](https://mlflow.org/docs/latest/index.html) and [ClearML](https://clear.ml/docs/latest/docs), making it easy to track and visualize metrics like loss and accuracy.

Below, we'll show you how to configure an experiment management tool in just one line, based on the example from [15 minutes to get started with MMEngine](../get_started/15_minutes.md).

## TensorBoard

Configure the `visualizer` in the initialization parameters of the Runner, and set `vis_backends` to `TensorboardVisBackend`.

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

Configure the `visualizer` in the initialization parameters of the Runner, and set `vis_backends` to `WandbVisBackend`.

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

Configure the `visualizer` in the initialization parameters of the Runner, and set `vis_backends` to `ClearMLVisBackend`.

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
