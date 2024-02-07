# Migrate Model from MMCV to MMEngine

## Introduction

The early computer vision tasks supported by MMCV, such as detection and classification, used a general process to optimize model. It can be summarized as the following four steps:

1. Calculate the loss
2. Calculate the gradients
3. Update the model parameters
4. Clean the gradients of the last iteration

For most of the high-level tasks, "where" and "when" to perform the above processes is commonly fixed, therefore it seems reasonable to use [Hook](../design/hook.md) to implement it. MMCV implements series of hooks, such as `OptimizerHook`, `Fp16OptimizerHook` and `GradientCumulativeFp16OptimizerHook` to provide varies of optimization strategies.

On the other hand, tasks like GAN (Generative adversarial network) and Self-supervision require more flexible training processes, which do not meet the characteristics mentioned above, and it could be hard to use hooks to implement them. To meet the needs of these tasks, MMCV will pass `optimizer` to `train_step` and users can customize the optimization process as they want. Although it works, it cannot utilize various `OptimizerHook` implemented in MMCV, and downstream repositories have to implement mix-precision training, and gradient accumulation on their own.

To unify the training process of various deep learning tasks, MMEngine designed the [OptimWrapper](mmengine.optim.OptimWrapper), which integrates the mixed-precision training, gradient accumulation and other optimization strategies into a unified interface.

## Migrate optimization process

Since MMEngine designs the `OptimWrapper` and deprecates series of `OptimizerHook`, there would be some differences between the optimization process in MMCV and MMEngine.

### Commonly used optimization process

Considering tasks like detection and classification, the optimization process is usually the same, so `BaseModel` integrates the process into `train_step`.

**Model based on MMCV**

Before describing how to migrate the model, let's look at a minimal example to train a model based on the MMCV.

```python
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from mmcv.runner import Runner
from mmcv.utils.logging import get_logger


train_dataset = [(torch.ones(1, 1), torch.ones(1, 1))] * 50
train_dataloader = DataLoader(train_dataset, batch_size=2)


class MMCVToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, img, label, return_loss=False):
        feat = self.linear(img)
        loss1 = (feat - label).pow(2)
        loss2 = (feat - label).abs()
        loss = (loss1 + loss2).sum()
        return dict(loss=loss,
                    num_samples=len(img),
                    log_vars=dict(
                        loss1=loss1.sum().item(),
                        loss2=loss2.sum().item()))

    def train_step(self, data, optimizer=None):
        return self(*data, return_loss=True)

    def val_step(self, data, optimizer=None):
        return self(*data, return_loss=False)


model = MMCVToyModel()
optimizer = SGD(model.parameters(), lr=0.01)
logger = get_logger('demo')

lr_config = dict(policy='step', step=[2, 3])
optimizer_config = dict(grad_clip=None)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])


runner = Runner(
    model=model,
    work_dir='tmp_dir',
    optimizer=optimizer,
    logger=logger,
    max_epochs=5)

runner.register_training_hooks(
    lr_config=lr_config,
    optimizer_config=optimizer_config,
    log_config=log_config)
runner.run([train_dataloader], [('train', 1)])
```

Model based on MMCV must implement `train_step`, and return a `dict` which contains the following keys:

- `loss`: Passed to `OptimizerHook` to calculate gradient.
- num_samples: Passed to `LogBuffer` to count the averaged loss
- log_vars: Passed to `LogBuffer` to count the averaged loss

**Model based on MMEngine**

The same model based on MMEngine

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mmengine.runner import Runner
from mmengine.model import BaseModel

train_dataset = [(torch.ones(1, 1), torch.ones(1, 1))] * 50
train_dataloader = DataLoader(train_dataset, batch_size=2)


class MMEngineToyModel(BaseModel):

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, img, label, mode):
        feat = self.linear(img)
        # Called by train_step and return the loss dict
        if mode == 'loss':
            loss1 = (feat - label).pow(2)
            loss2 = (feat - label).abs()
            return dict(loss1=loss1, loss2=loss2)
        # Called by val_step and return the predictions
        elif mode == 'predict':
            return [_feat for _feat in feat]
        # tensor model, find more details in tutorials/model.md
        else:
            pass


runner = Runner(
    model=MMEngineToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=5),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)))
runner.train()
```

In MMEngine, users can customize their model based on `BaseModel`, which implements the same logic as `OptimizerHook` in `train_step`. For high-level tasks, `train_step` will be called in [EpochBasedTrainLoop](mmengine.runner.EpochBasedTrainLoop) or [IterBasedTrainLoop](mmengine.runner.IterBasedTrainLoop) with specific arguments, and users do not need to care about the optimization process. For low-level tasks, users can override the `train_step` to customize the optimization process.

<table class="docutils">
<thead>
  <tr>
    <th>Model in MMCV</th>
    <th>Model in MMEngine</th>
<tbody>
  <tr>
  <td valign="top" class='two-column-table-wrapper'><div style="overflow-x: auto">

```python
class MMCVToyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, img, label, return_loss=False):
        feat = self.linear(img)
        loss1 = (feat - label).pow(2)
        loss2 = (feat - label).abs()
        loss = (loss1 + loss2).sum()
        return dict(loss=loss,
                    num_samples=len(img),
                    log_vars=dict(
                        loss1=loss1.sum().item(),
                        loss2=loss2.sum().item()))

    def train_step(self, data, optimizer=None):
        return self(*data, return_loss=True)

    def val_step(self, data, optimizer=None):
        return self(*data, return_loss=False)
```

</div>
  </td>
  <td valign="top" class='two-column-table-wrapper'><div style="overflow-x: auto">

```python
class MMEngineToyModel(BaseModel):

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, img, label, mode):
        if mode == 'loss':
            feat = self.linear(img)
            loss1 = (feat - label).pow(2)
            loss2 = (feat - label).abs()
            return dict(loss1=loss1, loss2=loss2)
        elif mode == 'predict':
            return [_feat for _feat in feat]
        else:
            pass

    # The equivalent code snippet of `train_step`
    # def train_step(self, data, optim_wrapper):
    #     data = self.data_preprocessor(data)
    #     loss_dict = self(*data, mode='loss')
    #     loss_dict['loss1'] = loss_dict['loss1'].sum()
    #     loss_dict['loss2'] = loss_dict['loss2'].sum()
    #     loss = (loss_dict['loss1'] + loss_dict['loss2']).sum()
    #     Call the optimizer wrapper to update parameters.
    #     optim_wrapper.update_params(loss)
    #     return loss_dict
```

</td>
</div>
</tr>
</thead>
</table>

```{note}
See more information about `data_preprocessor` and `optim_wrapper` in docs [optim_wrapper](../tutorials/optim_wrapper.md) and [data_preprocessor](../tutorials/model.md).
```

The main differences of model in MMCV and MMEngine can be summarized as follows:

- `MMCVToyModel` inherits from `nn.Module`, and `MMEngineToyModel` inherits from `BaseModel`

- `MMCVToyModel` must implement `train_step` method and return a `dict` with keys `loss`, `log_vars`, and `num_samples`. `MMEngineToyModel` only needs to implement `forward` method for high level tasks, and return a `dict` with differentiable losses.

- `MMCVToyModel.forward` and `MMEngineToyModel.forward` must match with `train_step` which will call it. Since `MMEngineToyModel` does not override the `train_step`, `BaseModel.train_step` will be directly called, which requires that forward must accept `mode` parameter. Find more details in [tutorials of model](../tutorials/model.md)

### Custom optimization process

Takes training a GAN model as an example, generator and discriminator need to be optimized in turn and the optimization strategy could change as the training iteration grows. Therefore it could be hard to use `OptimizerHook` to meet such requirements in MMCV. GAN model based on MMCV will accept an optimizer in `train_step` and update parameters in it. Actually, MMEngine borrows this way and simplifies it by passing an [optim_wrapper](../tutorials/optim_wrapper.md) rather than an optimizer.

Referred to [training a GAN model](../examples/train_a_gan.md), The differences of MMCV and MMEngine are as follows:

<table class="docutils">
<thead>
  <tr>
    <th>Training gan in MMCV</th>
    <th>Training gan in MMEngine</th>
<tbody>
  <tr>
  <td valign="top" class='two-column-table-wrapper'><div style="overflow-x: auto">

```python
    def train_discriminator(self, inputs, optimizer):
        real_imgs = inputs['inputs']
        z = torch.randn(
            (real_imgs.shape[0], self.noise_size)).type_as(real_imgs)
        with torch.no_grad():
            fake_imgs = self.generator(z)

        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)

        parsed_losses, log_vars = self.disc_loss(disc_pred_fake,
                                                 disc_pred_real)
        parsed_losses.backward()
        optimizer.step()
        optimizer.zero_grad()
        return log_vars

    def train_generator(self, inputs, optimizer):
        real_imgs = inputs['inputs']
        z = torch.randn(inputs['inputs'].shape[0], self.noise_size).type_as(
            real_imgs)

        fake_imgs = self.generator(z)

        disc_pred_fake = self.discriminator(fake_imgs)
        parsed_loss, log_vars = self.gen_loss(disc_pred_fake)

        parsed_losses.backward()
        optimizer.step()
        optimizer.zero_grad()
        return log_vars
```

</td>
  </div>
  <td valign="top" class='two-column-table-wrapper'><div style="overflow-x: auto">

```python
    def train_discriminator(self, inputs, optimizer_wrapper):
        real_imgs = inputs['inputs']
        z = torch.randn(
            (real_imgs.shape[0], self.noise_size)).type_as(real_imgs)
        with torch.no_grad():
            fake_imgs = self.generator(z)

        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)

        parsed_losses, log_vars = self.disc_loss(disc_pred_fake,
                                                 disc_pred_real)
        optimizer_wrapper.update_params(parsed_losses)
        return log_vars



    def train_generator(self, inputs, optimizer_wrapper):
        real_imgs = inputs['inputs']
        z = torch.randn(real_imgs.shape[0], self.noise_size).type_as(real_imgs)

        fake_imgs = self.generator(z)

        disc_pred_fake = self.discriminator(fake_imgs)
        parsed_loss, log_vars = self.gen_loss(disc_pred_fake)

        optimizer_wrapper.update_params(parsed_loss)
        return log_vars
```

</td>
  </div>
</tr>
</thead>
</table>

Apart from the differences mentioned in the previous section, the main difference in the optimization process in MMCV and MMEngine is that the latter can use `optim_wrapper` in a more simple way. The convenience of `optim_wrapper` would be more obvious if gradient accumulation and mix-precision training are applied.

## Migrate validation/testing process

Model based on MMCV usually does not need to provide `test_step` or `val_step` for testing/validation. However, MMEngine performs the testing/validation by [ValLoop](mmengine.runner.ValLoop) and [TestLoop](mmengine.runner.TestLoop), which will call `runner.model.val_step` and `runner.model.test_step`. Therefore model based on MMEngine needs to implement `val_step` and `test_step`, of which input data and output predictions should be compatible with DataLoader and [Evaluator.process](mmengine.evaluator.Evaluator.process) respectively. You can find more details in the [model tutorial](../tutorials/model.md). Therefore, `MMEngineToyModel.forward` will slice the feat and return the predictions as a list.

```python

class MMEngineToyModel(BaseModel):

    ...
    def forward(self, img, label, mode):
        if mode == 'loss':
            ...
        elif mode == 'predict':
            # Slice the data to a list
            return [_feat for _feat in feat]
        else:
            ...
```

## Migrate the distributed training

MMCV will wrap the model with distributed wrapper before building the runner, while MMEngine will wrap the model in Runner. Therefore, we need to configure the `launcher` and `model_wrapper_cfg` for Runner. [Migrate Runner from MMCV to MMEngine](./runner.md) will introduce it in detail.

1. **Commonly used training process**

   For the high-level tasks mentioned in [introduction](#introduction), the default [distributed model wrapper](mmengine.model.MMDistributedDataParallel) is enough. Therefore, we only need to configure the `launcher` for MMEngine Runner.

   <table class="docutils">
    <thead>
    <tr>
        <th>Distributed training in MMCV </th>
        <th>Distributed training in MMEngine</th>
    <tbody>
    <tr>
    <td valign="top" class='two-column-table-wrapper'><div style="overflow-x: auto">

   ```python
   model = MMDistributedDataParallel(
       model,
       device_ids=[int(os.environ['LOCAL_RANK'])],
       broadcast_buffers=False,
       find_unused_parameters=find_unused_parameters)
   ...
   runner = Runner(model=model, ...)
   ```

   </div>
    </td>
    <td valign="top" class='two-column-table-wrapper'><div style="overflow-x: auto">

   ```python
   runner = Runner(
       model=model,
       launcher='pytorch', # enable distributed training
       ...,
   )
   ```

   </div>
    </td>
    </tr>
    </thead>
    </table>

&#160;

2. **optimize modules independently with custom optimization process**

   Again, taking the example of training a GAN model, the generator and discriminator need to be optimized separately. Therefore, the model needs to be wrapped by `MMSeparateDistributedDataParallel`, which need to be specified when building the runner.

   ```python
   cfg = dict(model_wrapper_cfg='MMSeparateDistributedDataParallel')
   runner = Runner(
       model=model,
       ...,
       launcher='pytorch',
       cfg=cfg)
   ```

&#160;

3. **Optimize a model with a custom optimization process**

Sometimes we need to optimize the whole model with a custom optimization process, where we cannot reuse `BaseModel.train_step`, but need to override it, e.g. we want to optimize the model twice with the same batch of images: the first time with batch data augmentation on, and the second time with it off

```python
class CustomModel(BaseModel):

    def train_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data, training=True)  # Enable batch augmentation
        loss = self(data, mode='loss')
        optim_wrapper.update_params(loss)
        data = self.data_preprocessor(data, training=False)  # Disable batch augmentation
        loss = self(data, mode='loss')
        optim_wrapper.update_params(loss)
```

In this case, we need to customize a model wrapper that overrides the `train_step` and performs the same process as `CustomModel.train_step`.

```python
   class CustomDistributedDataParallel(MMSeparateDistributedDataParallel):

       def train_step(self, data, optim_wrapper):
           data = self.data_preprocessor(data, training=True)  # Enable batch augmentation
           loss = self(data, mode='loss')
           optim_wrapper.update_params(loss)
           data = self.data_preprocessor(data, training=False)  # Disable batch augmentation
           loss = self(data, mode='loss')
           optim_wrapper.update_params(loss)
```

Then we can specify it when building Runner:

```python
cfg = dict(model_wrapper_cfg=dict(type='CustomDistributedDataParallel'))
runner = Runner(
    model=model,
    ...,
    launcher='pytorch',
    cfg=cfg
)
```
