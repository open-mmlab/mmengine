# 迁移 MMCV 执行器到 MMEngine

## 简介

MMCV 早期主要适配上层（High level）深度学习任务，例如目标检测、物体识别。考虑到这类任务模型参数优化流程基本一致：

1. 计算损失
2. 计算梯度
3. 更新参数
4. 梯度清零

上述流程的一大特点就是调用位置统一（在训练迭代后调用）、执行步骤统一（依次执行 1->2->3->4），非常契合[钩子（Hook）](../design/hook.md)的设计原则。因此
MMCV 通过 `OptimizerHook` 来完成模型优化的流程。

然而，对于底层（Low level）深度学习任务，如生成对抗网络（GAN），自监督（Self-supervision），它们的模型参数优化流程完全不同，并不满足调用位置统一、执行步骤统一的原则。因此这类任务通常无法使用 `OptimizerHook` 进行参数优化，而选择在 `model.train_step` 方法里实现了完整的优化流程。这意味着这类深度学习任务无法使用 MMCV 封装的 `OptimizerHook`、`Fp16OptimizerHook`，`GradientCumulativeFp16OptimizerHook` 实现混合精度训练、梯度累加等策略，而需要在 `train_step` 里自自行实现相应逻辑。

为了统一底层深度学习任务和上层深度学习任务的训练流程，MMEngine 设计了[优化器封装](mmengine.optim.OptimWrapper)，并在 `model.train_step` 里执行优化流程。

## 迁移模型

### 上层深度学习任务

考虑到上层深度学习任务模型参数优化的流程基本一致，我们可以通过继承[模型基类](../tutorials/model.md)来完成迁移。

**基于 MMCV 执行器的模型**

在介绍如何迁移模型之前，我们先来看一个基于 MMCV 执行器训练模型的最简示例：

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

基于 MMCV 执行器训练模型时，我们必须实现 `train_step` 接口，并返回一个字典，字典需要包含以下三个字段：

- loss：传给 `OptimizerHook` 计算梯度
- num_samples：传给 `LogBuffer`，用于计算平滑后的损失
- log_vars：传给 `LogBuffer` 用于计算平滑后的损失

**基于 MMEngine 执行器的模型**

基于 MMEngine 的执行器，实现同样逻辑的代码：

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
        if mode == 'loss':
            feat = self.linear(img)
            loss1 = (feat - label).pow(2)
            loss2 = (feat - label).abs()
            return dict(loss1=loss1, loss2=loss2)
        elif mode == 'tensor':
            return [feat]
        else:
            # tensor 模式，功能详见模型教程文档： tutorials/model.md
            pass


runner = Runner(
    model=MMEngineToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=5),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)))
runner.train()
```

MMEngine 实现了模型基类，模型基类在 `train_step` 里实现了 `OptimizerHook` 的优化流程。因此上例中，我们无需实现 `train_step`，运行时直接调用基类的 `train_step`。

<table class="docutils">
<thead>
  <tr>
    <th>MMCV 模型</th>
    <th>MMEngine 模型</th>
<tbody>
  <tr>
  <td valign="top">

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

</td>
  <td valign="top">

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
        elif mode == 'tensor':
            return [feat]
        else:
            # tensor 模式，功能详见模型教程文档： tutorials/model.md
            pass

    # 模型基类 `train_step` 等效代码
    # def train_step(self, data, optim_wrapper):
    #     data = self.data_preprocessor(data)
    #     loss_dict = self(*data, mode='loss')
    #     loss_dict['loss1'] = loss_dict['loss1'].sum()
    #     loss_dict['loss2'] = loss_dict['loss2'].sum()
    #     loss = (loss_dict['loss1'] + loss_dict['loss2']).sum()
    # 调用优化器封装更新模型参数
    #     optim_wrapper.update_params(loss)
    #     return loss_dict

    # 模型基类 `val_step`、`test_step` 等效代码。二者可以根据需求有不同实现
    # def test_step(self, data, optim_wrapper):
    # 调用数据处理器处理 data
    #     data = self.data_preprocessor(data)
    #     return self(*data, mode='predict')
```

</td>
</tr>
</thead>
</table>

等效代码中的[数据处理器（data_preprocessor）](mmengine.model.BaseDataPreprocessor) 和[优化器封装（optim_wrapper）](mmengine.optim.OptimWrapper) 的说明，详见[模型教程](../tutorials/model.md#数据处理器（DataPreprocessor）)和[优化器封装教程](../tutorials/optim_wrapper.md)。

模型具体差异如下：

- `MMCVToyModel` 继承自 `nn.Module`，而 `MMEngineToyModel` 继承自 `BaseModel`
- `MMCVToyModel` 必须实现 `train_step`，且必须返回损失字典，损失字典包含 `loss` 和 `log_vars` 和 `num_samples` 字段。`MMEngineToyModel` 继承自 `BaseModel`，只需要实现 `forward` 接口，并返回损失字典，损失字典的每一个值必须是可微的张量
- `MMCVToyModel` 和 `MMEngineModel` 的 `forward` 的接口需要匹配 `train_step` 中的调用方式，由于 `MMEngineToyModel` 直接调用基类的 `train_step` 方法，因此 `forward` 需要接受参数 `mode`，具体规则详见[模型教程文档](../tutorials/model.md)
- `MMEngineModel` 如果没有继承 `BaseModel`，必须实现 `train_step`、`test_step` 和 `val_step` 方法。

### 底层深度学习任务

底层深度学习任务可能需要实现自定义的参数更新流程，以训练生成对抗网络为例，生成器和判别器的优化需要交替进行，且优化流程可能会随着迭代次数的增多发生变化，因此很难使用 `OptimizerHook` 来满足这种需求。在基于 MMCV 训练生成对抗网络时，通常会在模型的 `train_step` 接口中传入 optimizer，然后在 `train_step` 里实现自定义的参数更新逻辑。这种训练流程和 MMEngine 非常相似，只不过 MMEngine 在 `train_step` 接口中传入[优化器封装](../tutorials/optim_wrapper.md)，能够更加简单的优化模型。

参考示例[训练生成对抗网络](../examples/train_a_gan.md)，如果用 MMCV 进行训练，`GAN` 的模型优化接口如下：

```python
    def train_discriminator(
            self, inputs, data_sample,
            optimizer):
        real_imgs = inputs['inputs']
        z = torch.randn((real_imgs.shape[0], self.noise_size))
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

    def train_generator(self, inputs, data_sample, optimizer_wrapper):
        z = torch.randn(inputs['inputs'].shape[0], self.noise_size)

        fake_imgs = self.generator(z)

        disc_pred_fake = self.discriminator(fake_imgs)
        parsed_loss, log_vars = self.gen_loss(disc_pred_fake)

        parsed_losses.backward()
        optimizer.step()
        optimizer.zero_grad()
        return log_vars
```

对比 MMEngine 的实现：

<table class="docutils">
<thead>
  <tr>
    <th>MMCV 优化 GAN</th>
    <th>MMEngine 优化 GAN</th>
<tbody>
  <tr>
  <td valign="top">

```python
    def train_discriminator(
            self, inputs, data_sample,
            optimizer):
        real_imgs = inputs['inputs']
        z = torch.randn((real_imgs.shape[0], self.noise_size))
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

    def train_generator(self, inputs, data_sample, optimizer_wrapper):
        z = torch.randn(inputs['inputs'].shape[0], self.noise_size)

        fake_imgs = self.generator(z)

        disc_pred_fake = self.discriminator(fake_imgs)
        parsed_loss, log_vars = self.gen_loss(disc_pred_fake)

        parsed_losses.backward()
        optimizer.step()
        optimizer.zero_grad()
        return log_vars
```

</td>
  <td valign="top">

```python
    def train_discriminator(
            self, inputs, data_sample,
            optimizer_wrapper):
        real_imgs = inputs['inputs']
        z = torch.randn((real_imgs.shape[0], self.noise_size))
        with torch.no_grad():
            fake_imgs = self.generator(z)

        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)

        parsed_losses, log_vars = self.disc_loss(disc_pred_fake,
                                                 disc_pred_real)
        optimizer_wrapper.update_params(parsed_losses)
        return log_vars



    def train_generator(self, inputs, data_sample, optimizer_wrapper):
        z = torch.randn(inputs['inputs'].shape[0], self.noise_size)

        fake_imgs = self.generator(z)

        disc_pred_fake = self.discriminator(fake_imgs)
        parsed_loss, log_vars = self.gen_loss(disc_pred_fake)

        optimizer_wrapper.update_params(parsed_loss)
        return log_vars
```

</td>
</tr>
</thead>
</table>

二者的区别主要在于优化器的使用方式。此外，`train_step` 接口返回值的差异和[上一节](上层深度学习任务的模型迁移)提到的一致。

### 迁移分布式训练

MMCV 需要在执行器构建之前,使用 `MMDistributedDataParallel` 对模型进行分布式封装。MMEngine 实现了 [MMDistributedDataParallel](mmengine.model.MMDistributedDataParallel) 和 [MMSeparateDistributedDataParallel](mmengine.model.MMSeparateDistributedDataParallel) 两种分布式模型封装，供不同类型的任务选择。执行器会在构建时对模型进行分布式封装。

1. **常用训练流程**

   对于[简介](简介)中提到的常用优化流程的训练任务，即一次参数更新可以被拆解成梯度计算、参数优化、梯度清零的任务，使用 Runner 默认的 `MMDistributedDataParallel` 即可满足需求，无需为 runner 其他额外参数。

   <table class="docutils">
    <thead>
    <tr>
        <th>MMCV 分布式训练构建模型</th>
        <th>MMEngine 分布式训练</th>
    <tbody>
    <tr>
    <td valign="top">

   ```python
   model = MMDistributedDataParallel(
       model,
       device_ids=[int(os.environ['LOCAL_RANK'])],
       broadcast_buffers=False,
       find_unused_parameters=find_unused_parameters)
   ...
   runner = Runner(model=model, ...)
   ```

   </td>
    <td valign="top">

   ```python
   runner = Runner(
       model=model,
       launcher='pytorch', #开启分布式寻
       ..., # 其他参数
   )
   ```

   </td>
    </tr>
    </thead>
    </table>

&#160;

2. **分模块优化的学习任务**

   同样以训练生成对抗网络为例，生成对抗网络有两个需要分别优化的子模块，即生成器和判别器。因此需要使用 `MMSeparateDistributedDataParallel` 对模型进行封装。我们需要在构建执行器时指定：

   ```python
   cfg = dict(model_wrapper_cfg=dict(type='MMSeparateDistributedDataParallel'))
   runner = Runner(
       model=model,
       ..., # 其他配置
       launcher='pytorch',
       cfg=cfg # 模型封装配置
   )
   ```

   即可进行分布式训练。

&#160;

3. **单模块优化、自定义流程的深度学习任务**

   有时候我们需要对用自定义的优化流程来优化单个模块，这时候我们就不能复用模型基类的 `train_step`，而需要重新实现，例如我们想用同一批图片对模型优化两次，第一次开启批数据增强，第二次关闭：

   ```python
   class CustomModel(BaseModel):

       def train_step(self, data, optim_wrapper):
           data = self.data_preprocessor(data, training=True)  # 开启批数据增强
           loss = self(data, mode='loss')
           optim_wrapper.update_params(loss)
           data = self.data_preprocessor(data, training=False)  # 关闭批数据增强
           loss = self(data, mode='loss')
           optim_wrapper.update_params(loss)
   ```

   要想启用分布式训练，我们就需要重载 `MMSeparateDistributedDataParallel`，并在 `train_step` 中实现和 `CustomModel.train_step` 相同的流程（`test_step`、`val_step` 同理）。

   ```python
   class CustomDistributedDataParallel(MMSeparateDistributedDataParallel):

       def train_step(self, data, optim_wrapper):
           data = self.data_preprocessor(data, training=True)  # 开启批数据增强
           loss = self(data, mode='loss')
           optim_wrapper.update_params(loss)
           data = self.data_preprocessor(data, training=False)  # 关闭批数据增强
           loss = self(data, mode='loss')
           optim_wrapper.update_params(loss)
   ```

   最后在构建 `runner` 时指定：

   ```python
   # 指定封装类型为 `CustomDistributedDataParallel`，并基于默认参数封装模型。
   cfg = dict(model_wrapper_cfg=dict(type='CustomDistributedDataParallel'))
   runner = Runner(
       model=model,
       ..., # 其他配置
       launcher='pytorch',
       cfg=cfg # 模型封装配置
   )
   ```
