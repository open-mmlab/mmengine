# 迁移 MMCV 模型到 MMEngine

## 简介

MMCV 早期支持的计算机视觉任务，例如目标检测、物体识别等，都采用了一种典型的模型参数优化流程，可以被归纳为以下四个步骤：

1. 计算损失
2. 计算梯度
3. 更新参数
4. 梯度清零

上述流程的一大特点就是调用位置统一（在训练迭代后调用）、执行步骤统一（依次执行步骤 1->2->3->4），非常契合[钩子（Hook）](../design/hook.md)的设计原则，因此这类任务通常会使用 `Hook`
来优化模型。MMCV 为此实现了一系列的 `Hook`，例如 `OptimizerHook`（单精度训练）、`Fp16OptimizerHook`（混合精度训练） 和 `GradientCumulativeFp16OptimizerHook`（混合精度训练 + 梯度累加），为这类任务提供各种优化策略。

一些例如生成对抗网络（GAN），自监督（Self-supervision）等领域的算法一般有更加灵活的训练流程，这类流程并不满足调用位置统一、执行步骤统一的原则，难以使用 `Hook` 对参数进行优化。为了支持训练这类任务，MMCV 的执行器会在调用 `model.train_step` 时，额外传入 `optimizer` 参数，让模型在 `train_step` 里实现自定义的优化流程。这样虽然可以支持训练这类任务，但也会导致无法使用各种 `OptimizerHook`，需要算法在 `train_step` 中实现混合精度训练、梯度累加等训练策略。

为了统一深度学习任务的参数优化流程，MMEngine 设计了[优化器封装](mmengine.optim.OptimWrapper)，集成了混合精度训练、梯度累加等训练策略，各类深度学习任务一律在 `model.train_step` 里执行参数优化流程。

## 优化流程的迁移

### 常用的参数更新流程

考虑到目标检测、物体识别一类的深度学习任务参数优化的流程基本一致，我们可以通过继承[模型基类](../tutorials/model.md)来完成迁移。

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
        feat = self.linear(img)
        # 被 `train_step` 调用，返回用于更新参数的损失字典
        if mode == 'loss':
            loss1 = (feat - label).pow(2)
            loss2 = (feat - label).abs()
            return dict(loss1=loss1, loss2=loss2)
        # 被 `val_step` 调用，返回传给 `evaluator` 的预测结果
        elif mode == 'predict':
            return [_feat for _feat in feat]
        # tensor 模式，功能详见模型教程文档： tutorials/model.md
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

MMEngine 实现了模型基类，模型基类在 `train_step` 里实现了 `OptimizerHook` 的优化流程。因此上例中，我们无需实现 `train_step`，运行时直接调用基类的 `train_step`。

<table class="docutils">
<thead>
  <tr>
    <th>MMCV 模型</th>
    <th>MMEngine 模型</th>
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

    # 模型基类 `train_step` 等效代码
    # def train_step(self, data, optim_wrapper):
    #     data = self.data_preprocessor(data)
    #     loss_dict = self(*data, mode='loss')
    #     loss_dict['loss1'] = loss_dict['loss1'].sum()
    #     loss_dict['loss2'] = loss_dict['loss2'].sum()
    #     loss = (loss_dict['loss1'] + loss_dict['loss2']).sum()
    #     调用优化器封装更新模型参数
    #     optim_wrapper.update_params(loss)
    #     return loss_dict
```

</div>
  </td>
</tr>
</thead>
</table>

关于等效代码中的[数据处理器（data_preprocessor）](mmengine.model.BaseDataPreprocessor)和[优化器封装（optim_wrapper）](mmengine.optim.OptimWrapper)的说明，详见[模型教程](../tutorials/model.md#数据预处理器datapreprocessor)和[优化器封装教程](../tutorials/optim_wrapper.md)。

模型具体差异如下：

- `MMCVToyModel` 继承自 `nn.Module`，而 `MMEngineToyModel` 继承自 `BaseModel`
- `MMCVToyModel` 必须实现 `train_step`，且必须返回损失字典，损失字典包含 `loss` 和 `log_vars` 和 `num_samples` 字段。`MMEngineToyModel` 继承自 `BaseModel`，只需要实现 `forward` 接口，并返回损失字典，损失字典的每一个值必须是可微的张量
- `MMCVToyModel` 和 `MMEngineModel` 的 `forward` 的接口需要匹配 `train_step` 中的调用方式，由于 `MMEngineToyModel` 直接调用基类的 `train_step` 方法，因此 `forward` 需要接受参数 `mode`，具体规则详见[模型教程文档](../tutorials/model.md)

### 自定义的参数更新流程

以训练生成对抗网络为例，生成器和判别器的优化需要交替进行，且优化流程可能会随着迭代次数的增多发生变化，因此很难使用 `OptimizerHook` 来满足这种需求。在基于 MMCV 训练生成对抗网络时，通常会在模型的 `train_step` 接口中传入 `optimizer`，然后在 `train_step` 里实现自定义的参数更新逻辑。这种训练流程和 MMEngine 非常相似，只不过 MMEngine 在 `train_step` 接口中传入[优化器封装](../tutorials/optim_wrapper.md)，能够更加简单地优化模型。

参考[训练生成对抗网络](../examples/train_a_gan.md)，MMCV 和 MMEngine 的对比实现如下：

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

    def train_generator(self, inputs, optimizer_wrapper):
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

二者的区别主要在于优化器的使用方式。此外，`train_step` 接口返回值的差异和[上一节](#常用的参数更新流程)提到的一致。

## 验证/测试流程的迁移

基于 MMCV 执行器实现的模型通常不需要为验证、测试流程提供独立的 `val_step`、`test_step`（测试流程由 `EvalHook` 实现，这里不做展开）。基于 MMEngine 执行器实现的模型则有所不同，[ValLoop](mmengine.runner.ValLoop)、[TestLoop](mmengine.runner.TestLoop) 会分别调用模型的 `val_step` 和 `test_step` 接口，输出会进一步传给 [Evaluator.process](mmengine.evaluator.Evaluator.process)。因此模型的 `val_step` 和 `test_step` 接口输出需要和 `Evaluator.process` 的入参（第一个参数）对齐，即返回列表（推荐，也可以是其他可迭代类型）类型的结果。列表中的每一个元素代表一个批次（batch）数据中每个样本的预测结果。模型的 `test_step` 和 `val_step` 会调 `forward` 接口（详见[模型教程文档](../tutorials/model.md)），因此在上一节的模型示例中，模型 `forward` 的 `predict` 模式会将 `feat` 切片后，以列表的形式返回预测结果。

```python

class MMEngineToyModel(BaseModel):

    ...
    def forward(self, img, label, mode):
        if mode == 'loss':
            ...
        elif mode == 'predict':
            # 把一个 batch 的预测结果切片成列表，每个元素代表一个样本的预测结果
            return [_feat for _feat in feat]
        else:
            ...
            # tensor 模式，功能详见模型教程文档： tutorials/model.md
```

## 迁移分布式训练

MMCV 需要在执行器构建之前,使用 `MMDistributedDataParallel` 对模型进行分布式封装。MMEngine 实现了 [MMDistributedDataParallel](mmengine.model.MMDistributedDataParallel) 和 [MMSeparateDistributedDataParallel](mmengine.model.MMSeparateDistributedDataParallel) 两种分布式模型封装，供不同类型的任务选择。执行器会在构建时对模型进行分布式封装。

1. **常用训练流程**

   对于[简介](#简介)中提到的常用优化流程的训练任务，即一次参数更新可以被拆解成梯度计算、参数优化、梯度清零的任务，使用 Runner 默认的 `MMDistributedDataParallel` 即可满足需求，无需为 runner 其他额外参数。

   <table class="docutils">
    <thead>
    <tr>
        <th>MMCV 分布式训练构建模型</th>
        <th>MMEngine 分布式训练</th>
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
    launcher='pytorch', #开启分布式训练
    ..., # 其他参数
)
```

</div>
  </td>
  </tr>
</thead>
</table>

&#160;

2. **以自定义流程分模块优化模型的学习任务**

   同样以训练生成对抗网络为例，生成对抗网络有两个需要分别优化的子模块，即生成器和判别器。因此需要使用 `MMSeparateDistributedDataParallel` 对模型进行封装。我们需要在构建执行器时指定：

   ```python
   cfg = dict(model_wrapper_cfg='MMSeparateDistributedDataParallel')
   runner = Runner(
       model=model,
       ...,
       launcher='pytorch',
       cfg=cfg)
   ```

   即可进行分布式训练。

&#160;

3. **以自定义流程优化整个模型的深度学习任务**

   有时候我们需要用自定义的优化流程来优化单个模块，这时候我们就不能复用模型基类的 `train_step`，而需要重新实现，例如我们想用同一批图片对模型优化两次，第一次开启批数据增强，第二次关闭：

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
       ...,
       launcher='pytorch',
       cfg=cfg
   )
   ```
