# 训练生成对抗网络

生成对抗网络(Generative Adversarial Network, GAN)可以用来生成图像视频等数据。这篇教程将带你一步步用 MMEngine 训练 GAN ！

我们可以通过以下步骤来训练一个生成对抗网络。

- [训练生成对抗网络](#训练生成对抗网络)
  - [构建数据加载器](#构建数据加载器)
    - [构建数据集](#构建数据集)
  - [构建生成器网络和判别器网络](#构建生成器网络和判别器网络)
  - [构建一个生成对抗网络模型](#构建一个生成对抗网络模型)
  - [构建优化器](#构建优化器)
  - [使用执行器进行训练](#使用执行器进行训练)

## 构建数据加载器

### 构建数据集

接下来, 我们为 MNIST 数据集构建一个数据集类 `MNISTDataset`, 继承自数据集基类 [BaseDataset](mmengine.dataset.BaseDataset), 并且重载数据集基类的 `load_data_list` 函数, 保证返回值为 `list[dict]`，其中每个 `dict` 代表一个数据样本。更多关于 MMEngine 中数据集的用法，可以参考[数据集教程](../advanced_tutorials/basedataset.md)。

```python
import numpy as np
from mmcv.transforms import to_tensor
from torch.utils.data import random_split
from torchvision.datasets import MNIST

from mmengine.dataset import BaseDataset


class MNISTDataset(BaseDataset):

    def __init__(self, data_root, pipeline, test_mode=False):
        # 下载 MNIST 数据集
        if test_mode:
            mnist_full = MNIST(data_root, train=True, download=True)
            self.mnist_dataset, _ = random_split(mnist_full, [55000, 5000])
        else:
            self.mnist_dataset = MNIST(data_root, train=False, download=True)

        super().__init__(
            data_root=data_root, pipeline=pipeline, test_mode=test_mode)

    @staticmethod
    def totensor(img):
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return to_tensor(img)

    def load_data_list(self):
        return [
            dict(inputs=self.totensor(np.array(x[0]))) for x in self.mnist_dataset
        ]


dataset = MNISTDataset("./data", [])

```

使用 Runner 中的函数 build_dataloader 来构建数据加载器。

```python
import os
import torch
from mmengine.runner import Runner

NUM_WORKERS = int(os.cpu_count() / 2)
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dataset)
train_dataloader = Runner.build_dataloader(train_dataloader)
```

## 构建生成器网络和判别器网络

下面的代码构建并实例化了一个生成器(Generator)和一个判别器(Discriminator)。

```python
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_size, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.noise_size = noise_size

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(noise_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
```

```python
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
```

```python
generator = Generator(100, (1, 28, 28))
discriminator = Discriminator((1, 28, 28))
```

## 构建一个生成对抗网络模型

在使用 MMEngine 时，我们用 [ImgDataPreprocessor](mmengine.model.ImgDataPreprocessor) 来对数据进行归一化和颜色通道的转换。

```python
from mmengine.model import ImgDataPreprocessor

data_preprocessor = ImgDataPreprocessor(mean=([127.5]), std=([127.5]))
```

下面的代码实现了基础 GAN 的算法。使用 MMEngine 实现算法类，需要继承 [BaseModel](mmengine.model.BaseModel) 基类，在 train_step 中实现训练过程。GAN 需要交替训练生成器和判别器，分别由 train_discriminator 和 train_generator 实现，并实现 disc_loss 和 gen_loss 计算判别器损失函数和生成器损失函数。
关于 BaseModel 的更多信息，请参考[模型教程](../tutorials/model.md).

```python
import torch.nn.functional as F
from mmengine.model import BaseModel

class GAN(BaseModel):

    def __init__(self, generator, discriminator, noise_size,
                 data_preprocessor):
        super().__init__(data_preprocessor=data_preprocessor)
        assert generator.noise_size == noise_size
        self.generator = generator
        self.discriminator = discriminator
        self.noise_size = noise_size

    def train_step(self, data, optim_wrapper):
        # 获取数据和数据预处理
        inputs_dict = self.data_preprocessor(data, True)
        # 训练判别器
        disc_optimizer_wrapper = optim_wrapper['discriminator']
        with disc_optimizer_wrapper.optim_context(self.discriminator):
            log_vars = self.train_discriminator(inputs_dict,
                                                disc_optimizer_wrapper)

        # 训练生成器
        set_requires_grad(self.discriminator, False)
        gen_optimizer_wrapper = optim_wrapper['generator']
        with gen_optimizer_wrapper.optim_context(self.generator):
            log_vars_gen = self.train_generator(inputs_dict,
                                                gen_optimizer_wrapper)

        set_requires_grad(self.discriminator, True)
        log_vars.update(log_vars_gen)

        return log_vars

    def forward(self, batch_inputs, data_samples=None, mode=None):
        return self.generator(batch_inputs)

    def disc_loss(self, disc_pred_fake, disc_pred_real):
        losses_dict = dict()
        losses_dict['loss_disc_fake'] = F.binary_cross_entropy(
            disc_pred_fake, 0. * torch.ones_like(disc_pred_fake))
        losses_dict['loss_disc_real'] = F.binary_cross_entropy(
            disc_pred_real, 1. * torch.ones_like(disc_pred_real))

        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

    def gen_loss(self, disc_pred_fake):
        losses_dict = dict()
        losses_dict['loss_gen'] = F.binary_cross_entropy(
            disc_pred_fake, 1. * torch.ones_like(disc_pred_fake))
        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

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

其中一个函数 set_requires_grad 用来锁定训练生成器时判别器的权重。

```python
def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not.
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
```

```python

model = GAN(generator, discriminator, 100, data_preprocessor)

```

## 构建优化器

MMEngine 使用 [OptimWrapper](mmengine.optim.OptimWrapper) 来封装优化器，对于多个优化器的情况，使用 [OptimWrapperDict](mmengine.optim.OptimWrapperDict) 对 OptimWrapper 再进行一次封装。
关于优化器的更多信息，请参考[优化器教程](../tutorials/optim_wrapper.md).

```python
from mmengine.optim import OptimWrapper, OptimWrapperDict

opt_g = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
opt_g_wrapper = OptimWrapper(opt_g)

opt_d = torch.optim.Adam(
    discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
opt_d_wrapper = OptimWrapper(opt_d)

opt_wrapper_dict = OptimWrapperDict(
    generator=opt_g_wrapper, discriminator=opt_d_wrapper)

```

## 使用执行器进行训练

下面的代码演示了如何使用 Runner 进行模型训练。关于 Runner 的更多信息，请参考[执行器教程](../tutorials/runner.md)。

```python
train_cfg = dict(by_epoch=True, max_epochs=220)
runner = Runner(
    model,
    work_dir='runs/gan/',
    train_dataloader=train_dataloader,
    train_cfg=train_cfg,
    optim_wrapper=opt_wrapper_dict)
runner.train()
```

到这里，我们就完成了一个 GAN 的训练，通过下面的代码可以查看刚才训练的 GAN 生成的结果。

```python
z = torch.randn(64, 100).cuda()
img = model(z)

from torchvision.utils import save_image
save_image(img, "result.png", normalize=True)
```

![GAN生成图像](https://user-images.githubusercontent.com/22982797/186811532-1517a0f7-5452-4a39-b6d0-6c685e4545e2.png)

如果你想了解更多如何使用 MMEngine 实现 GAN 和生成模型，我们强烈建议你使用同样基于 MMEngine 开发的生成框架 [MMGen](https://github.com/open-mmlab/mmgeneration/tree/dev-1.x)。
