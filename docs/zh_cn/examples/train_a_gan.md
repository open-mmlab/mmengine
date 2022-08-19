# 训练生成对抗网络
生成对抗网络(GAN)可以用来生成图像视频等数据。这篇教程将带你一步步用 MMEngine 训练 GAN ！

## 设置

```python
import os

import numpy as np
import torch        
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from mmengine import BaseDataset, Runner
from mmengine.model import BaseModel


from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)
```

## 准备数据


```python
class MNISTDataset(BaseDataset):
    def __init__(self,
                 data_root,
                 pipeline,
                 test_mode=False):
        # download full dataset
        self.data_root = data_root
        MNIST(self.data_root, train=True, download=True)
        MNIST(self.data_root, train=False, download=True)
        super().__init__(
            data_root=data_root, pipeline=pipeline, test_mode=test_mode)

    def load_data_list(self):
        if self.test_mode:
            mnist_full = MNIST(self.data_root, train=True)
            mnist_dataset, _ = random_split(mnist_full, [55000, 5000])
        else:
            mnist_dataset = MNIST(self.data_root, train=False)
        return [dict(inputs=np.array(x[0])) for x in mnist_dataset]
```

## 数据流

```python
from mmgen.datasets import PackGenInputs
dataset = MNISTDataset("./data", [PackGenInputs(keys='inputs', meta_keys=[])])

```


```python
train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    collate_fn =dict(type='pseudo_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dataset)
train_dataloader = Runner.build_dataloader(train_dataloader)
```

## 模块


```python
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

## 模型


```python
def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
```


```python
from mmengine.model import ImgDataPreprocessor
data_preprocessor = ImgDataPreprocessor()
```


```python
class GAN(BaseModel):
    def __init__(self,
                 generator,
                 discriminator, 
                 noise_size,
                 data_preprocessor):
        super().__init__(data_preprocessor=data_preprocessor)
        assert generator.noise_size == noise_size
        self.generator = generator
        self.discriminator = discriminator
        self.noise_size = noise_size
    
    def train_step(self, data, optim_wrapper):
        inputs_dict, data_sample = data
        inputs_dict = self.data_preprocessor(inputs_dict, True)
        disc_optimizer_wrapper = optim_wrapper['discriminator']
        with disc_optimizer_wrapper.optim_context(self.discriminator):
            log_vars = self.train_discriminator(inputs_dict, data_sample,
                                                disc_optimizer_wrapper)


        set_requires_grad(self.discriminator, False)
        gen_optimizer_wrapper = optim_wrapper['generator']
        with gen_optimizer_wrapper.optim_context(self.generator):
            log_vars_gen = self.train_generator(
                inputs_dict, data_sample, gen_optimizer_wrapper)

        set_requires_grad(self.discriminator, True)
        log_vars.update(log_vars_gen)

        return log_vars
        
    def forward(self, batch_inputs, data_samples, mode= None):
        return batch_inputs
    
    def disc_loss(self, disc_pred_fake, disc_pred_real):
        losses_dict = dict()
        losses_dict['loss_disc_fake'] = F.binary_cross_entropy_with_logits(
            disc_pred_fake, 0. * torch.ones_like(disc_pred_fake))
        losses_dict['loss_disc_real'] = F.binary_cross_entropy_with_logits(
            disc_pred_real, 1. * torch.ones_like(disc_pred_real))

        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

    def gen_loss(self, disc_pred_fake):
        losses_dict = dict()
        losses_dict['loss_gen'] = F.binary_cross_entropy_with_logits(
            disc_pred_fake, 1. * torch.ones_like(disc_pred_fake))
        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

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


```python

model = GAN(generator, discriminator, 100, data_preprocessor)

```

## 优化器


```python
from mmengine.optim import OptimWrapperDict, OptimWrapper
opt_g = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
opt_g_wrapper = OptimWrapper(opt_g)

opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
opt_d_wrapper = OptimWrapper(opt_d)

opt_wrapper_dict = OptimWrapperDict(generator=opt_g_wrapper, discriminator=opt_d_wrapper)

```

## 训练


```python
train_cfg = dict(by_epoch=False, max_iters=5000)
runner = Runner(model, work_dir='runs/gan/', train_dataloader=train_dataloader, train_cfg=train_cfg, optim_wrapper=opt_wrapper_dict)
runner.train()
```
