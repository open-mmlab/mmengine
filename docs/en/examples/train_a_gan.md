# Train a GAN

Generative Adversarial Network (GAN) can be used to generate data such as images and videos. This tutorial will show you how to train a GAN with MMEngine step by step!

It will be divided into the following steps:

> - [Train Generative Adversarial Network](#train-a-gan)
>   - [Build a DataLoader](#building-a-dataloader)
>     - [Build a Dataset](#building-a-dataset)
> - [Build a Generator Network and a Discriminator Network](#build-a-generator-network-and-a-discriminator-network)
> - [Build a Generative Adversarial Network Model](#build-a-generative-adversarial-network-model)
> - [Build an Optimizer](#building-an-optimizer)
> - [Train with Runner](#training-with-runner)

## Building a DataLoader

### Building a Dataset

First, we will build a dataset class `MNISTDataset` for the MNIST dataset, inheriting from the base dataset class [BaseDataset](mmengine.dataset.BaseDataset), and overwrite the `load_data_list` function of the base dataset class to ensure that the return value is a `list[dict]`, where each `dict` represents a data sample.
More details about using datasets in MMEngine, refer to [the Dataset tutorial](../advanced_tutorials/basedataset.md).

```python
import numpy as np
from mmcv.transforms import to_tensor
from torch.utils.data import random_split
from torchvision.datasets import MNIST

from mmengine.dataset import BaseDataset


class MNISTDataset(BaseDataset):

    def __init__(self, data_root, pipeline, test_mode=False):
        # Download MNIST Dataset
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

Use the function `build_dataloader` in Runner to build the dataloader.

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

## Build a Generator Network and a Discriminator Network

The following code builds and instantiates a Generator and a Discriminator.

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

## Build a Generative Adversarial Network Model

In MMEngine, we use [ImgDataPreprocessor](mmengine.model.ImgDataPreprocessor) to normalize the data and convert the color channels.

```python
from mmengine.model import ImgDataPreprocessor

data_preprocessor = ImgDataPreprocessor(mean=([127.5]), std=([127.5]))
```

The following code implements the basic algorithm of GAN. To implement the algorithm using MMEngine, you need to inherit from the [BaseModel](mmengine.model.BaseModel) and implement the training process in the train_step.  GAN requires alternating training of the generator and discriminator, which are implemented by train_discriminator and train_generator and implement disc_loss and gen_loss to calculate the discriminator loss function and generator loss function.
More details about BaseModel, refer to [Model tutorial](../tutorials/model.md).

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
        # Acquiring and preprocessing data
        inputs_dict = self.data_preprocessor(data, True)
        # Training the discriminator
        disc_optimizer_wrapper = optim_wrapper['discriminator']
        with disc_optimizer_wrapper.optim_context(self.discriminator):
            log_vars = self.train_discriminator(inputs_dict,
                                                disc_optimizer_wrapper)

        # Training the generator
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

The function, set_requires_grad, is used to lock the weights of the discriminator when training the generator.

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

## Building an Optimizer

MMEngine uses [OptimWrapper](mmengine.optim.OptimWrapper) to wrap optimizers. For multiple optimizers, we use [OptimWrapperDict](mmengine.optim.OptimWrapperDict) to further wrap OptimWrapper.
More details about optimizers, refer to the [Optimizer tutorial](../tutorials/optim_wrapper.md).

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

## Training with Runner

The following code demonstrates how to use Runner for model training.
More details about Runner, please refer to the [Runner tutorial](../tutorials/runner.md).

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

Till now, we have completed an example of training a GAN. The following code can be used to view the results generated by the GAN we just trained.

![GAN generate an image](https://user-images.githubusercontent.com/22982797/186811532-1517a0f7-5452-4a39-b6d0-6c685e4545e2.png)

If you want to learn more about using MMEngine to implement GAN and generative models, we highly recommend you try the generative framework [MMGeneration](https://github.com/open-mmlab/mmgeneration/tree/dev-1.x) based on MMEngine.
