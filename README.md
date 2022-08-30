<div align="center">
  <img src="https://user-images.githubusercontent.com/58739961/187154444-fce76639-ac8d-429b-9354-c6fac64b7ef8.jpg" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmengine)](https://pypi.org/project/mmengine/)
[![PyPI](https://img.shields.io/pypi/v/mmengine)](https://pypi.org/project/mmengine)
[![license](https://img.shields.io/github/license/open-mmlab/mmengine.svg)](https://github.com/open-mmlab/mmengine/blob/main/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmengine.svg)](https://github.com/open-mmlab/mmengine/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmengine.svg)](https://github.com/open-mmlab/mmengine/issues)

[🤔Reporting Issues](https://github.com/open-mmlab/mmengine/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Introduction

MMEngine is a base library depending on PyTorch for deep learning model training, and supports Linux, Windows, and MacOS platforms. It has three highlights as follows.

1. Universality: MMEngine implements a high-level general trainer that

   - Support training different tasks with a small amount of code, e.g. imagenet can be trained using only 80 lines of code (pytorch example 400 lines)
   - Easy compatibility with models from popular algorithm libraries such as TIMM, TorchVision and Detectron2

2. Uniformity: MMEngine has designed an open architecture with a uniform interface, allowing

   - Users can rely on a single piece of code to achieve all tasks, e.g. MMRazor 1.x reduces 40% of the code compared to MMRazor 0.x
   - The upstream and downstream interfaces are more unified and convenient, supporting multiple backend devices while providing a unified abstraction for the upper-level algorithm library. Currently, MMEngine supports Nvidia CUDA, Mac MPS, AMD, MLU and other devices for model training.

3. Flexibility: MMEngine implements a "Lego" style training process, which supports

   - Dynamically adjust training process, optimization strategies and data augmentation strategies based on the number of iterations, loss and evaluation results. Early stopping is a typical example of adjusting training based on loss and evaluation metrics
   - Arbitrary forms of model weight averaging, such as Exponential Momentum Average (EMA) and Stochastic Weight Averaging (SWA)
   - Flexible visualization and logging control for arbitrary data and arbitrary layers during training
   - Fine-grained adjustment of the optimization strategies of each layer in the neural network model
   - Flexible control of mixed precision training

## Installation

Before installing MMEngine, make sure that PyTorch has been successfully installed following the [official guide](https://pytorch.org/get-started/locally/).

Install MMEngine

```bash
pip install -U openmim
mim install mmengine
```

Verify installation

```bash
python -c 'from mmengine.utils.dl_utils import collect_env;print(collect_env())'
```

## Get Started

<details>
<summary>Create Models</summary>

First, we need to define a **Model** that follows two conventions. One, it inherits from `BaseModel`. Second, its `forward` method needs to accept an additional parameter `mode`, in addition to several real parameters from the dataset. For training, the value of `mode` should be "loss" and the `forward` method should return a dictionary containing the "loss" key-value. For validation, `mode` should be "predict" and it should return results containing both predictitions and labels.

```python
import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel

class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels
```

</details>

<details>
<summary>Create Datasets</summary>

Next, we need to create a **Dataset** and **DataLoader** needed for training and validation.
For simple examples, we can just use built-in datasets from torchvision.

```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
train_dataloader = DataLoader(batch_size=32,
                              shuffle=True,
                              dataset=torchvision.datasets.CIFAR10(
                                  'data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(**norm_cfg)
                                  ])))
val_dataloader = DataLoader(batch_size=32,
                            shuffle=False,
                            dataset=torchvision.datasets.CIFAR10(
                                'data/cifar10',
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(**norm_cfg)
                                ])))
```

</details>

<details>
<summary>Create Metrics</summary>

For validation and testing purposes, we need to define a **Metric** to evaluate the accuracy of the model. This metric needs to inherit from `BaseMetric` and implement the `process` and `compute_metrics` methods.

```python
from mmengine.evaluator import BaseMetric

class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        # Save the results of a batch to `self.results`
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })
    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        # Returns a dictionary with the results of the evaluated metrics,
        # where the key is the name of the metric
        return dict(accuracy=100 * total_correct / total_size)
```

</details>

<details>
<summary>Create a Runner</summary>

Finally, passing Model, Data Loader, Metrics and some other configs to **Runner**.

```python
from torch.optim import SGD
from mmengine.runner import Runner

runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    # a wapper to execute back propagation and gradient update, etc.
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    # set some training configs like epochs
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
)
```

</details>

<details>
<summary>Launch Training</summary>

```python
runner.train()
```

</details>

## Contributing

We appreciate all contributions to improve MMEngine. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
