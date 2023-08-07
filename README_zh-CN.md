<div align="center">
  <img src="https://user-images.githubusercontent.com/58739961/187154444-fce76639-ac8d-429b-9354-c6fac64b7ef8.jpg" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
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

[📘使用文档](https://mmengine.readthedocs.io/zh_CN/latest/) |
[🛠️安装教程](https://mmengine.readthedocs.io/zh_CN/latest/get_started/installation.html) |
[🤔报告问题](https://github.com/open-mmlab/mmengine/issues/new/choose)

</div>

<div align="center">

[English](README.md) | 简体中文

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219255827-67c1a27f-f8c5-46a9-811d-5e57448c61d1.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.com/channels/1037617289144569886/1073056342287323168" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://space.bilibili.com/1293512903" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026751-d7d14cce-a7c9-4e82-9942-8375fca65b99.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.zhihu.com/people/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026120-ba71e48b-6e94-4bd4-b4e9-b7d175b5e362.png" width="3%" alt="" /></a>
</div>

## 最近进展

最新版本 v0.8.4 在 2023.08.03 发布。

亮点：

- 支持使用 `efficient_conv_bn_eval` 参数开启更高效的 `ConvBN` 推理模式。详见[节省显存文档](https://mmengine.readthedocs.io/zh_CN/latest/common_usage/save_gpu_memory.html)

- 新增微调 Llama2 的[示例](./examples/llama2/)。

- 支持使用 [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html?highlight=fsdp) 和 [DeepSpeed](https://www.deepspeed.ai/) 进行训练。可阅读[大模型训练](https://mmengine.readthedocs.io/zh_cn/latest/common_usage/large_model_training.html)了解用法。

- 引入纯 Python 风格的配置文件：

  - 支持在 IDE 中导航到基础配置文件
  - 支持在 IDE 中导航到基础变量
  - 支持在 IDE 中导航到类的源代码
  - 支持继承包含相同字段的两个配置文件
  - 在加载配置文件时不需要其他第三方依赖

  请参考[教程](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html#python-beta)以获取更详细的用法说明。

  ![new-config-zh_cn](https://github.com/open-mmlab/mmengine/assets/57566630/c2da9a73-c911-4f78-8253-e3f29496d9f8)

如果想了解更多版本更新细节和历史信息，请阅读[更新日志](./docs/en/notes/changelog.md#v083-08032023)

## 目录

- [简介](#简介)
- [安装](#安装)
- [快速上手](#快速上手)
- [了解更多](#了解更多)
- [贡献指南](#贡献指南)
- [引用](#引用)
- [开源许可证](#开源许可证)
- [生态项目](#生态项目)
- [OpenMMLab 的其他项目](#openmmlab-的其他项目)
- [欢迎加入 OpenMMLab 社区](#欢迎加入-openmmlab-社区)

## 简介

MMEngine 是一个基于 PyTorch 实现的，用于训练深度学习模型的基础库。它为开发人员提供了坚实的工程基础，以此避免在工作流上编写冗余代码。作为 OpenMMLab 所有代码库的训练引擎，其在不同研究领域支持了上百个算法。此外，MMEngine 也可以用于非 OpenMMLab 项目中。

主要特性：

1. **通用且强大的执行器**：

   - 支持用少量代码训练不同的任务，例如仅使用 80 行代码就可以训练 ImageNet（原始 PyTorch 示例需要 400 行）。
   - 轻松兼容流行的算法库（如 TIMM、TorchVision 和 Detectron2）中的模型。

2. **接口统一的开放架构**：

   - 使用统一的接口处理不同的算法任务，例如，实现一个方法并应用于所有的兼容性模型。
   - 上下游的对接更加统一便捷，在为上层算法库提供统一抽象的同时，支持多种后端设备。目前 MMEngine 支持 Nvidia CUDA、Mac MPS、AMD、MLU 等设备进行模型训练。

3. **可定制的训练流程**：

   - 定义了“乐高”式的训练流程。
   - 提供了丰富的组件和策略。
   - 使用不同等级的 API 控制训练过程。

![mmengine_dataflow](https://github.com/open-mmlab/mmengine/assets/58739961/267db9cb-72e4-4af2-a58b-877b30091acc)

## 安装

在安装 MMEngine 之前，请确保 PyTorch 已成功安装在环境中，可以参考 [PyTorch 官方安装文档](https://pytorch.org/get-started/locally/)。

安装 MMEngine

```bash
pip install -U openmim
mim install mmengine
```

验证是否安装成功

```bash
python -c 'from mmengine.utils.dl_utils import collect_env;print(collect_env())'
```

更多安装方式请阅读[安装文档](https://mmengine.readthedocs.io/zh_CN/latest/get_started/installation.html)。

## 快速上手

以在 CIFAR-10 数据集上训练一个 ResNet-50 模型为例，我们将使用 80 行以内的代码，利用 MMEngine 构建一个完整的、可配置的训练和验证流程。

<details>
<summary>构建模型</summary>

首先，我们需要构建一个**模型**，在 MMEngine 中，我们约定这个模型应当继承 `BaseModel`，并且其 `forward` 方法除了接受来自数据集的若干参数外，还需要接受额外的参数 `mode`。

- 对于训练，我们需要 `mode` 接受字符串 "loss"，并返回一个包含 "loss" 字段的字典。
- 对于验证，我们需要 `mode` 接受字符串 "predict"，并返回同时包含预测信息和真实信息的结果。

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
<summary>构建数据集</summary>

其次，我们需要构建训练和验证所需要的**数据集（Dataset）**和**数据加载器（DataLoader）**。在该示例中，我们使用 TorchVision 支持的方式构建数据集。

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
<summary>构建评测指标</summary>

为了进行验证和测试，我们需要定义模型推理结果的**评测指标**。我们约定这一评测指标需要继承 `BaseMetric`，并实现 `process` 和 `compute_metrics` 方法。

```python
from mmengine.evaluator import BaseMetric

class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        # 将一个批次的中间结果保存至 `self.results`
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })
    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        # 返回保存有评测指标结果的字典，其中键为指标名称
        return dict(accuracy=100 * total_correct / total_size)
```

</details>

<details>
<summary>构建执行器</summary>

最后，我们利用构建好的`模型`，`数据加载器`，`评测指标`构建一个**执行器（Runner）**，并伴随其他的配置信息，如下所示。

```python
from torch.optim import SGD
from mmengine.runner import Runner

runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    # 优化器包装，用于模型优化，并提供 AMP、梯度累积等附加功能
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    # 训练配置，例如 epoch 等
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
)
```

</details>

<details>
<summary>开始训练</summary>

```python
runner.train()
```

</details>

## 了解更多

<details>
<summary>入门教程</summary>

- [执行器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html)
- [数据集与数据加载器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/dataset.html)
- [模型](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/model.html)
- [模型精度评测](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/evaluation.html)
- [优化器封装](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html)
- [优化器参数调整策略](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/param_scheduler.html)
- [钩子](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/hook.html)

</details>

<details>
<summary>进阶教程</summary>

- [注册器](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/registry.html)
- [配置](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html)
- [数据集基类](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/basedataset.html)
- [数据变换](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/data_transform.html)
- [权重初始化](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/initialize.html)
- [可视化](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/visualization.html)
- [抽象数据接口](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/data_element.html)
- [分布式通信原语](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/distributed.html)
- [记录日志](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/logging.html)
- [文件读写](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/fileio.html)
- [全局管理器 (ManagerMixin)](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/manager_mixin.html)
- [跨库调用模块](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/cross_library.html)
- [测试时增强](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/test_time_augmentation.html)

</details>

<details>
<summary>示例</summary>

- [训练生成对抗网络](https://mmengine.readthedocs.io/zh_CN/latest/examples/train_a_gan.html)

</details>

<details>
<summary>常用功能</summary>

- [恢复训练](https://mmengine.readthedocs.io/zh_CN/latest/common_usage/resume_training.html)
- [加速训练](https://mmengine.readthedocs.io/zh_CN/latest/common_usage/speed_up_training.html)
- [节省显存](https://mmengine.readthedocs.io/zh_CN/latest/common_usage/save_gpu_memory.html)

</details>

<details>
<summary>架构设计</summary>

- [钩子](https://mmengine.readthedocs.io/zh_CN/latest/design/hook.html)
- [执行器](https://mmengine.readthedocs.io/zh_CN/latest/design/runner.html)
- [模型精度评测](https://mmengine.readthedocs.io/zh_CN/latest/design/evaluation.html)
- [可视化](https://mmengine.readthedocs.io/zh_CN/latest/design/visualization.html)
- [日志系统](https://mmengine.readthedocs.io/zh_CN/latest/design/logging.html)
- [推理接口](https://mmengine.readthedocs.io/zh_CN/latest/design/infer.html)

</details>

<details>
<summary>迁移指南</summary>

- [迁移 MMCV 执行器到 MMEngine](https://mmengine.readthedocs.io/zh_CN/latest/migration/runner.html)
- [迁移 MMCV 钩子到 MMEngine](https://mmengine.readthedocs.io/zh_CN/latest/migration/hook.html)
- [迁移 MMCV 模型到 MMEngine](https://mmengine.readthedocs.io/zh_CN/latest/migration/model.html)
- [迁移 MMCV 参数调度器到 MMEngine](https://mmengine.readthedocs.io/zh_CN/latest/migration/param_scheduler.html)
- [数据变换类的迁移](https://mmengine.readthedocs.io/zh_CN/latest/migration/transform.html)

</details>

## 贡献指南

我们感谢所有的贡献者为改进和提升 MMEngine 所作出的努力。请参考[贡献指南](CONTRIBUTING_zh-CN.md)来了解参与项目贡献的相关指引。

## 引用

如果您觉得 MMEngine 对您的研究有所帮助，请考虑引用它：

```
@article{mmengine2022,
  title   = {{MMEngine}: OpenMMLab Foundational Library for Training Deep Learning Models},
  author  = {MMEngine Contributors},
  howpublished = {\url{https://github.com/open-mmlab/mmengine}},
  year={2022}
}
```

## 开源许可证

该项目采用 [Apache 2.0 license](LICENSE) 开源许可证。

## 生态项目

- [APES: Attention-based Point Cloud Edge Sampling](https://github.com/JunweiZheng93/APES)
- [DiffEngine: diffusers training toolbox with mmengine](https://github.com/okotaku/diffengine)

## OpenMMLab 的其他项目

- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMLab 项目、算法、模型的统一入口
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MMEval](https://github.com/open-mmlab/mmeval): 统一开放的跨框架算法评测库
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab 深度学习预训练工具箱
- [MMagic](https://github.com/open-mmlab/mmagic): OpenMMLab 新一代人工智能内容生成（AIGC）工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO 系列工具箱与测试基准
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [Playground](https://github.com/open-mmlab/playground): 收集和展示 OpenMMLab 相关的前沿、有趣的社区项目

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=K0QI8ByU)，或通过添加微信“Open小喵Lab”加入官方交流微信群。

<div align="center">
<img src="https://user-images.githubusercontent.com/58739961/187154320-f3312cdf-31f2-4316-9dbb-8d7b0e1b7e08.jpg" height="400" />  <img src="https://user-images.githubusercontent.com/25839884/203904835-62392033-02d4-4c73-a68c-c9e4c1e2b07f.jpg" height="400" />  <img src="https://user-images.githubusercontent.com/58739961/187151778-d17c1368-125f-4fde-adbe-38cc6eb3be98.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
