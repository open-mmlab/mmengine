# 训练语义分割模型

语义分割的样例大体可以分成四个步骤:

- [下载 Camvid 数据集](#下载-camvid-数据集)
- [实现 Camvid 数据类](#实现-camvid-数据类)
- [实现语义分割模型](#实现语义分割模型)
- [使用 Runner 训练模型](#使用-runner-训练模型)

```{note}
如果你更喜欢 notebook 风格的样例，也可以在[此处](https://colab.research.google.com/github/open-mmlab/mmengine/blob/main/examples/segmentation/train.ipynb) 体验。
```

## 下载 Camvid 数据集

首先，从 opendatalab 下载 Camvid 数据集:

```bash
# https://opendatalab.com/CamVid
# Configure install
pip install opendatalab
# Upgraded version
pip install -U opendatalab
# Login
odl login
# Download this dataset
mkdir data
odl get CamVid -d data
# Preprocess data in Linux. You should extract the files to data manually in
# Windows
tar -xzvf data/CamVid/raw/CamVid.tar.gz.00 -C ./data
```

## 实现 Camvid 数据类

实现继承自 VisionDataset 的 CamVid 数据类。在这个类中，我们重写了`__getitem__`和`__len__`方法，以确保每个索引返回一个包含图像和标签的字典。此外，我们还实现了color_to_class字典，将 mask 的颜色映射到类别索引。

```python
import os
import numpy as np
from torchvision.datasets import VisionDataset
from PIL import Image
import csv


def create_palette(csv_filepath):
    color_to_class = {}
    with open(csv_filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            r, g, b = int(row['r']), int(row['g']), int(row['b'])
            color_to_class[(r, g, b)] = idx
    return color_to_class

class CamVid(VisionDataset):

    def __init__(self,
                 root,
                 img_folder,
                 mask_folder,
                 transform=None,
                 target_transform=None):
        super().__init__(
            root, transform=transform, target_transform=target_transform)
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.images = list(
            sorted(os.listdir(os.path.join(self.root, img_folder))))
        self.masks = list(
            sorted(os.listdir(os.path.join(self.root, mask_folder))))
        self.color_to_class = create_palette(
            os.path.join(self.root, 'class_dict.csv'))

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.img_folder, self.images[index])
        mask_path = os.path.join(self.root, self.mask_folder,
                                 self.masks[index])

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')  # Convert to RGB

        if self.transform is not None:
            img = self.transform(img)

        # Convert the RGB values to class indices
        mask = np.array(mask)
        mask = mask[:, :, 0] * 65536 + mask[:, :, 1] * 256 + mask[:, :, 2]
        labels = np.zeros_like(mask, dtype=np.int64)
        for color, class_index in self.color_to_class.items():
            rgb = color[0] * 65536 + color[1] * 256 + color[2]
            labels[mask == rgb] = class_index

        if self.target_transform is not None:
            labels = self.target_transform(labels)
        data_samples = dict(
            labels=labels, img_path=img_path, mask_path=mask_path)
        return img, data_samples

    def __len__(self):
        return len(self.images)

```

基于 CamVid 数据类，选择相应的数据增强方式，构建 train_dataloader 和 val_dataloader，供后续 runner 使用

```python
import torch
import torchvision.transforms as transforms

norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(**norm_cfg)])

target_transform = transforms.Lambda(
        lambda x: torch.tensor(np.array(x), dtype=torch.long))

train_set = CamVid(
    'data/CamVid',
    img_folder='train',
    mask_folder='train_labels',
    transform=transform,
    target_transform=target_transform)

valid_set = CamVid(
    'data/CamVid',
    img_folder='val',
    mask_folder='val_labels',
    transform=transform,
    target_transform=target_transform)

train_dataloader = dict(
    batch_size=3,
    dataset=train_set,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'))

val_dataloader = dict(
    batch_size=3,
    dataset=valid_set,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'))
```

## 实现语义分割模型

定义一个名为`MMDeeplabV3`的模型类。该类继承自`BaseModel`，并集成了DeepLabV3架构的分割模型。`MMDeeplabV3` 重写了`forward`方法，以处理输入图像和标签，并支持在训练和预测模式下计算损失和返回预测结果。

关于`BaseModel`的更多信息，请参考[模型教程](../tutorials/model.md)。

```python
from mmengine.model import BaseModel
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn.functional as F


class MMDeeplabV3(BaseModel):

    def __init__(self, num_classes):
        super().__init__()
        self.deeplab = deeplabv3_resnet50()
        self.deeplab.classifier[4] = torch.nn.Conv2d(
            256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, imgs, data_samples=None, mode='tensor'):
        x = self.deeplab(imgs)['out']
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, data_samples['labels'])}
        elif mode == 'predict':
            return x, data_samples
```

## 使用 Runner 训练模型

在使用 Runner 进行训练之前，我们需要实现 IoU（交并比）指标来评估模型的性能。

```python
from mmengine.evaluator import BaseMetric

class IoU(BaseMetric):

    def process(self, data_batch, data_samples):
        preds, labels = data_samples[0], data_samples[1]['labels']
        preds = torch.argmax(preds, dim=1)
        intersect = (labels == preds).sum()
        union = (torch.logical_or(preds, labels)).sum()
        iou = (intersect / union).cpu()
        self.results.append(
            dict(batch_size=len(labels), iou=iou * len(labels)))

    def compute_metrics(self, results):
        total_iou = sum(result['iou'] for result in self.results)
        num_samples = sum(result['batch_size'] for result in self.results)
        return dict(iou=total_iou / num_samples)
```

实现可视化钩子（Hook）也很重要，它可以便于更轻松地比较模型预测的好坏。

```python
from mmengine.hooks import Hook
import shutil
import cv2
import os.path as osp


class SegVisHook(Hook):

    def __init__(self, data_root, vis_num=1) -> None:
        super().__init__()
        self.vis_num = vis_num
        self.palette = create_palette(osp.join(data_root, 'class_dict.csv'))

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch=None,
                       outputs=None) -> None:
        if batch_idx > self.vis_num:
            return
        preds, data_samples = outputs
        img_paths = data_samples['img_path']
        mask_paths = data_samples['mask_path']
        _, C, H, W = preds.shape
        preds = torch.argmax(preds, dim=1)
        for idx, (pred, img_path,
                  mask_path) in enumerate(zip(preds, img_paths, mask_paths)):
            pred_mask = np.zeros((H, W, 3), dtype=np.uint8)
            runner.visualizer.set_image(pred_mask)
            for color, class_id in self.palette.items():
                runner.visualizer.draw_binary_masks(
                    pred == class_id,
                    colors=[color],
                    alphas=1.0,
                )
            # Convert RGB to BGR
            pred_mask = runner.visualizer.get_image()[..., ::-1]
            saved_dir = osp.join(runner.log_dir, 'vis_data', str(idx))
            os.makedirs(saved_dir, exist_ok=True)

            shutil.copyfile(img_path,
                            osp.join(saved_dir, osp.basename(img_path)))
            shutil.copyfile(mask_path,
                            osp.join(saved_dir, osp.basename(mask_path)))
            cv2.imwrite(
                osp.join(saved_dir, f'pred_{osp.basename(img_path)}'),
                pred_mask)
```

准备完毕，让我们用 Runner 开始训练吧！

```python
from torch.optim import AdamW
from mmengine.optim import AmpOptimWrapper
from mmengine.runner import Runner


num_classes = 32  # Modify to actual number of categories.

runner = Runner(
    model=MMDeeplabV3(num_classes),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(
        type=AmpOptimWrapper, optimizer=dict(type=AdamW, lr=2e-4)),
    train_cfg=dict(by_epoch=True, max_epochs=10, val_interval=10),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=IoU),
    custom_hooks=[SegVisHook('data/CamVid')],
    default_hooks=dict(checkpoint=dict(type='CheckpointHook', interval=1)),
)
runner.train()
```

训练完成后，你可以在 `./work_dir/{timestamp}/vis_data` 文件夹中找到可视化结果，如下图所示：

<table class="docutils">
<thead>
<tr>
  <th>原图</th>
  <th>预测结果</th>
  <th>标签</th>
</tr>
<tr>
  <th><img src="https://github.com/open-mmlab/mmengine/assets/57566630/de70c138-fb8e-402c-9497-574b01725b6c" width="200"></th>
  <th><img src="https://github.com/open-mmlab/mmengine/assets/57566630/ea9221e7-48ca-4515-8815-56b5ff091f53" width="200"></th>
  <th><img src="https://github.com/open-mmlab/mmengine/assets/57566630/dcb2324f-a2df-4e5c-a038-df896dde2471" width="200"></th>
</tr>
</thead>
</table>
