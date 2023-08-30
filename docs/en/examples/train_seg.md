# Train a Segmentation Model

This segmentation task example will be divided into the following steps:

- [Download Camvid Dataset](#download-camvid-dataset)
- [Implement Camvid Dataset](#implement-the-camvid-dataset)
- [Implement a Segmentation Model](#implement-the-segmentation-model)
- [Train with Runner](#training-with-runner)

```{note}
You can also experience the notebook example [here](https://colab.research.google.com/github/open-mmlab/mmengine/blob/main/examples/segmentation/train.ipynb).
```

## Download Camvid Dataset

First, you should download the Camvid dataset from OpenDataLab:

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

## Implement the Camvid Dataset

We have implemented the CamVid class here, which inherits from VisionDataset. Within this class, we have overridden the `__getitem__` and `__len__` methods to ensure that each index returns a dict of images and labels. Additionally, we have implemented the color_to_class dictionary to map the mask's color to the class index.

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

We utilize the Camvid dataset to create the `train_dataloader` and `val_dataloader`, which serve as the data loaders for training and validation in the subsequent Runner.

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

## Implement the Segmentation Model

The provided code defines a model class named `MMDeeplabV3`. This class is derived from `BaseModel` and incorporates the segmentation model of the DeepLabV3 architecture. It overrides the `forward` method to handle both input images and labels and supports computing losses and returning predictions in both training and prediction modes.

For additional information about `BaseModel`, you can refer to the [Model tutorial](../tutorials/model.md).

```python
from mmengine.model import BaseModel
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn.functional as F


class MMDeeplabV3(BaseModel):

    def __init__(self, num_classes):
        super().__init__()
        self.deeplab = deeplabv3_resnet50(num_classes=num_classes)

    def forward(self, imgs, data_samples=None, mode='tensor'):
        x = self.deeplab(imgs)['out']
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, data_samples['labels'])}
        elif mode == 'predict':
            return x, data_samples
```

## Training with Runner

Before training with the Runner, we need to implement the IoU (Intersection over Union) metric to evaluate the model's performance.

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

Implementing a visualization hook is also important to facilitate easier comparison between predictions and labels.

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

Finnaly, just train the model with Runner!

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

Finnaly, you can check the training results in the folder `./work_dir/{timestamp}/vis_data`.

<table class="docutils">
<thead>
<tr>
  <th>image</th>
  <th>prediction</th>
  <th>label</th>
</tr>
<tr>
  <th><img src="https://github.com/open-mmlab/mmengine/assets/57566630/de70c138-fb8e-402c-9497-574b01725b6c" width="200"></th>
  <th><img src="https://github.com/open-mmlab/mmengine/assets/57566630/ea9221e7-48ca-4515-8815-56b5ff091f53" width="200"></th>
  <th><img src="https://github.com/open-mmlab/mmengine/assets/57566630/dcb2324f-a2df-4e5c-a038-df896dde2471" width="200"></th>
</tr>
</thead>
</table>
