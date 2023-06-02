# Segmentation Task Example

This segmentation task example will be divided into the following steps:

> - \[Download Camvid Dataset\](#Download Camvid Dataset)
> - \[Building Camvid Dataset\](#Building Camvid Dataset)
> - \[Build a Segmentation Model\](#Build a Segmentation Model)
> - [Train with Runner](#training-with-runner)

## Download Camvid Dataset

First, you should get the collated Camvid dataset on OpenDataLab to use for the segmentation training example. The official download steps are shown below.

```bash
# https://opendatalab.com/CamVid
# Configure install
pip install opendatalab
# Upgraded version
pip install -U opendatalab
# Login
odl login
# Download this dataset
odl get    CamVid
# preprocess data
python prepare_data.py
```

Or, after register your account, you can use the following code to download the dataset.

```python
import os
import shutil
import subprocess
import tarfile

print('Downloading dataset from https://opendatalab.com/CamVid.\n')
print('You need register an account before downloading the dataset.\n')
try:
    # Install and upgrade opendatalab
    subprocess.check_call(['pip', 'install', 'opendatalab'],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
    print('Installed opendatalab.\n')
    subprocess.check_call(['pip', 'install', '-U', 'opendatalab'],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
    print('Upgraded opendatalab.\n')
except subprocess.CalledProcessError as e:
    print(f'Error: {e}')

print('Please input your username and password to login to OpenDataLab.')
# Login to OpenDataLab (Please manually enter your credentials when asked)
subprocess.check_call(['odl', 'login'])

# Check if the file exists
file_path = 'CamVid/raw/CamVid.tar.gz.00'
if not os.path.exists(file_path):
    # Download the dataset
    subprocess.check_call(['odl', 'get', 'CamVid'])

# Unzip dataset
with tarfile.open(file_path, 'r:gz') as tar:
    tar.extractall('data')
    print('Extracted dataset to /data.\n')

# Delete the CamVid folder
shutil.rmtree('CamVid')
```

## Building Camvid Dataset

First, we define a function `create_color_to_class_mapping` and dataset class called `CamVid`, which inherits from VisionDataset. in this class, we override the `__getitem__` and `__len__` methods to ensure that each index returns a tuple of images and the corresponding class-indexed masks. In addition, we create the color_to_class dictionary to map the color of the mask to the class index.

```python
import numpy as np
from torchvision.datasets import VisionDataset

def create_color_to_class_mapping(csv_filepath):
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
        self.color_to_class = create_color_to_class_mapping(
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
        mask_class = np.zeros_like(mask, dtype=np.int64)
        for color, class_index in self.color_to_class.items():
            rgb = color[0] * 65536 + color[1] * 256 + color[2]
            mask_class[mask == rgb] = class_index

        if self.target_transform is not None:
            mask_class = self.target_transform(mask_class)

        return img, mask_class

    def __len__(self):
        return len(self.images)
```

Then we use the Camvid dataset to create the `train_dataloader` and `val_dataloader`, the data loaders for training and validation. We will use them in the subsequent Runner.

```python
import os
import torch
from mmengine.runner import Runner

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

## Build a Segmentation Model

This code defines a model class named MMDeeplabV3. The MMDeeplabV3 model class inherits from [BaseModel](mmengine.model.BaseModel) and uses the segmentation model of the DeepLabV3 architecture, while rewriting the forward propagation method to handle images and labels, and also supports calculating losses and returning predictions in both training and prediction modes.
More details about BaseModel, refer to [Model tutorial](../tutorials/model.md).

```python
from mmengine.model import BaseModel
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights

class MMDeeplabV3(BaseModel):

    def __init__(self, num_classes, device):
        super().__init__()
        self.deeplab = deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
            num_classes=21)
        self.deeplab.classifier[4] = torch.nn.Conv2d(
            256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.device = device
        self.deeplab = self.deeplab.to(self.device)  # Move model to device

    def forward(self, imgs, labels, mode):
        imgs = imgs.to(self.device)  # Move images to device
        labels = labels.to(self.device)  # Move labels to device
        x = self.deeplab(imgs)['out']
        labels = labels.squeeze(
            1)  # Make sure to remove the second dimension of labels.
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels, ignore_index=255)}
        elif mode == 'predict':
            return x, labels
```

## Training with Runner

In the validation process of model training, we choose to use `Intersection over Union` (IoU) to evaluate the segmentation model performance.

```python
class IoU(BaseMetric):

    def process(self, data_batch, data_samples):
        score, gt = data_samples
        pred = torch.argmax(score, dim=1)
        gt = gt.where(gt != 255, torch.tensor(-1, device=gt.device))
        intersection = (pred * gt).sum(dim=(1, 2)).float()
        union = (pred + gt).sum(dim=(1, 2)).float() - intersection
        iou = (intersection / union).cpu()
        self.results.append({'batch_size': len(gt), 'iou': iou})

    def compute_metrics(self, results):
        total_iou = sum(item['iou'].sum() for item in results)
        total_size = sum(item['batch_size'] for item in results)
        return dict(iou=100 * total_iou / total_size)
```

The following code demonstrates how to use Runner for model training.
More details about Runner, please refer to the [Runner tutorial](../tutorials/runner.md).

```python
num_classes = 32  # Modify to actual number of categories.
norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

runner = Runner(
    model=MMDeeplabV3(num_classes, device),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.01, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=IoU),
    launcher=args.launcher,
    cfg=dict(
        model_wrapper='MMDistributedDataParallel',
        find_unused_parameters=True),
)
runner.train()
```

Till now, we have completed an example of training a segmentation model. Here are the results of the model after two rounds of epoch. By adjusting the parameters with epoch, the model will work better.

![Result](https://raw.githubusercontent.com/W-ZN/Images/main/Blog/NLPResult.png)
