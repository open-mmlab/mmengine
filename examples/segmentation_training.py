import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.optim import SGD
from torchvision.datasets import VisionDataset
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import \
    DeepLabV3_ResNet50_Weights

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner


def create_color_to_class_mapping(csv_filepath):
    color_to_class = {}
    with open(csv_filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            r, g, b = int(row['r']), int(row['g']), int(row['b'])
            color_to_class[(r, g, b)] = idx
    return color_to_class


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


def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    return args


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


def main():
    args = parse_args()
    num_classes = 32  # Modify to actual number of categories.
    norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    if torch.cuda.is_available(
    ):  # Added this line to clear GPU cache before each training iteration
        torch.cuda.empty_cache()
    runner.train()


if __name__ == '__main__':
    main()
