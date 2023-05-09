# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from torch.optim import SGD
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import \
    MaskRCNN_ResNet50_FPN_V2_Weights

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner


class MMMaskRCNN(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights)

    def forward(self, images, targets, mode):
        if mode == 'loss':
            loss_dict = self.model(images, targets)
            return loss_dict
        elif mode == 'predict':
            predictions = self.model(images)
            return predictions, targets


class Accuracy(BaseMetric):

    def __init__(self, iou_threshold=0.5):
        super().__init__()
        self.iou_threshold = iou_threshold

    def process(self, data_batch, data_samples):
        predictions, targets = data_samples
        batch_size = targets.size(0)

        # Compute IoU for each prediction and target
        ious = self.calculate_iou(predictions, targets)

        # Count correct predictions based on IoU threshold
        correct = (ious > self.iou_threshold).sum().item()

        self.results.append({'batch_size': batch_size, 'correct': correct})

    def calculate_iou(self, predictions, targets):
        # Compute the intersection and union areas
        intersection = torch.min(predictions[:, 2:], targets[:, 2:]).clamp(0)
        # union = torch.max(predictions[:, 2:], targets[:, 2:]).clamp(0)

        # Calculate areas
        pred_area = (predictions[:, 2] - predictions[:, 0]) * (
            predictions[:, 3] - predictions[:, 1])
        target_area = (targets[:, 2] - targets[:, 0]) * (
            targets[:, 3] - targets[:, 1])

        # Calculate IoU
        iou = intersection.prod(dim=1) / (
            pred_area + target_area - intersection.prod(dim=1) + 1e-6)

        return iou

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        accuracy = 100 * total_correct / total_size
        return {'accuracy': accuracy}


class COCODataset(torch.utils.data.Dataset):

    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # COCO annotation
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        print(f'img_id: {img_id}')
        # List of annotation IDs for the image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Annotation for the image
        coco_annotation = coco.loadAnns(ann_ids)

        # Path of the input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # Open the input image
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        # Bounding boxes for objects
        boxes = []
        # Labels for objects
        labels = []
        # Areas of bounding boxes
        areas = []
        for obj in coco_annotation:
            # In COCO format, bbox = [xmin, ymin, width, height]
            # Convert it to [xmin, ymin, xmax, ymax]
            xmin = obj['bbox'][0]
            ymin = obj['bbox'][1]
            xmax = xmin + obj['bbox'][2]
            ymax = ymin + obj['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

            # The labels in the label_map start from 1, so subtract 1 from the
            # category ID
            labels.append(obj['category_id'] - 1)

            areas.append(obj['area'])

        # Convert the lists to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((len(coco_annotation), ), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation['boxes'] = boxes
        my_annotation['labels'] = labels
        my_annotation['image_id'] = torch.tensor([img_id])
        my_annotation['area'] = areas
        my_annotation['iscrowd'] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--batch_size', type=int, default=32, help='batch size for training')

    args = parser.parse_args()
    return args


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def main():
    args = parse_args()
    norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
    train_dataset = torchvision.datasets.CocoDetection(
        root=r'data/COCO128/train',
        annFile=r'data/COCO128/train/_annotations.coco.json',
        transforms=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg)
        ]))
    val_dataset = torchvision.datasets.CocoDetection(
        root=r'data/COCO128/valid',
        annFile=r'data/COCO128/valid/_annotations.coco.json',
        transforms=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg)
        ]))

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn='default_collate')
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn='default_collate')
    runner = Runner(
        model=MMMaskRCNN(),
        work_dir='./work_dir',
        train_dataloader=train_dataloader,
        optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
        train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        val_evaluator=dict(type=Accuracy),
        launcher=args.launcher,
    )
    runner.train()


if __name__ == '__main__':
    main()
