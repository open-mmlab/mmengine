# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
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
            print(f'TARGET: {targets}')
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
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg)
        ]))
    val_dataset = torchvision.datasets.CocoDetection(
        root=r'data/COCO128/valid',
        annFile=r'data/COCO128/valid/_annotations.coco.json',
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(**norm_cfg)]))
    # print(train_dataset[0])
    train_dataloader = dict(
        dataset=train_dataset,
        batch_size=32,
        sampler=dict(type='DefaultSampler', shuffle=False),
        collate_fn=dict(type='default_collate'))
    val_dataloader = dict(
        dataset=val_dataset,
        batch_size=32,
        sampler=dict(type='DefaultSampler', shuffle=False),
        collate_fn=dict(type='default_collate'))

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
