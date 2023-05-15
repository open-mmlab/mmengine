# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_convert

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner


class MMMaskRCNN(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn(pretrained=True, num_classes=91)

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
        batch_size = len(targets)

        # Compute IoU for each prediction and target
        iou_scores = self.calculate_iou(predictions, targets)
        print(f'iou: {iou_scores}')

        # Count correct predictions based on IoU threshold
        correct = 0
        for i in range(batch_size):
            if i < len(iou_scores):  # Check if there are valid IoU scores
                iou = iou_scores[i]
                num_correct = np.sum(iou > self.iou_threshold)
                correct += num_correct

        self.results.append({'batch_size': batch_size, 'correct': correct})

    def calculate_iou(self, predictions, targets):
        iou_scores = []

        for i in range(len(predictions)):
            box1 = predictions[i]['boxes'].cpu().numpy()
            box2 = targets[i]['boxes'].cpu().numpy()

            if len(box1) > 0 and len(box2) > 0:
                num_box1 = len(box1)
                num_box2 = len(box2)
                iou_matrix = torch.zeros((num_box1, num_box2))

                # Calculate IoU for each pair of boxes
                for j in range(num_box1):
                    for k in range(num_box2):
                        iou = self.compute_iou(box1[j], box2[k])
                        iou_matrix[j, k] = iou

                # Find the best matching target box for each predicted box
                for j in range(num_box1):
                    best_iou = torch.max(iou_matrix[j])
                    iou_scores.append(best_iou.item())

        return iou_scores

    def compute_iou(self, box1, box2):
        # Calculate the coordinates of the intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate the areas of intersection and union
        intersection_area = torch.max(torch.tensor(0.0), x2 - x1) * torch.max(
            torch.tensor(0.0), y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        # Calculate the IoU
        iou = intersection_area / union_area

        return iou

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        accuracy = 100 * total_correct / total_size
        return {'accuracy': accuracy}


def collate_fn(batches):
    # Remove None values from the batch
    # Separate images and targets
    images = [batch[0] for batch in batches]
    targets = [batch[1] for batch in batches]
    # print(targets[5])
    new_targets = []

    for target in targets:
        if len(target) > 0:
            imageIds = [item['image_id'] for item in target]
            bboxes = [item['bbox'] for item in target]
            bboxes = torch.tensor(bboxes)
            bboxes = box_convert(bboxes, in_fmt='xywh', out_fmt='xyxy')
            ids = [item['id'] for item in target]
            categoryIds = [item['category_id'] for item in target]
            areas = [item['area'] for item in target]
            new_target = {
                'id': torch.tensor(ids),
                'image_id': torch.tensor(imageIds),
                'category_id': torch.tensor(categoryIds, dtype=torch.int64),
                'boxes': bboxes,
                'area': torch.tensor(areas),
                'labels': torch.tensor(categoryIds).to(torch.int64)
            }
            new_targets.append(new_target)
            # print(f"bboxes: {bboxes}")
        else:
            new_target = {
                'id': torch.tensor([]),
                'image_id': torch.tensor(0),
                'category_id': torch.tensor([]),
                'boxes': torch.empty((0, 4)),
                'area': torch.tensor([]),
                'labels': torch.tensor([]).to(torch.int64)
            }
            new_targets.append(new_target)
    # print(f"TARGETS: {len(new_targets)}")

    return images, new_targets


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


def main():
    args = parse_args()
    norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
    folder = '/content/mmengine/examples'
    train_dataset = torchvision.datasets.CocoDetection(
        root=f'{folder}/train',
        annFile=f'{folder}/train/_annotations.coco.json',
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg)
        ]))
    val_dataset = torchvision.datasets.CocoDetection(
        root=f'{folder}/valid',
        annFile=f'{folder}/valid/_annotations.coco.json',
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(**norm_cfg)]))
    train_dataloader = dict(
        dataset=train_dataset,
        batch_size=16,
        sampler=dict(type='DefaultSampler', shuffle=False),
        collate_fn=collate_fn)
    val_dataloader = dict(
        dataset=val_dataset,
        batch_size=16,
        sampler=dict(type='DefaultSampler', shuffle=False),
        collate_fn=collate_fn)

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
