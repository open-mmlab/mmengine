# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from mmeval import COCODetection
from torch.optim import Adam
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_convert

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner

num_classes = 80


class MMMaskRCNN(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_feats = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_feats, num_classes)

    def forward(self, images, targets, mode):
        if mode == 'loss':
            output = self.model(images, targets)
            return output
        elif mode == 'predict':
            predictions = self.model(images)
            return predictions


class Accuracy(BaseMetric):

    def process(self, targets, predictions):
        batch_size = len(targets)
        fake_dataset_metas = {
            'classes': tuple([str(i) for i in range(num_classes)])
        }
        self.coco_det_metric = COCODetection(
            dataset_meta=fake_dataset_metas, metric=[
                'bbox',
            ])
        print(predictions)
        for i in range(batch_size):
            print(f'TARGETS: {targets[i]}')
            self.coco_det_metric(
                predictions=[
                    predictions[i],
                ], groundtruths=[
                    targets[i],
                ])
            self.results.append = {
                'batch_size': batch_size,
                'bbox_result': self.coco_det_metric['bbox_result']
            }
        # for i in range(batch_size):
        #   prediction = predictions[i]
        #   target = targets[i]
        #   print(f"TARGET: {target}")
        #   print(f"PREDICTIONS: {prediction}")
        # self.results.append({
        #     'batch_size': len(gt),
        #     'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        # })
        # print(f"data_batch: {data_batch}\n data_samples: {data_samples}\n")

    def compute_metrics(self, results):
        pass
        # print(f"RESULTS: {results}\n")
        # total_correct = sum(item['correct'] for item in results)
        # total_size = sum(item['batch_size'] for item in results)
        # return dict(accuracy=100 * total_correct / total_size)


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
    parser = argparse.ArgumentParser(description='Object Detection')
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
        optim_wrapper=dict(optimizer=dict(type=Adam, lr=0.0001)),
        train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        val_evaluator=dict(type=Accuracy),
        launcher=args.launcher,
    )
    runner.train()


if __name__ == '__main__':
    main()
