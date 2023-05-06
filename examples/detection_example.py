# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD
from torchvision.models.detection import maskrcnn_resnet50_fpn

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner


class MMMaskRCNN(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = maskrcnn_resnet50_fpn(pretrained=True)

    def forward(self, images, targets, mode):
        if mode == 'loss':
            loss_dict = self.model(images, targets)
            return loss_dict
        elif mode == 'predict':
            predictions = self.model(images)
            return predictions, targets


class Accuracy(BaseMetric):

    def process(self, data_batch, data_samples):
        predictions, labels = data_samples
        correct = (predictions.argmax(dim=1) == labels).sum().item()
        batch_size = labels.size(0)
        self.results.append({'batch_size': batch_size, 'correct': correct})

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


def main():
    args = parse_args()
    norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
    train_set = torchvision.datasets.Kitti(
        'data/kitti',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg)
        ]))
    valid_set = torchvision.datasets.Kitti(
        'data/kitti',
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(**norm_cfg)]))
    train_dataloader = dict(
        batch_size=args.batch_size,
        dataset=train_set,
        sampler=dict(type='DefaultSampler', shuffle=True),
        collate_fn=dict(type='default_collate'))
    val_dataloader = dict(
        batch_size=args.batch_size,
        dataset=valid_set,
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
