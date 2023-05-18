import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.backends.cudnn.benchmark = True


class MMDeeplabV3(BaseModel):

    def __init__(self, num_classes):
        super().__init__()
        self.deeplab = torchvision.models.segmentation.deeplabv3_resnet50(
            pretrained=True, num_classes=num_classes)

    def forward(self, imgs, labels, mode):
        x = self.deeplab(imgs)['out']
        labels = labels.squeeze(
            1)  # Change the shape of labels from (B, 1, H, W) to (B, H, W).
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


def main():
    args = parse_args()
    num_classes = 21  # Modify to actual number of categories.
    norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_set = torchvision.datasets.VOCSegmentation(
        'data/VOC2012',
        image_set='train',
        download=True,
        transform=transforms.Compose([
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg)
        ]),
        target_transform=transforms.Compose([
            transforms.Resize((520, 520), interpolation=0),
            transforms.Lambda(
                lambda x: torch.tensor(np.array(x), dtype=torch.long))
        ]))
    valid_set = torchvision.datasets.VOCSegmentation(
        'data/VOC2012',
        image_set='val',
        download=True,
        transform=transforms.Compose([
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg)
        ]),
        target_transform=transforms.Compose([
            transforms.Resize((520, 520), interpolation=0),
            transforms.Lambda(
                lambda x: torch.tensor(np.array(x), dtype=torch.long))
        ]))
    train_dataloader = dict(
        batch_size=8,
        dataset=train_set,
        sampler=dict(type='DefaultSampler', shuffle=True),
        collate_fn=dict(type='default_collate'))
    val_dataloader = dict(
        batch_size=8,
        dataset=valid_set,
        sampler=dict(type='DefaultSampler', shuffle=False),
        collate_fn=dict(type='default_collate'))
    runner = Runner(
        model=MMDeeplabV3(num_classes),
        work_dir='./work_dir',
        train_dataloader=train_dataloader,
        optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.01, momentum=0.9)),
        train_cfg=dict(by_epoch=True, max_epochs=10, val_interval=1),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        val_evaluator=dict(type=IoU),
        launcher=args.launcher,
    )
    runner.train()


if __name__ == '__main__':
    main()
