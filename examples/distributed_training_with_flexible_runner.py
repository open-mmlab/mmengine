# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner._flexible_runner import FlexibleRunner


class MMResNet50(BaseModel):

    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels


class Accuracy(BaseMetric):

    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        return dict(accuracy=100 * total_correct / total_size)


def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--strategy', choices=['deepspeed', 'fsdp', 'none'], default='none')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
    train_set = torchvision.datasets.CIFAR10(
        'data/cifar10',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg)
        ]))
    valid_set = torchvision.datasets.CIFAR10(
        'data/cifar10',
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(**norm_cfg)]))
    train_dataloader = dict(
        batch_size=128,
        dataset=train_set,
        sampler=dict(type='DefaultSampler', shuffle=True),
        collate_fn=dict(type='default_collate'))
    val_dataloader = dict(
        batch_size=128,
        dataset=valid_set,
        sampler=dict(type='DefaultSampler', shuffle=False),
        collate_fn=dict(type='default_collate'))

    if args.strategy == 'deepspeed':
        strategy = dict(
            type='DeepSpeedStrategy',
            zero_optimization=dict(
                stage=0,
                allgather_partitions=True,
                reduce_scatter=True,
                allgather_bucket_size=50000000,
                reduce_bucket_size=50000000,
                overlap_comm=True,
                contiguous_gradients=True,
                cpu_offload=False))
        optim_wrapper = dict(
            type='DeepSpeedOptimWrapper',
            optimizer=dict(type=SGD, lr=0.001, momentum=0.9))
    elif args.strategy == 'fsdp':
        from functools import partial

        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        size_based_auto_wrap_policy = partial(
            size_based_auto_wrap_policy, min_num_params=1e7)
        strategy = dict(
            type='FSDPStrategy',
            model_wrapper=dict(auto_wrap_policy=size_based_auto_wrap_policy))
        optim_wrapper = dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9))
    else:
        strategy = None
        optim_wrapper = dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9))

    runner = FlexibleRunner(
        model=MMResNet50(),
        work_dir='./work_dirs',
        strategy=strategy,
        train_dataloader=train_dataloader,
        optim_wrapper=optim_wrapper,
        param_scheduler=dict(type='LinearLR'),
        train_cfg=dict(by_epoch=True, max_epochs=10, val_interval=1),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        val_evaluator=dict(type=Accuracy))
    runner.train()


if __name__ == '__main__':
    main()
