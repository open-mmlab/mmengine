# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD

from mmengine.dist import launch
from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner


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
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--num-gpus',
        nargs='+',
        type=int,
        help='number of gpus to use',
        default=[2])
    parser.add_argument(
        '--num-machine', type=int, help='number of machines to use', default=1)
    parser.add_argument(
        '--machine-rank',
        type=int,
        help='The rank of current machine.',
        default=0)
    parser.add_argument(
        '--master-addr',
        type=str,
        help='The FQDN of the host that is running worker with rank 0.',
        default='127.0.0.1')
    parser.add_argument(
        '--master-port',
        type=str,
        default='auto',
        help='The port on the ``master_addr``.',
    )
    args = parser.parse_args()
    return args


def main(args):
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
        batch_size=32,
        dataset=train_set,
        sampler=dict(type='DefaultSampler', shuffle=True),
        collate_fn=dict(type='default_collate'))
    val_dataloader = dict(
        batch_size=32,
        dataset=valid_set,
        sampler=dict(type='DefaultSampler', shuffle=False),
        collate_fn=dict(type='default_collate'))
    runner = Runner(
        model=MMResNet50(),
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
    args = parse_args()
    if args.num_gpus == [1] and args.num_machine == 1:
        assert args.launcher == 'none', (
            'launcher must be none for single GPU training')
        main(args)
    else:
        assert args.launcher != 'none', (
            'launcher must be set for multi-GPU training')

        launch(
            main,
            args.num_gpus,
            args.num_machine,
            args.machine_rank,
            args.master_addr,
            args.master_port,
            args=(args, ),
        )
