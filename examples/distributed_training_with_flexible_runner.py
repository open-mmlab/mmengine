# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

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


class MMViT(BaseModel):

    def __init__(self):
        super().__init__()
        self.vit = torchvision.models.vit_h_16()

    def forward(self, imgs, labels, mode):
        x = self.vit(imgs)
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
    parser.add_argument('--model', default='reset', choices=['reset', 'vit'])
    parser.add_argument('--use-fsdp', action='store_true')
    parser.add_argument('--use-deepspeed', action='store_true')
    parser.add_argument(
        '--dummy-data', action='store_true', help='use fake data to benchmark')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.model == 'reset':
        model = MMResNet50()
        if args.dummy_data:
            train_set = torchvision.datasets.FakeData(50000, (3, 32, 32), 10,
                                                      transforms.ToTensor())
            valid_set = torchvision.datasets.FakeData(10000, (3, 32, 32), 10,
                                                      transforms.ToTensor())
        else:
            norm_cfg = dict(
                mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
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
    else:
        model = MMViT()
        if args.dummy_data:
            train_set = torchvision.datasets.FakeData(1281167, (3, 224, 224),
                                                      1000,
                                                      transforms.ToTensor())
            valid_set = torchvision.datasets.FakeData(50000, (3, 224, 224),
                                                      1000,
                                                      transforms.ToTensor())
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            train_set = torchvision.datasets.ImageFolder(
                'data/imagenet/train',
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            valid_set = torchvision.datasets.ImageFolder(
                'data/imagenet/val',
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
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

    if args.use_deepspeed:
        strategy = dict(
            type='DeepSpeedStrategy',
            fp16=dict(
                enabled=True,
                fp16_master_weights_and_grads=False,
                loss_scale=0,
                loss_scale_window=500,
                hysteresis=2,
                min_loss_scale=1,
                initial_scale_power=15,
            ),
            inputs_to_half=[0],
            zero_optimization=dict(
                stage=3,
                allgather_partitions=True,
                reduce_scatter=True,
                allgather_bucket_size=50000000,
                reduce_bucket_size=50000000,
                # stage3_max_live_parameters=1e9,
                # stage3_max_reuse_distance=1e9,
                # stage3_prefetch_bucket_size=5e8,
                # stage3_param_persistence_threshold=1e6,
                overlap_comm=True,
                contiguous_gradients=True,
                cpu_offload=False),
        )
        optim_wrapper = dict(
            type='DeepSpeedOptimWrapper',
            optimizer=dict(type='AdamW', lr=1e-3))
    elif args.use_fsdp:
        from functools import partial

        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        size_based_auto_wrap_policy = partial(
            size_based_auto_wrap_policy, min_num_params=1e7)
        strategy = dict(
            type='FSDPStrategy',
            model_wrapper=dict(auto_wrap_policy=size_based_auto_wrap_policy))
        optim_wrapper = dict(
            type='AmpOptimWrapper', optimizer=dict(type='AdamW', lr=1e-3))
    else:
        strategy = None
        optim_wrapper = dict(
            type='AmpOptimWrapper', optimizer=dict(type='AdamW', lr=1e-3))

    runner = FlexibleRunner(
        model=model,
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
    # torchrun --nproc-per-node 2 distributed_training_with_flexible_runner.py --use-fsdp  # noqa: 501
    # torchrun --nproc-per-node 2 distributed_training_with_flexible_runner.py --use-deepspeed  # noqa: 501
    # torchrun --nproc-per-node 2 distributed_training_with_flexible_runner.py
    # python distributed_training_with_flexible_runner.py
    main()
