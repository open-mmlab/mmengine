# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
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
    parser.add_argument('--use-fsdp', action='store_true')
    parser.add_argument('--use-deepspeed', action='store_true')
    parser.add_argument('--use-colossalai', action='store_true')
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
            # bf16=dict(
            #     enabled=True,
            # ),
            zero_optimization=dict(
                stage=3,
                allgather_partitions=True,
                reduce_scatter=True,
                allgather_bucket_size=50000000,
                reduce_bucket_size=50000000,
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
    elif args.use_colossalai:
        from colossalai.tensor.op_wrapper import colo_op_impl

        # ColossalAI overwrite some torch ops with their custom op to
        # make it compatible with `ColoTensor`. However, a backward error
        # is more likely to happen if there are inplace operation in the
        # model.
        # For example, layers like `conv` + `bn` + `relu` is OK when `relu` is
        # inplace since PyTorch builtin ops `batch_norm` could handle it.
        # However, if `relu` is an `inplaced` op while `batch_norm` is an
        # custom op, an error will be raised since PyTorch thinks the custom op
        # could not handle the backward graph modification caused by inplace
        # op.
        # In this example, the inplace op `add_` in resnet could raise an error
        # since PyTorch consider the custom op before it could not handle the
        # backward graph modification
        colo_op_impl(torch.Tensor.add_)(torch.add)
        strategy = dict(type='ColossalAIStrategy')
        optim_wrapper = dict(optimizer=dict(type='HybridAdam', lr=1e-3))
    else:
        strategy = None
        optim_wrapper = dict(
            type='AmpOptimWrapper', optimizer=dict(type='AdamW', lr=1e-3))

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
    # torchrun --nproc-per-node 2 distributed_training_with_flexible_runner.py --use-fsdp  # noqa: 501
    # torchrun --nproc-per-node 2 distributed_training_with_flexible_runner.py --use-deepspeed  # noqa: 501
    # torchrun --nproc-per-node 2 distributed_training_with_flexible_runner.py
    # python distributed_training_with_flexible_runner.py
    main()
