import argparse
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.registry import DATASETS, METRICS, MODELS
from mmengine.tune import find_optimial_lr


class ToyModel(BaseModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor)
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if isinstance(data_samples, list):
            data_samples = torch.stack(data_samples)
        outputs = self.linear1(inputs)
        outputs = self.linear2(outputs)

        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (data_samples - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        elif mode == 'predict':
            return outputs


class ToyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_samples=self.label[index])


class ToyMetric(BaseMetric):

    def __init__(self, collect_device='cpu', dummy_metrics=None):
        super().__init__(collect_device=collect_device)
        self.dummy_metrics = dummy_metrics

    def process(self, data_batch, predictions):
        result = {'acc': 1}
        self.results.append(result)

    def compute_metrics(self, results):
        return dict(acc=1)


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

    MODELS.register_module(module=ToyModel, force=True)
    METRICS.register_module(module=ToyMetric, force=True)
    DATASETS.register_module(module=ToyDataset, force=True)

    temp_dir = tempfile.TemporaryDirectory()

    runner_cfg = dict(
        work_dir=temp_dir.name,
        model=dict(type='ToyModel'),
        train_dataloader=dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='DefaultSampler', shuffle=True),
            batch_size=3,
            num_workers=0),
        val_dataloader=dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='DefaultSampler', shuffle=False),
            batch_size=3,
            num_workers=0),
        val_evaluator=[dict(type='ToyMetric')],
        test_dataloader=dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='DefaultSampler', shuffle=False),
            batch_size=3,
            num_workers=0),
        test_evaluator=[dict(type='ToyMetric')],
        optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.1)),
        train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
        val_cfg=dict(),
        test_cfg=dict(),
        launcher=args.launcher,
        default_hooks=dict(logger=dict(type='LoggerHook', interval=1)),
        custom_hooks=[],
        env_cfg=dict(dist_cfg=dict(backend='nccl')),
        experiment_name='test1')

    temp_dir.cleanup()

    best_lr, lowest_loss = find_optimial_lr(
        runner_cfg=runner_cfg,
        num_trials=32,
        tunning_epoch=1,
    )
    print('best_lr: ', best_lr)
    print('lowest_loss: ', lowest_loss)


if __name__ == '__main__':
    main()
