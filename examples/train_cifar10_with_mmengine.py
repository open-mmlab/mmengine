import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from mmengine import Config, Runner
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS, MODELS


@MODELS.register_module()
class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.criterion = nn.CrossEntropyLoss()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward_train(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_test(self, x):
        with torch.no_grad():
            x = self.forward_train(x)
        return x

    def forward(self, data_batch, return_loss=False):
        inputs, labels = zip(*data_batch)
        inputs = torch.stack(inputs).to(self.device)
        labels = torch.tensor(labels).to(self.device)

        if return_loss:
            outputs = self.forward_train(inputs)
            loss = self.criterion(outputs, labels)
            return {'loss': loss, 'log_vars': {'loss': loss.item()}}
        else:
            outputs = self.forward_test(inputs)
            predictions = torch.argmax(outputs, 1)
            return predictions


@METRICS.register_module()
class Accuracy(BaseMetric):
    default_prefix = 'ACC'

    def process(self, data_batch, predictions):
        result = {
            'gt': [data[1] for data in data_batch],
            'pred': predictions,
        }
        self.results.append(result)

    def compute_metrics(self, results):
        preds = np.concatenate([res['pred'] for res in results])
        gts = np.concatenate([res['gt'] for res in results])
        acc = (preds == gts).sum() / preds.size
        return {'accuracy': acc}


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    cfg = dict(
        model=dict(type='ToyModel'),
        work_dir='./work_dir',
        train_dataloader=dict(
            dataset=train_dataset,
            sampler=dict(type='DefaultSampler', shuffle=True),
            batch_size=64,
            num_workers=4),
        optimizer=dict(type='SGD', lr=0.001, momentum=0.9),
        param_scheduler=dict(type='ConstantLR', factor=1),
        train_cfg=dict(by_epoch=True, max_epochs=5),
        val_dataloader=dict(
            dataset=test_dataset,
            sampler=dict(type='DefaultSampler', shuffle=False),
            batch_size=64,
            num_workers=4),
        val_cfg=dict(interval=2),
        val_evaluator=dict(type='Accuracy'),
        test_dataloader=dict(
            dataset=test_dataset,
            sampler=dict(type='DefaultSampler', shuffle=False),
            batch_size=64,
            num_workers=4),
        test_cfg=dict(),
        test_evaluator=dict(type='Accuracy'),
        env_cfg=dict(dist_cfg=dict(backend='nccl')),
    )

    cfg = Config(cfg)
    runner = Runner.from_cfg(cfg)
    runner.train()
    runner.test()


if __name__ == '__main__':
    main()
