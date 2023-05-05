# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner


class MMBertForClassify(BaseModel):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, label, input_ids, token_type_ids, attention_mask, mode):
        output = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=label)
        if mode == 'loss':
            return {'loss': output.loss}
        elif mode == 'predict':
            return output.logits, label


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
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    return args


def collate_fn(data):
    labels = []
    input_ids = []
    token_type_ids = []
    attention_mask = []
    for item in data:
        labels.append(item['label'])
        input_ids.append(torch.tensor(item['input_ids']))
        token_type_ids.append(torch.tensor(item['token_type_ids']))
        attention_mask.append(torch.tensor(item['attention_mask']))

    input_ids = torch.stack(input_ids)
    token_type_ids = torch.stack(token_type_ids)
    attention_mask = torch.stack(attention_mask)
    label = torch.tensor(labels)
    return dict(
        label=label,
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask)


def main():
    args = parse_args()
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_set = load_dataset('imdb', split='train')
    test_set = load_dataset('imdb', split='test')

    train_set = train_set.map(
        lambda x: tokenizer(
            x['text'], truncation=True, padding=True, max_length=128),
        batched=True)
    test_set = test_set.map(
        lambda x: tokenizer(
            x['text'], truncation=True, padding=True, max_length=128),
        batched=True)

    train_loader = dict(
        batch_size=32,
        dataset=train_set,
        sampler=dict(type='DefaultSampler', shuffle=True),
        collate_fn=collate_fn)
    test_loader = dict(
        batch_size=32,
        dataset=test_set,
        sampler=dict(type='DefaultSampler', shuffle=False),
        collate_fn=collate_fn)
    runner = Runner(
        model=MMBertForClassify(model),
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        optim_wrapper=dict(optimizer=dict(type=torch.optim.Adam, lr=2e-5)),
        train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
        val_cfg=dict(),
        work_dir='bert_work_dir',
        val_evaluator=dict(type=Accuracy),
        launcher=args.launcher,
    )
    runner.train()


if __name__ == '__main__':
    main()
