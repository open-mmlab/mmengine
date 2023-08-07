import numpy as np
import torch
from datasets import load_dataset
from torchtext.data.metrics import bleu_score
from transformers import AutoTokenizer, T5ForConditionalGeneration

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner

tokenizer = AutoTokenizer.from_pretrained('t5-small')


class MMT5ForTranslation(BaseModel):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, label, input_ids, attention_mask, mode):
        if mode == 'loss':
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label)
            return {'loss': output.loss}
        elif mode == 'predict':
            output = self.model.generate(input_ids)
            return output, label


def post_process(preds, labels):
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.split() for pred in preds]
    decoded_labels = [[label.split()] for label in labels]
    return decoded_preds, decoded_labels


class Accuracy(BaseMetric):

    def process(self, data_batch, data_samples):
        outputs, labels = data_samples
        decoded_preds, decoded_labels = post_process(outputs, labels)
        score = bleu_score(decoded_preds, decoded_labels)
        prediction_lens = torch.tensor([
            torch.count_nonzero(pred != tokenizer.pad_token_id)
            for pred in outputs
        ],
                                       dtype=torch.float64)

        gen_len = torch.mean(prediction_lens).item()
        self.results.append({
            'gen_len': gen_len,
            'bleu': score,
        })

    def compute_metrics(self, results):
        return dict(
            gen_len=np.mean([item['gen_len'] for item in results]),
            bleu_score=np.mean([item['bleu'] for item in results]),
        )


def collate_fn(data):
    prefix = 'translate English to French: '
    input_sequences = [prefix + item['translation']['en'] for item in data]
    target_sequences = [item['translation']['fr'] for item in data]
    input_dict = tokenizer(
        input_sequences,
        padding='longest',
        return_tensors='pt',
    )

    label = tokenizer(
        target_sequences,
        padding='longest',
        return_tensors='pt',
    ).input_ids
    label[label ==
          tokenizer.pad_token_id] = -100  # ignore contribution to loss
    return dict(
        label=label,
        input_ids=input_dict.input_ids,
        attention_mask=input_dict.attention_mask)


def main():
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    books = load_dataset('opus_books', 'en-fr')
    books = books['train'].train_test_split(test_size=0.2)
    train_set, test_set = books['train'], books['test']

    train_loader = dict(
        batch_size=16,
        dataset=train_set,
        sampler=dict(type='DefaultSampler', shuffle=True),
        collate_fn=collate_fn)
    test_loader = dict(
        batch_size=32,
        dataset=test_set,
        sampler=dict(type='DefaultSampler', shuffle=False),
        collate_fn=collate_fn)
    runner = Runner(
        model=MMT5ForTranslation(model),
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        optim_wrapper=dict(optimizer=dict(type=torch.optim.Adam, lr=2e-5)),
        train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
        val_cfg=dict(),
        work_dir='t5_work_dir',
        val_evaluator=dict(type=Accuracy))

    runner.train()


if __name__ == '__main__':
    main()
