#  modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py  # noqa: E501
import argparse
import copy
import logging
from dataclasses import dataclass
from functools import partial
from typing import Dict, Sequence

import torch
import transformers
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from mmengine import load, print_log
from mmengine._strategy import FSDPStrategy
from mmengine.dataset import DefaultSampler
from mmengine.dist.utils import is_main_process
from mmengine.optim.scheduler import CosineAnnealingLR
from mmengine.utils import apply_to
from mmengine.visualization import Visualizer, WandbVisBackend

IGNORE_INDEX = -100
ORI_BATCH_SIZE = 8
PROMPT_DICT = {
    'prompt_input':
    ('Below is an instruction that describes a task, paired with an input '
     'that provides further context. '
     'Write a response that appropriately completes the request.\n\n'
     '### Instruction:\n{instruction}\n\n'
     '### Input:\n{input}\n\n### Response:'),
    'prompt_no_input':
    ('Below is an instruction that describes a task. '
     'Write a response that appropriately completes the request.\n\n'
     '### Instruction:\n{instruction}\n\n### Response:'),
}


def smart_tokenizer_and_embedding_resize(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size
    not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors='pt',
            padding='longest',
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = (
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources))
    input_ids = examples_tokenized['input_ids']
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized['input_ids_lens']):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """""Dataset for supervised fine-tuning.""."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        print_log('Loading data...', level=logging.WARNING)
        list_data_dict = load(data_path)

        print_log('Formatting inputs...', level=logging.WARNING)
        prompt_input, prompt_no_input = \
            PROMPT_DICT['prompt_input'], PROMPT_DICT['prompt_no_input']
        sources = [
            prompt_input.format_map(example) if example.get('input', '') != ''
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [
            f'{example["output"]}{tokenizer.eos_token}'
            for example in list_data_dict
        ]

        print_log(
            'Tokenizing inputs... This may take some time...',
            level=logging.WARNING)
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict['input_ids']
        self.labels = data_dict['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ('input_ids', 'labels'))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(  # type: ignore
            self.tokenizer.pad_token_id)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument('data_root', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--output-dir', type=str, default='work_dirs')
    parser.add_argument('--max-epoch', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    return args


def train():
    args = parse_args()
    strategy = FSDPStrategy(
        model_wrapper=dict(
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={LlamaDecoderLayer}), ),
        state_dict_cfg='full')
    visualizer = Visualizer(
        name='mmengine',
        save_dir=args.output_dir,
        vis_backends=[dict(type=WandbVisBackend)])

    tokenizer = LlamaTokenizer.from_pretrained(args.checkpoint)
    model = LlamaForCausalLM.from_pretrained(args.checkpoint)
    model.train()

    smart_tokenizer_and_embedding_resize(
        tokenizer=tokenizer,
        model=model,
    )

    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=args.data_root)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=DefaultSampler(dataset=train_dataset),
        collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer))
    epoch_length = len(train_dataloader)
    max_iters = epoch_length * args.max_epoch
    optim_cfg = dict(
        optimizer=dict(
            type=AdamW, lr=5e-5, betas=(0.9, 0.95), eps=1e-5,
            weight_decay=0.1),
        accumulative_counts=ORI_BATCH_SIZE / args.batch_size)
    scheduler_cfgs = [
        dict(
            type=CosineAnnealingLR,
            T_max=max_iters * args.batch_size / ORI_BATCH_SIZE,
            eta_min_ratio=0.1),
    ]

    model, optimizer, schedulers = strategy.prepare(
        model,
        optim_wrapper=optim_cfg,
        param_scheduler=scheduler_cfgs,
        dispatch_kwargs=dict(max_iters=max_iters, max_epochs=args.max_epoch))

    for epoch in range(args.max_epoch):
        for idx, inputs in enumerate(train_dataloader):
            cur_iter = epoch * epoch_length + idx
            # Convert inputs to target device.
            inputs = apply_to(inputs, lambda m: isinstance(m, torch.Tensor),
                              lambda m: m.cuda())

            loss = model(**inputs)['loss'].mean()
            optimizer.update_params(loss)

            # Keep the lr update frequency the same as the original batch size.
            if cur_iter * args.batch_size % ORI_BATCH_SIZE == 0:
                for scheduler in schedulers:
                    scheduler.step()

            max_memory = torch.cuda.max_memory_allocated()
            strategy.logger.info(f'Epoch: {epoch}/{args.max_epoch}, '
                                 f'Iter: {idx}/{epoch_length}, '
                                 f'Loss: {loss.item():.3f}, '
                                 f'Lr: {optimizer.get_lr()["lr"][0]:.6f} '
                                 f'Memory: {max_memory/1e9:.3f}G')
            visualizer.add_scalars({'loss': loss.item()})
            torch.cuda.reset_peak_memory_stats()
        save_dir = f'{args.output_dir}/epoch_{epoch}'
        state_dict = model.state_dict()

        if is_main_process():
            model.save_pretrained(
                save_dir,
                state_dict=state_dict,
            )
            tokenizer.save_pretrained(save_dir)


if __name__ == '__main__':
    train()
