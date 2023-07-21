#  modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py  # noqa: E501
import argparse
import copy
from functools import partial

import torch
import transformers
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.data import default_data_collator
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from mmengine import load
from mmengine._strategy import FSDPStrategy
from mmengine.dataset import DefaultSampler
from mmengine.dist.utils import is_main_process
from mmengine.optim.scheduler import StepLR
from mmengine.utils import apply_to
from mmengine.visualization import Visualizer, WandbVisBackend

IGNORE_INDEX = -100
ORI_BATCH_SIZE = 4
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
    num_new_tokens = tokenizer.add_special_tokens({'pad_token': '<PAD>'})
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


class AlpacaDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_words=224):
        self.ann = load(data_path)
        self.max_words = max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        if ann.get('input', '') == '':
            prompt = PROMPT_DICT['prompt_no_input'].format_map(ann)
        else:
            prompt = PROMPT_DICT['prompt_input'].format_map(ann)
        example = prompt + ann['output']
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat(
                (example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            'input_ids': example,
            'labels': labels,
            'attention_mask': example_mask,
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument('data_root', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--output-dir', type=str, default='work_dirs')
    parser.add_argument('--max-epoch', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--save-interval', type=int, default=500)
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

    train_dataset = AlpacaDataset(
        tokenizer=tokenizer, data_path=args.data_root)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=DefaultSampler(dataset=train_dataset),
        collate_fn=default_data_collator)
    epoch_length = len(train_dataloader)
    max_iters = epoch_length * args.max_epoch
    optim_cfg = dict(
        optimizer=dict(type=AdamW, lr=1e-4, weight_decay=0.0),
        accumulative_counts=ORI_BATCH_SIZE / args.batch_size)
    scheduler_cfgs = [
        dict(
            type=StepLR,
            step_size=1,
        ),
    ]

    model, optimizer, schedulers = strategy.prepare(
        model,
        optim_wrapper=optim_cfg,
        param_scheduler=scheduler_cfgs,
        dispatch_kwargs=dict(max_iters=max_iters, max_epochs=args.max_epoch))

    for epoch in range(args.max_epoch):
        for idx, inputs in enumerate(train_dataloader):
            cur_iter = epoch * epoch_length + idx + 1
            # Convert inputs to target device.
            inputs = apply_to(inputs, lambda m: isinstance(m, torch.Tensor),
                              lambda m: m.cuda())

            loss = model(**inputs)['loss'].mean()
            optimizer.update_params(loss)

            max_memory = torch.cuda.max_memory_allocated()
            strategy.logger.info(f'Epoch: {epoch+1}/{args.max_epoch}, '
                                 f'Iter: {idx+1}/{epoch_length}, '
                                 f'Loss: {loss.item():.3f}, '
                                 f'Lr: {optimizer.get_lr()["lr"][0]:.6f} '
                                 f'Memory: {max_memory/1e9:.3f}G')
            visualizer.add_scalars({'loss': loss.item()})
            torch.cuda.reset_peak_memory_stats()

            if cur_iter % args.save_interval == 0:
                save_dir = f'{args.output_dir}/epoch_{epoch}_{idx+1}'
                state_dict = model.state_dict()

                if is_main_process():
                    model.save_pretrained(
                        save_dir,
                        state_dict=state_dict,
                    )
                    tokenizer.save_pretrained(save_dir)

        for scheduler in schedulers:
            scheduler.step()


if __name__ == '__main__':
    train()
