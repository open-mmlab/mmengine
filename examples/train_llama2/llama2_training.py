#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import argparse
import copy
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from mmengine import load, print_log
from mmengine._strategy import FSDPStrategy
from mmengine.dataset import DefaultSampler
from mmengine.optim import AmpOptimWrapper
from mmengine.optim.scheduler import LinearLR
from mmengine.utils import apply_to

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="sgd")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )



def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
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
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        print_log("Loading data...", level=logging.WARNING)
        list_data_dict = load(data_path)

        print_log("Formatting inputs...", level=logging.WARNING)
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        print_log("Tokenizing inputs... This may take some time...", level=logging.WARNING)
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )



# def parse_args():
#     parser = argparse.ArgumentParser(description='Distributed Training')
#     parser.add_argument('data_root', type=str)
#     parser.add_argument('checkpoint', type=str)
#     parser.add_argument('--max-epoch', type=int, default=3)

#     args = parser.parse_args()
#     return args
    
def train():
    # args = parse_args()
    strategy = FSDPStrategy(
        model_wrapper=dict(
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy, transformer_layer_cls=set([LlamaDecoderLayer])
            ),
        ),
    )

    tokenizer = LlamaTokenizer.from_pretrained('/nvme/data/yehaochen/checkpoints/llama-2-13b-hf/')
    model = LlamaForCausalLM.from_pretrained('/nvme/data/yehaochen/checkpoints/llama-2-13b-hf/')

    # tokenizer = LlamaTokenizer.from_pretrained('/nvme/data/yehaochen/checkpoints/llama-2-13b-hf/')
    # model = LlamaForCausalLM(LlamaConfig())
    model.train()
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    # TODO replace it with AdamW
    optim_cfg = dict(
        optimizer=dict(type=AdamW, lr=5e-5),
        # type=AmpOptimWrapper,
        # cache_enabled=False,
    )
    model, optimizer = strategy.prepare(model, optim_wrapper=optim_cfg)
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path='/home/yehaochen/codebase/stanford_alpaca/alpaca_data.json')
    collate_fn = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset, collate_fn=collate_fn,
        sampler=DefaultSampler(dataset=train_dataset),
        batch_size=2)

    strategy.logger.info(f'memory cost for model weights: {torch.cuda.max_memory_allocated()/1e9:.3f}G')
    for epoch in range(3):
        for idx, inputs in enumerate(train_dataloader):
            inputs = apply_to(
                inputs, lambda m: isinstance(m, torch.Tensor), lambda m: m.cuda())
            with optimizer.optim_context(model):
                loss = model(**inputs)['loss'].mean()
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            max_memory = torch.cuda.max_memory_allocated()
            strategy.logger.info(f'Epoch: {epoch}, Iter: {idx}, Loss: {loss.item()}, memory: {max_memory/1e9:.3f}G')
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    train()
