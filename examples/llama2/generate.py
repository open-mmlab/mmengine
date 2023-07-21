import argparse

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

# flake8: noqa

prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Imagine you are from the 1700s. Try to write a sentence in the language used in that era.

### Response:"""


def parse_args():
    parser = argparse.ArgumentParser(description='llama2 inference')
    parser.add_argument('checkpoint', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = LlamaForCausalLM.from_pretrained(args.checkpoint).half().cuda()
    model.eval()

    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        generate_ids = model.generate(inputs.input_ids.cuda(), max_length=300)
    print(
        tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0])
