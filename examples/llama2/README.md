# Train Llama2 in MMEngine

## Setup env

```bash
git clone https://github.com/open-mmlab/mmengine.git
cd mmengine
pip install -e . -v
pip install -U transformers
```

## Prepare data

```bash
mkdir data
wget https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json -O data/alpaca_data.json
```

## Prepare model

Download model weights from https://huggingface.co/meta-llama/Llama-2-13b-hf

## Train

```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:500 torchrun --nproc-per-node 8 examples/llama2/fsdp_finetune.py  data/alpaca_data.json ${model_weights}
```

## Inference

```bash
python examples/llama2/generate.py ${checkpoints}
```
