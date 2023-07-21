# Train Llama2 in MMEngine

## Setup env

Note: This example requires PyTorch 2.0+ and MMEngine 0.8.0+.

- Install MMEngine

  ```bash
  git clone https://github.com/open-mmlab/mmengine.git
  cd mmengine
  pip install -e . -v
  ```

- Install third-party dependencies

  ```bash
  pip install -U transformers accelerate tokenizers
  ```

## Prepare data

```bash
mkdir data
wget https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json -O data/alpaca_data.json
```

## Prepare model

Download model weights from https://huggingface.co/meta-llama/Llama-2-7b-hf

## Train

```bash
torchrun --nproc-per-node 8 examples/llama2/fsdp_finetune.py data/alpaca_data.json ${model_weights}
```

## Inference

```bash
python examples/llama2/generate.py ${checkpoints}
```
