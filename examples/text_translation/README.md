# Train a Text Translation Model

## Install Dependencies

- Build MMEngine from Source

  ```bash
  git clone https://github.com/open-mmlab/mmengine.git
  cd mmengine
  pip install -e . -v
  ```

- Install thirty-party libraries

  ```bash
  pip install datasets transformers torchtext
  ```

## Run the Example

- Single device training

  ```bash
  python examples/text_translation/train.py
  ```

- Distributed data parallel training

  ```bash
  tochrun -nnodes 1 -nproc_per_node 8 examples/text_translation/train.py --launcher pytorch
  ```
