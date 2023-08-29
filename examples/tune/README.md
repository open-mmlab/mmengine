# Find the Optimal Learning Rate

## Install external dependencies

First, you should install `nevergrad` for tuning.

```bash
pip install nevergrad
```

## Run the example

Single device training

```bash
python examples/tune/find_lr.py
```

Distributed data parallel tuning

```bash
torchrun -nnodes 1 -nproc_per_node 8 examples/tune/find_lr.py --launcher pytorch
```
