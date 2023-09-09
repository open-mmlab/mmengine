#!/bin/bash

# module load anaconda/2020.11
# source activate mmengine
# python -m torch.distributed.launch --nproc_per_node=1 ./examples/distributed_training_demo.py --launcher pytorch

python ./examples/distributed_training_demo.py --launcher none
