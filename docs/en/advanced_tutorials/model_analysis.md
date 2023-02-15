# Model Complexity Analysis

We provide a tool to help with the complexity analysis for the network. We borrow the idea from the implementation of [fvcore](https://github.com/facebookresearch/fvcore) to build this tool, and plan to support more custom operators in the future. Currently, it provides the interfaces to compute "parameter", "activation" and "flops" of the given model, and supports printing the related information layer-by-layer in terms of network structure or table. The analysis tool provides both operator-level and module-level flop counts simultaneously. Please refer to [Flop Count](https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md) for implementation details of how to accurately measure the flops of one operator if interested.

## What's FLOPs

Flop is not a well-defined metric in complexity analysis, we follow [detectron2](https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#fvcore.nn.FlopCountAnalysis) to use one fused multiple-add as one flop.

## What's Activation

Activation is used to measure the feature quantity produced from one layer.

For example, given the inputs with shape `inputs = torch.randn((1, 3, 10, 10))`, and one linear layer with `conv = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=1)`.

We get the `output` with shape `(1, 10, 10, 10)` after feeding the `inputs` into `conv`. The activation quantity of `output` of this `conv` layer is `1000=10*10*10`

Let's start with the following examples.

## Usage Example 1: Model built with native nn.Module

### Code

```python
import torch
from torch import nn
from mmengine.analysis import get_model_complexity_info
# return a dict of analysis results, including:
# ['flops', 'flops_str', 'activations', 'activations_str', 'params', 'params_str', 'out_table', 'out_arch']

class InnerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10,10)
        self.fc2 = nn.Linear(10,10)
    def forward(self, x):
        return self.fc1(self.fc2(x))


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10,10)
        self.fc2 = nn.Linear(10,10)
        self.inner = InnerNet()
    def forward(self, x):
        return self.fc1(self.fc2(self.inner(x)))

input_shape = (1, 10)
model = TestNet()

analysis_results = get_model_complexity_info(model, input_shape)

print(analysis_results['out_table'])
print(analysis_results['out_arch'])

print("Model Flops:{}".format(analysis_results['flops_str']))
print("Model Parameters:{}".format(analysis_results['params_str']))
```

### Description of Results

The return outputs is dict, which contains the following keys:

- `flops`: number of total flops, e.g., 10000, 10000
- `flops_str`: with formatted string, e.g., 1.0G, 100M
- `params`: number of total parameters, e.g., 10000, 10000
- `params_str`: with formatted string, e.g., 1.0G, 100M
- `activations`: number of total activations, e.g., 10000, 10000
- `activations_str`: with formatted string, e.g., 1.0G, 100M
- `out_table`: print related information by table

```
| model                            | #parameters or shape| #flops    |
|:---------------------------------|:--------------------|:----------|
| model                            | 34.6M               | 65.7G     |
|  s1                              |  15.4K              |  4.32G    |
|   s1.pathway0_stem               |   9.54K             |   1.23G   |
|    s1.pathway0_stem.conv         |    9.41K            |    1.23G  |
|    s1.pathway0_stem.bn           |    0.128K           |           |
|   s1.pathway1_stem               |   5.9K              |   3.08G   |
|    s1.pathway1_stem.conv         |    5.88K            |    3.08G  |
|    s1.pathway1_stem.bn           |    16               |           |
|  s1_fuse                         |  0.928K             |  29.4M    |
|   s1_fuse.conv_f2s               |   0.896K            |   29.4M   |
|    s1_fuse.conv_f2s.weight       |    (16, 8, 7, 1, 1) |           |
|   s1_fuse.bn                     |   32                |           |
|    s1_fuse.bn.weight             |    (16,)            |           |
|    s1_fuse.bn.bias               |    (16,)            |           |
|  s2                              |  0.226M             |  7.73G    |
|   s2.pathway0_res0               |   80.1K             |   2.58G   |
|    s2.pathway0_res0.branch1      |    20.5K            |    0.671G |
|    s2.pathway0_res0.branch1_bn   |    0.512K           |           |
|    s2.pathway0_res0.branch2      |    59.1K            |    1.91G  |
|   s2.pathway0_res1.branch2       |   70.4K             |   2.28G   |
|    s2.pathway0_res1.branch2.a    |    16.4K            |    0.537G |
|    s2.pathway0_res1.branch2.a_bn |    0.128K           |           |
|    s2.pathway0_res1.branch2.b    |    36.9K            |    1.21G  |
|    s2.pathway0_res1.branch2.b_bn |    0.128K           |           |
|    s2.pathway0_res1.branch2.c    |    16.4K            |    0.537G |
|    s2.pathway0_res1.branch2.c_bn |    0.512K           |           |
|   s2.pathway0_res2.branch2       |   70.4K             |   2.28G   |
|    s2.pathway0_res2.branch2.a    |    16.4K            |    0.537G |
|    s2.pathway0_res2.branch2.a_bn |    0.128K           |           |
|    s2.pathway0_res2.branch2.b    |    36.9K            |    1.21G  |
|    s2.pathway0_res2.branch2.b_bn |    0.128K           |           |
|    s2.pathway0_res2.branch2.c    |    16.4K            |    0.537G |
|    s2.pathway0_res2.branch2.c_bn |    0.512K           |           |
|    ............................. |    ......           |    ...... |
```

- `out_arch`:  print related information by network layers

```bash
    TestNet(
    #params: 0.44K, #flops: 0.4K
    (fc1): Linear(
    in_features=10, out_features=10, bias=True
    #params: 0.11K, #flops: 100
    )
    (fc2): Linear(
    in_features=10, out_features=10, bias=True
    #params: 0.11K, #flops: 100
    )
    (inner): InnerNet(
    #params: 0.22K, #flops: 0.2K
    (fc1): Linear(
        in_features=10, out_features=10, bias=True
        #params: 0.11K, #flops: 100
    )
    (fc2): Linear(
        in_features=10, out_features=10, bias=True
        #params: 0.11K, #flops: 100
    )
    )
)
```

## Usage Example 2: Model built with mmengine

### Code

```python
import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel
from mmengine.analysis import get_model_complexity_info


class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels=None, mode='extract_feat'):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels
        elif mode == 'extract_feat':
            return x


input_shape = (3, 224, 224)
model = MMResNet50()

analysis_results = get_model_complexity_info(model, input_shape)


print("Model Flops:{}".format(analysis_results['flops_str']))
print("Model Parameters:{}".format(analysis_results['params_str']))
```

### Output

```bash
Model Flops:4.145G
Model Parameters:25.557M
```

## Interface

We provide more options to support custom output

- `model`: (nn.Module) the model to be analyzed
- `input_shape`: (tuple) the shape of the input, e.g., (3, 224, 224)
- `inputs`: (optional: torch.Tensor), if given, `input_shape` will be ignored
- `show_table`: (bool) whether return the statistics in the form of table, default: True
- `show_arch`: (bool) whether return the statistics in the form of table,  default: True
