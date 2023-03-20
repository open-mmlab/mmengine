# Model Complexity Analysis

We provide a tool to help with the complexity analysis for the network. We borrow the idea from the implementation of [fvcore](https://github.com/facebookresearch/fvcore) to build this tool, and plan to support more custom operators in the future. Currently, it provides the interfaces to compute "FLOPs", "Activations" and "Parameters",  of the given model, and supports printing the related information layer-by-layer in terms of network structure or table. The analysis tool provides both operator-level and module-level flop counts simultaneously. Please refer to [Flop Count](https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md) for implementation details of how to accurately measure the flops of one operator if interested.

## Definition

The model complexity has three indicators, namely floating-point operations (FLOPs), activations, and parameters. Their definitions are as follows:

- FLOPs

  Floating-point operations (FLOPs) is not a clearly defined indicator. Here, we refer to the description in  [detectron2](https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#fvcore.nn.FlopCountAnalysis), which defines a set of multiply-accumulate operations as 1 FLOP.

- Activations

  Activation is used to measure the feature quantity produced from one layer.

- Parameters

  The parameter count of a model.

For example, given an input size of `inputs = torch.randn((1, 3, 10, 10))` and a convolutional layer `conv = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)`, if the output feature map size is `(1, 10, 8, 8)`, then its FLOPs are `17280 = 10*8*8*3*3*3` (where `10*8*8` represents the output feature map size, and `3*3*3` represents the computation for each output), activations are `640 = 10*8*8`, and the parameter count is `280 = 3*10*3*3 + 10` (where `3*10*3*3` represents the size of weights, and 10 represents the size of bias).

## Usage

### Model built with native nn.Module

Build a model

```python
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
```

The `analysis_results` returned by `get_model_complexity_info` is a dict, which contains the following keys:

- `flops`: number of total flops, e.g., 10000, 10000
- `flops_str`: with formatted string, e.g., 1.0G, 100M
- `params`: number of total parameters, e.g., 10000, 10000
- `params_str`: with formatted string, e.g., 1.0G, 100M
- `activations`: number of total activations, e.g., 10000, 10000
- `activations_str`: with formatted string, e.g., 1.0G, 100M
- `out_table`: print related information by table

Print the results

- print related information by table

  ```python
  print(analysis_results['out_table'])
  ```

  ```text
  +---------------------+----------------------+--------+--------------+
  | module              | #parameters or shape | #flops | #activations |
  +---------------------+----------------------+--------+--------------+
  | model               | 0.44K                | 0.4K   | 40           |
  |  fc1                |  0.11K               |  100   |  10          |
  |   fc1.weight        |   (10, 10)           |        |              |
  |   fc1.bias          |   (10,)              |        |              |
  |  fc2                |  0.11K               |  100   |  10          |
  |   fc2.weight        |   (10, 10)           |        |              |
  |   fc2.bias          |   (10,)              |        |              |
  |  inner              |  0.22K               |  0.2K  |  20          |
  |   inner.fc1         |   0.11K              |   100  |   10         |
  |    inner.fc1.weight |    (10, 10)          |        |              |
  |    inner.fc1.bias   |    (10,)             |        |              |
  |   inner.fc2         |   0.11K              |   100  |   10         |
  |    inner.fc2.weight |    (10, 10)          |        |              |
  |    inner.fc2.bias   |    (10,)             |        |              |
  +---------------------+----------------------+--------+--------------+
  ```

- print related information by network layers

  ```python
  print(analysis_results['out_arch'])
  ```

  ```bash
  TestNet(
    #params: 0.44K, #flops: 0.4K, #acts: 40
    (fc1): Linear(
      in_features=10, out_features=10, bias=True
      #params: 0.11K, #flops: 100, #acts: 10
    )
    (fc2): Linear(
      in_features=10, out_features=10, bias=True
      #params: 0.11K, #flops: 100, #acts: 10
    )
    (inner): InnerNet(
      #params: 0.22K, #flops: 0.2K, #acts: 20
      (fc1): Linear(
        in_features=10, out_features=10, bias=True
        #params: 0.11K, #flops: 100, #acts: 10
      )
      (fc2): Linear(
        in_features=10, out_features=10, bias=True
        #params: 0.11K, #flops: 100, #acts: 10
      )
    )
  )
  ```

- print results with formatted string

  ```python
  print("Model Flops:{}".format(analysis_results['flops_str']))
  # Model Flops:0.4K
  print("Model Parameters:{}".format(analysis_results['params_str']))
  # Model Parameters:0.44K
  ```

### Model built with mmengine

```python
import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel
from mmengine.analysis import get_model_complexity_info


class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels=None, mode='tensor'):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels
        elif mode == 'tensor':
            return x


input_shape = (3, 224, 224)
model = MMResNet50()

analysis_results = get_model_complexity_info(model, input_shape)

print("Model Flops:{}".format(analysis_results['flops_str']))
# Model Flops:4.145G
print("Model Parameters:{}".format(analysis_results['params_str']))
# Model Parameters:25.557M
```

## Interface

We provide more options to support custom output

- `model`: (nn.Module) the model to be analyzed
- `input_shape`: (tuple) the shape of the input, e.g., (3, 224, 224)
- `inputs`: (optional: torch.Tensor), if given, `input_shape` will be ignored
- `show_table`: (bool) whether return the statistics in the form of table, default: True
- `show_arch`: (bool) whether return the statistics by network layers,  default: True
