# 模型复杂度分析

我们提供了一个工具来帮助分析网络的复杂性。我们借鉴了 [fvcore](https://github.com/facebookresearch/fvcore) 的实现思路来构建这个工具，并计划在未来支持更多的自定义算子。目前的工具提供了用于计算给定模型的浮点运算量（FLOPs）、激活量（Activations）和参数量（Parameters）的接口，并支持以网络结构或表格的形式逐层打印相关信息，同时提供了算子级别（operator）和模块级别（Module）的统计。如果您对统计浮点运算量的实现细节感兴趣，请参考 [Flop Count](https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md)。

## 定义

模型复杂度有 3 个指标，分别是浮点运算量（FLOPs）、激活量（Activations）以及参数量（Parameters），它们的定义如下：

- 浮点运算量

  浮点运算量不是一个定义非常明确的指标，在这里参考 [detectron2](https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#fvcore.nn.FlopCountAnalysis) 的描述，将一组乘加运算定义为 1 个 flop。

- 激活量

  激活量用于衡量某一层产生的特征数量。

- 参数量

  模型的参数量。

例如，给定输入尺寸 `inputs = torch.randn((1, 3, 10, 10))`，和一个卷积层 `conv = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)`，那么它输出的特征图尺寸为 `(1, 10, 8, 8)`，则它的浮点运算量是 `17280 = 10*8*8*3*3*3`（10*8*8 表示输出的特征图大小、3*3*3 表示每一个输出需要的计算量）、激活量是 `640 = 10*8*8`、参数量是 `280 = 3*10*3*3 + 10`（3*10*3\*3 表示权重的尺寸、10 表示偏置值的尺寸）。

## 用法

### 基于 `nn.Module` 构建的模型

构建模型

```python
from torch import nn

from mmengine.analysis import get_model_complexity_info


# 以字典的形式返回分析结果，包括:
# ['flops', 'flops_str', 'activations', 'activations_str', 'params', 'params_str', 'out_table', 'out_arch']
class InnerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc1(self.fc2(x))


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.inner = InnerNet()

    def forward(self, x):
        return self.fc1(self.fc2(self.inner(x)))


input_shape = (1, 10)
model = TestNet()
```

`get_model_complexity_info` 返回的 `analysis_results` 是一个包含 7 个值的字典:

- `flops`: flop 的总数, 例如, 1000, 1000000
- `flops_str`: 格式化的字符串, 例如, 1.0G, 1.0M
- `params`: 全部参数的数量, 例如, 1000, 1000000
- `params_str`: 格式化的字符串, 例如, 1.0K, 1M
- `activations`: 激活量的总数, 例如, 1000, 1000000
- `activations_str`: 格式化的字符串, 例如, 1.0G, 1M
- `out_table`: 以表格形式打印相关信息

打印结果

- 以表格形式打印相关信息

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

- 以网络层级结构打印相关信息

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

- 以字符串的形式打印结果

  ```python
  print("Model Flops:{}".format(analysis_results['flops_str']))
  # Model Flops:0.4K
  print("Model Parameters:{}".format(analysis_results['params_str']))
  # Model Parameters:0.44K
  ```

### 基于 BaseModel（来自 MMEngine）构建的模型

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

## 其他接口

除了上述基本用法，`get_model_complexity_info` 还能接受以下参数，输出定制化的统计结果：

- `model`: (nn.Module) 待分析的模型
- `input_shape`: (tuple) 输入尺寸，例如 (3, 224, 224)
- `inputs`: (optional: torch.Tensor), 如果传入该参数, `input_shape` 会被忽略
- `show_table`: (bool) 是否以表格形式返回统计结果，默认值：True
- `show_arch`: (bool) 是否以网络结构形式返回统计结果，默认值：True
