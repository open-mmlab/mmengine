# 模型复杂度分析

我们提供了一个工具来帮助进行网络的复杂性分析。我们借鉴了 [fvcore](https://github.com/facebookresearch/fvcore) 的实现思路来构建这个工具，并计划在未来支持更多的自定义运算符。目前的工具提供了用于计算给定模型的 "parameter"、"activation" 和 "flops "的接口，并支持以网络结构或表格的形式逐层打印相关信息，同时提供operator层级和模块级的flop计数。如果您对如何准确测量一个 operator 的flop的实现细节感兴趣，请参考 [Flop Count](https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md)。

## 什么是 FLOPs

浮点运算数（FLOPs）在复杂性分析中不是一个定义非常明确的指标，我们按照 [detectron2](https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#fvcore.nn.FlopCountAnalysis)，使用 1 组乘-加运算作为 1 个 flop。

## 什么是 activation

激活量（activation）用于衡量某一层产生的特征数量。

例如，给定输入尺寸 `inputs = torch.randn((1, 3, 10, 10))`，和一个有3个输入通道、10个输出通道的线性层 `conv = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=1)`。

将 `inputs` 输入到 `conv`，得到的输出特征图的尺寸是 `(1, 10, 10, 10)`。则 `output` 对这个 `conv` 层的激活量就是 `1000=10*10*10`。

让我们从下面的示例开始上手。

## 用法示例1: 从原始的 nn.Module 构建的模型

### 代码

```python
import torch
from torch import nn
from mmengine.analysis import get_model_complexity_info
# 以字典的形式返回分析结果，包括:
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

### 结果描述

返回的输出是一个包含以下7个键的字典:

- `flops`: flop 的总数, e.g., 10000, 10000
- `flops_str`: 格式化的字符串, e.g., 1.0G, 100M
- `params`: 全部参数的数量, e.g., 10000, 10000
- `params_str`: 格式化的字符串, e.g., 1.0G, 100M
- `activations`: 激活量的总数, e.g., 10000, 10000
- `activations_str`: 格式化的字符串, e.g., 1.0G, 100M
- `out_table`: 以表格形式打印相关信息

```
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

- `out_arch`: 以网络层级结构打印相关信息

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

## 用法示例2: 通过 mmengine 构建的模型

### 代码

```python
import torch.nn.functional as F
im
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
print("Model Parameters:{}".format(analysis_results['params_str']))
```

### 输出

```bash
Model Flops:4.145G
Model Parameters:25.557M
```

## 接口

我们提供了更多的选项来支持自定义输出内容：

- `model`: (nn.Module) 待分析的模型
- `input_shape`: (tuple) 输入尺寸, e.g., (3, 224, 224)
- `inputs`: (optional: torch.Tensor), 如果传入该参数, `input_shape` 会被忽略
- `show_table`: (bool) 是否以表格形式返回统计结果，默认：True
- `show_arch`: (bool) 是否以网络结构形式返回统计结果，默认：True
