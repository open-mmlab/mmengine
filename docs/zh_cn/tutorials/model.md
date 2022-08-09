# 模型

基于 Pytorch 构建模型时，我们会选择 [nn.Module](https://pytorch.org/docs/stable/nn.html?highlight=nn%20module#module-torch.nn.modules) 作为模型的基类，让模型能够轻松实现：

- 导出/加载/遍历模型参数，将模型参数转移至指定设备，设置模型的训练、测试状态等功能。
- `nn.Module` 能够将参数导出后传给接优化器[optimizer](https://pytorch.org/docs/stable/optim.html?highlight=optimizer#torch.optim.Optimizer)实现自动化的参数更新。
- 对接 [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html?highlight=distributeddataparallel#torch.nn.parallel.DistributedDataParallel)
  实现分布式训练。
- `nn.Mdoule` 能够将参数导出后，对接参数初始化模块 [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html?highlight=kaiming#torch.nn.init.kaiming_normal_)，轻松实现指定策略的参数初始化。

等一系列功能。正如上面提到的，`nn.Module` 对接参数初始化模块、优化器实现参数初始化、自动化的参数更新已经成为了使用 `nn.Module`
的标准流程，因此 `MMEngine` 在 `nn.Module` 的基础上做了进一步的抽象出了 `BaseModule` 和
`BaseModel`，前者用于配置模型初始化策略，后者定义了管理模型训练、验证、测试、推理的基本流程。

## 基本模块（`BaseModule`）

MMEngine 抽象出基本模块来配置模型初始化相关的参数。基本模块继承自 `nn.Module`，不仅具备 `nn.Module`
的基本功能，还能根据传参实现相应的参数初始化逻辑。我们可以让模型继承 `BaseModule`，通过配置 `init_cfg`
实现自定义的参数初始化逻辑。

### 加载预训练权重

```python
import torch
import torch.nn as nn

from mmengine.model import BaseModule


class ToyNet(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__()
        self.conv1 = nn.Linear(1, 1)

# 保存预训练权重
toy_net = ToyNet()
torch.save(toy_net.state_dict(), './pretrained.pth')
pretrained = './pretrained.pth'

# 配置加载预训练权重的初始化方式
toy_net = ToyNet(init_cfg=dict(
    type='Pretrained', checkpoint=pretrained, prefix='backbone'))
# 加载权重
toy_net.init_weights()
```

如上例所示，我们通过配置 `init_cfg` 让 `toy_net` 在调用 `init_weights`
时加载预训练权重。`pretrained` 的值不仅可以是本地磁盘路径，也可以是 url，如：

```python
pretrained = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_3rdparty-mill_in21k_20220331-faac000b.pth'  # noqa
toy_net = ToyNet(init_cfg=dict(
    type='Pretrained', checkpoint=pretrained, prefix='backbone'))
```
