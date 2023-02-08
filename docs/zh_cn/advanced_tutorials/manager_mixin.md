# 全局管理器（ManagerMixin）

Runner 在训练过程中，难免会使用全局变量来共享信息，例如我们会在 model 中获取全局的 [logger](mmengine.logging.MMLogger) 来打印初始化信息；在 model 中获取全局的 [Visualizer](./visualization.md) 来可视化预测结果、特征图；在 [Registry](../advanced_tutorials/registry.md) 中获取全局的 [DefaultScope](mmengine.registry.DefaultScope) 来确定注册域。为了管理这些功能相似的模块，MMEngine 设计了管理器 [ManagerMix](mmengine.utils.ManagerMixin) 来统一全局变量的创建和获取方式。

![ManagerMixin](https://user-images.githubusercontent.com/57566630/163429552-3c901fc3-9cc1-4b71-82b6-d051f452a538.png)

## 接口介绍

- get_instance(name='', \*\*kwargs)：创建或者返回对应名字的的实例。
- get_current_instance()：返回最近被创建的实例。
- instance_name：获取对应实例的 name。

## 使用方法

1. 定义有全局访问需求的类

```python
from mmengine.utils import ManagerMixin


class GlobalClass(ManagerMixin):
    def __init__(self, name, value):
        super().__init__(name)
        self.value = value
```

注意全局类的构造函数必须带有 name 参数，并在构造函数中调用 `super().__init__(name)`，以确保后续能够根据 name 来获取对应的实例。

2. 在任意位置实例化该对象，以 Hook 为例（要确保访问该实例时，对象已经被创建）：

```python
from mmengine import Hook

class CustomHook(Hook):
    def before_run(self, runner):
        GlobalClass.get_instance('mmengine', value=50)
        GlobalClass.get_instance(runner.experiment_name, value=100)
```

当我们调用子类的 `get_instance` 接口时，`ManagerMixin` 会根据名字来判断对应实例是否已经存在，进而创建/获取实例。如上例所示，当我们第一次调用  `GlobalClass.get_instance('mmengine', value=50)` 时，会创建一个名为 "mmengine" 的 `GlobalClass` 实例，其初始 value 为 50。为了方便后续介绍 `get_current_instance` 接口，这里我们创建了两个 `GlobalClass` 实例。

3. 在任意组件中访问该实例

```python
import torch.nn as nn


class CustomModule(nn.Module):
    def forward(self, x):
        value = GlobalClass.get_current_instance().value  # 最近一次被创建的实例 value 为 100（步骤二中按顺序创建）
        value = GlobalClass.get_instance('mmengine').value  # 名为 mmengine 的实例 value 为 50
        # value = GlobalClass.get_instance('mmengine', 1000).value  # mmengine 已经被创建，不能再接受额外参数
```

在同一进程里，我们可以在不同组件中访问 `GlobalClass` 实例。例如我们在 `CustomModule` 中，调用 `get_instance`  和 `get_current_instance` 接口来获取对应名字的实例和最近被创建的实例。需要注意的是，由于 "mmengine"  实例已经被创建，再次调用时不能再传入额外参数，否则会报错。
