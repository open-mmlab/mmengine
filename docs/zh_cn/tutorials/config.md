# 配置（Config）

MMEngine 实现了抽象的配置类，为用户提供统一的配置访问接口。配置类能够支持不同格式的配置文件，包括 `python`，`json`，`yaml`，用户可以根据需求选择自己偏好的格式。配置类提供了类似字典或者 Python 对象属性的访问接口，用户可以十分自然地进行配置字段的读取和修改。为了方便算法框架管理配置文件，配置类也实现了一些特性，例如配置文件的字段继承等。

## 配置文件读取

配置类提供了统一的接口 `Config.fromfile()`，来读取和解析配置文件。

合法的配置文件应该定义一系列键值对，这里举几个不同格式配置文件的例子。

Python 格式：

```Python
test_int = 1
test_list = [1, 2, 3]
test_dict = dict(key1='value1', key2=0.1)
```

Json 格式：

```json
{
  "test_int": 1,
  "test_list": [1, 2, 3],
  "test_dict": {"key1": "value1", "key2": 0.1}
}
```

YAML 格式：

```yaml
test_int: 1
test_list: [1, 2, 3]
test_dict:
  key1: "value1"
  key2: 0.1
```

对于以上三种格式的文件，假设文件名分别为 `config.py`，`config.json`，`config.yml`，则我们调用 `Config.fromfile('config.xxx')` 接口都会得到相同的结果，构造了包含 3 个字段的配置对象。我们以 `config.py` 为例：

```python
from mmengine import Config

cfg = Config.fromfile('/path/to/config.py')
# Config (path: config.py): {'test_int': 1, 'test_list': [1, 2, 3], 'test_dict': {'key1: 'value1', "key2": 0.1}
```

尽管配置类能够解析不同格式的配置文件，但是 `python`，`yaml`，`json` 格式的配置文件支持的功能有所不同。

- `python` 和 `yaml` 格式的配置文件支持解析元组（tuple）与集合（set），而 `json`
  格式的配置文件只能解析列表（list），或将元组与集合解析成列表。

  ```python
  from mmengine import Config

  cfg_python = Config(dict(a=tuple((1, 2, 3))))
  cfg_python.dump('a.yaml')
  cfg_yaml = cfg_python.fromfile('a.yaml')
  cfg_yaml._cfg_dict == cfg_python._cfg_dict  # True

  cfg_python.dump('a.json')
  cfg_json = cfg_python.fromfile('a.json')
  cfg_json._cfg_dict == cfg_python._cfg_dict  # False, a of cfg_json is a list
  ```

- `python` 格式的配置文件支持修改 _base_ 文件中的变量，如 `_base_.a=1`，而 `yaml` 和 `json`
  格式的配置文件不支持

## 配置文件的使用

通过读取配置文件来初始化配置对象后，就可以像使用普通字典或者 Python 类一样来使用这个变量了。
我们提供了两种访问接口，即类似字典的接口 `cfg['key']` 或者类似 Python 对象属性的接口 `cfg.key`。这两种接口都支持读写。

```python
cfg = Config.fromfile('config.py')

cfg.test_int  # 1
cfg.test_list  # [1, 2, 3]
cfg.test_dict  # ConfigDict(key1='value1', key2=0.1)
cfg.test_int = 2  # 这里发生了配置字段修改，test_int 字段的值变成了 2

cfg['test_int']  # 2
cfg['test_list']  # [1, 2, 3]
cfg['test_dict']  # ConfigDict(key1='value1', key2=0.1)
cfg['test_list'][1] = 3  # 这里发生了字段修改，test_list 字段的值变成了 [1, 3, 3]
```

注意，配置文件中定义的嵌套字段（即类似字典的字段），在 Config 中会将其转化为 ConfigDict 类，该类具有和 Python 内置字典类型相同的接口，可以直接当做普通字典使用。

在算法库中，可以将配置与注册器结合起来使用，达到通过配置文件来控制模块构造的目的。这里举一个在配置文件中定义优化器的例子。

假设我们已经定义了一个优化器的注册器 OPTIMIZERS，包括了各种优化器。那么首先写一个 `config_sgd.py`：

```python
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
```

然后在算法库中可以通过如下代码构造优化器对象。

```python
from mmengine import Config
from mmengine.Registry import OPTIMIZERS

cfg = Config.fromfile('config_sgd.py')
optimizer = OPTIMIZERS.build(cfg.optimizer)
# 这里 optimizer 就是一个 torch.optim.SGD 对象
```

这样，我们就可以在不改动算法库代码的情况下，仅通过修改配置文件，来使用不同的优化器。

## 配置文件的继承

有时候，两个不同的配置文件之间的差异很小，可能仅仅只改了一个字段，我们就需要将所有内容复制粘贴一次，而且在后续观察的时候，不容易定位到具体差异的字段。又有些情况下，多个配置文件可能都有相同的一批字段，我们不得不在这些配置文件中进行复制粘贴，给后续的修改和维护带来了不便。

为了解决这些问题，我们给配置文件增加了继承的机制，即一个配置文件 A 可以将另一个配置文件 B 作为自己的基础，直接继承了 B 中所有字段，而不必显式复制粘贴。

### 继承机制概述

这里我们举一个例子来说明继承机制。定义如下两个配置文件，

`optimizer_cfg.py`：

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

`resnet50.py`：

```python
_base_ = ['optimizer_cfg.py']
model = dict(type='ResNet', depth=50)
```

虽然我们在 `resnet50.py` 中没有定义 optimizer 字段，但由于我们写了 `_base_ = ['optimizer_cfg.py']`，会使这个配置文件获得 `optimizer_cfg.py` 中的所有字段。

```python
cfg = Config.fromfile('resnet50.py')
cfg.optimizer  # ConfigDict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

这里 `_base_` 是配置文件的保留字段，指定了该配置文件的继承来源。支持继承多个文件，将同时获得这多个文件中的所有字段，但是要求继承的多个文件中**没有**相同名称的字段，否则会报错。

`runtime_cfg.py`：

```python
gpu_ids = [0, 1]
```

`resnet50.py`：

```python
_base_ = ['optimizer_cfg.py', 'runtime_cfg.py']
model = dict(type='ResNet', depth=50)
```

这时，读取配置文件 `resnet50.py` 会获得 3 个字段 `model`，`optimizer`，`gpu_ids`。

通过这种方式，我们可以将配置文件进行拆分，定义一些通用配置文件，在实际配置文件中继承各种通用配置文件，可以减少具体任务的配置流程。

### 修改继承字段

有时候，我们继承一个配置文件之后，可能需要对其中个别字段进行修改，例如继承了 `optimizer_cfg.py` 之后，想将学习率从 0.02 修改为 0.01。

这时候，只需要在新的配置文件中，重新定义一下需要修改的字段即可。注意由于 optimizer 这个字段是一个字典，我们只需要重新定义这个字典里面需修改的下级字段即可。这个规则也适用于增加一些下级字段。

`resnet50_lr0.01.py`：

```python
_base_ = ['optimizer_cfg.py', 'runtime_cfg.py']
model = dict(type='ResNet', depth=50)
optimizer = dict(lr=0.01)
```

读取这个配置文件之后，就可以得到期望的结果。

```python
cfg = Config.fromfile('resnet50_lr0.01.py')
cfg.optimizer  # ConfigDict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
```

对于非字典类型的字段，例如整数，字符串，列表等，重新定义即可完全覆盖，例如下面的写法就将 `gpu_ids` 这个字段的值修改成了 `[0]`。

```python
_base_ = ['optimizer_cfg.py', 'runtime_cfg.py']
model = dict(type='ResNet', depth=50)
gpu_ids = [0]
```

### 删除字典中的 key

有时候我们对于继承过来的字典类型字段，不仅仅是想修改其中某些 key，可能还需要删除其中的一些 key。这时候在重新定义这个字典时，需要指定 `_delete_=True`，表示将没有在新定义的字典中出现的 key 全部删除。

`resnet50.py`：

```python
_base_ = ['optimizer_cfg.py', 'runtime_cfg.py']
model = dict(type='ResNet', depth=50)
optimizer = dict(_delete_=True, type='SGD', lr=0.01)
```

这时候，`optimizer` 这个字典中就只有 `type` 和 `lr` 这两个 key，`momentum` 和 `weight_decay` 将不再被继承。

```python
cfg = Config.fromfile('resnet50_lr0.01.py')
cfg.optimizer  # ConfigDict(type='SGD', lr=0.01)
```

### 引用被继承文件中的变量

有时我们想重复利用 `_base_` 中定义的字段内容，就可以通过 `{{_base_.xxxx}}` 获取来获取对应变量的拷贝。例如：

```python
_base_ = ['resnet50.py']
a = {{_base_.model}}
# 等价于 a = dict(type='ResNet', depth=50)
```

## 配置文件的导出

在启动训练脚本时，用户可能通过传参的方式来修改配置文件的部分字段，为此我们提供了 `dump`
接口来导出更改后的配置文件。与读取配置文件类似，用户可以通过 `cfg.dump('config.xxx')` 来选择导出文件的格式。`dump`
同样可以导出有继承关系的配置文件，导出的文件可以被独立使用，不再依赖于 `_base_` 中定义的文件。

基于继承一节定义的 `resnet50.py`

```python
_base_ = ['optimizer_cfg.py', 'runtime_cfg.py']
model = dict(type='ResNet', depth=50)
```

我们将其加载后导出:

```python
cfg = Config.fromfile('resnet50.py')
cfg.dump('resnet50_dump.py')
```

`dumped_resnet50.py`

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
gpu_ids = [0, 1]
model = dict(type='ResNet', depth=50)
```

类似的，我们可以导出 json、yaml 格式的配置文件

`dumped_resnet50.yaml`

```yaml
gpu_ids:
- 0
- 1
model:
  depth: 50
  type: ResNet
optimizer:
  lr: 0.02
  momentum: 0.9
  type: SGD
  weight_decay: 0.0001
```

`dumped_resnet50.json`

```json
{"optimizer": {"type": "SGD", "lr": 0.02, "momentum": 0.9, "weight_decay": 0.0001}, "gpu_ids": [0, 1], "model": {"type": "ResNet", "depth": 50}}
```

此外，`dump` 不仅能导出加载自文件的 `cfg`，还能导出加载自字典的 `cfg`

```python
cfg = Config(dict(a=1, b=2))
cfg.dump('demo.py')
```

`demo.py`

```python
a=1
b=2
```

## 其他进阶用法

这里介绍一下配置类的进阶用法，这些小技巧可能使用户开发和使用算法库更简单方便。

### 预定义字段

有时候我们希望配置文件中的一些字段和当前路径或者文件名等相关，这里举一个典型使用场景的例子。在训练模型时，我们会在配置文件中定义一个工作目录，存放这组实验配置的模型和日志，那么对于不同的配置文件，我们期望定义不同的工作目录。用户的一种常见选择是，直接使用配置文件名作为工作目录名的一部分，例如对于配置文件 `config_setting1.py`，工作目录就是 `./work_dir/config_setting1`。

使用预定义字段可以方便地实现这种需求，在配置文件 `config_setting1.py` 中可以这样写：

```Python
work_dir = './work_dir/{{ fileBasenameNoExtension }}'
```

这里 `{{ fileBasenameNoExtension }}` 表示该配置文件的文件名（不含拓展名），在配置类读取配置文件的时候，会将这种用双花括号包起来的字符串自动解析为对应的实际值。

```Python
cfg = Config.fromfile('./config_setting1.py')
cfg.work_dir  # "./work_dir/config_setting1"
```

目前支持的预定义字段有以下四种，变量名参考自 [VS Code](https://code.visualstudio.com/docs/editor/variables-reference) 中的相关字段：

- `{{ fileDirname }}` - 当前文件的目录名，例如 `/home/your-username/your-project/folder`
- `{{ fileBasename }}` - 当前文件的文件名，例如 `file.py`
- `{{ fileBasenameNoExtension }}` - 当前文件不包含扩展名的文件名，例如 file
- `{{ fileExtname }}` - 当前文件的扩展名，例如 `.py`

### 跨项目继承配置文件

为了避免基于已有算法库开发的新项目复制大量的配置文件，MMEngine 中的 config 支持配置文件的跨项目继承。
假定 MMDetection 项目中存在如下配置文件

```text
configs/_base_/schedules/schedule_1x.py
configs/_base_/datasets.coco_instance.py
configs/_base_/default_runtime.py
configs/_base_/models/faster_rcnn_r50_fpn.py
```

在 MMDetection 被安装进环境（如使用 `pip install mmdet`）以后，新的项目可以直接在自己的配置文件中继承 MMDetection 的配置文件而无需拷贝，使用方式如下所示

```python
_base_ = [
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/datasets.coco_instance.py',
    'mmdet::_base_/default_runtime.py'
    'mmdet::_base_/models/faster_rcnn_r50_fpn.py',
]
```

通过指定 `mmdet::` ，Config 类会去检索 mmdet 包中的配置文件目录，并继承指定的配置文件。
实际上，只要算法库的 `setup.py` 文件符合 [MMEngine 安装规范](todo)，在正确安装算法库以后，新的项目就可以使用上述用法去继承已有算法库的配置文件而无需拷贝。

### 跨项目使用配置文件

MMEngine 还提供了 `get_config` 和 `get_model` 两个接口，支持对符合 [MMEngine 安装规范](todo) 的算法库中的模型和配置文件做索引并进行 API 调用。通过 `get_model` 接口可以获得构建好的模型。通过 `get_config` 接口可以获得配置文件。

`get_model` 的使用样例如下所示，使用和跨项目继承配置文件相同的语法，指定 `mmdet::`，即可在 mmdet 包中检索对应的配置文件并构建和初始化相应模型。
用户可以通过指定 `pretrained=True` 获得已经加载预训练权重的模型以进行训练或者推理。

```python
from mmengine import get_model
model = get_model('mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco', pretrained=True)
```

`get_config` 的使用样例如下所示，使用和跨项目继承配置文件相同的语法，指定 `mmdet::`，即可实现去 mmdet 包中检索并加载对应的配置文件。
用户可以基于这样得到的配置文件进行推理修改并自定义自己的算法模型。
同时，如果用户指定 `pretrained=True` ，得到的配置文件中会新增 `model_path` 字段，指定了对应模型预训练权重的路径。

```python
from mmengine import get_config
cfg = get_config('mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco', pretrained=True)
model_path = cfg.model_path

from mmdet.models import build_model
model = build_model(cfg.model)
load_checkpoint(model, model_path)
```

### 导入自定义 Python 模块

将配置与注册器结合起来使用时，如果我们往注册器中注册了一些自定义的类，就可能会遇到一些问题。因为读取配置文件的时候，这部分代码可能还没有被执行到，所以并未完成注册过程，从而导致构建自定义类的时候报错。

例如我们新实现了一种优化器 `SuperOptim`，相应代码在 my_package/my_module.py 中。

```python
from mmengine.registry import OPTIMIZERS

@OPTIMIZERS.register_module()
class SuperOptim:
    pass
```

我们为这个优化器的使用写了一个新的配置文件 `optimizer_cfg.py`：

```python
optimizer = dict(type='SuperOptim')
```

那么就需要在读取配置文件和构造优化器之前，增加一行 `from my_package import my_module` 来保证将自定义的类 `SuperOptim` 注册到 OPTIMIZERS 注册器中：

```python
from mmengine import Config
from mmengine.Registry import OPTIMIZERS

from my_package import my_module

cfg = Config.fromfile('config_super_optim.py')
optimizer = OPTIMIZERS.build(cfg.optimizer)
```

这样就会导致除了修改配置文件之外，还需要根据配置文件的内容，来对应修改训练源代码（即增加一些 import 语句），违背了我们希望仅通过修改配置文件就能控制模块构造和使用的初衷。

为了解决这个问题，我们给配置文件定义了一个保留字段 `custom_imports`，用于将需要提前导入的 Python 模块，直接写在配置文件中。对于上述例子，就可以将配置文件写成如下：

```python
custom_imports = dict(imports=['my_package.my_module'], allow_failed_imports=False)
optimizer = dict(type='SuperOptim')
```

这样我们就不用在训练代码中增加对应的 import 语句，只需要修改配置文件就可以实现非侵入式导入自定义注册模块。
