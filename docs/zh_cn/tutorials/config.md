# **配置**

MMEngine 实现了抽象的配置类，为用户提供统一的配置访问接口。配置类能够支持不同格式的配置文件，包括 `python`，`json`，`yaml`，用户可以根据需求选择自己偏好的格式。配置类提供了类似 Python 类属性的访问接口，用户可以十分自然地进行配置字段的读取和修改。为了方便算法框架管理配置文件，配置类也实现了一些特性，例如配置文件的字段继承等。
```
Config 模块迁移自 mmcv 1.4.4
```
## 配置文件读取

以 `.py` 格式的配置文件为例，用户可以在配置文件中定义一些字段：

`config.py`：

```Python
test_int = 1
test_list = [1, 2, 3]
# include type, optimizer can be initiated by build_from_cfg
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
```

`mmengine.Config` 可以使用 `fromfile` 接口来解析配置文件。

```python
from mmengine import Config

cfg = Config.fromfile('/path/to/config.py')
# Config (path: config.py): {'test_int': 1, 'test_list': [1, 2, 3], 'optimizer': {'type': 'SGD', 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0.0001}}
```

对应的 `config.yaml` 和 `config.json` 可以被定义为：

`config.yaml`

````yaml
optimizer:
  lr: 0.1
  momentum: 0.9
  type: SGD
  weight_decay: 0.0001
test_int: 1
test_list:
- 1
- 2
- 3

````

`config.json`

````json
{
  "test_int": 1,
  "test_list": [
    1,
    2,
    3
  ],
  "optimizer": {
    "type": "SGD",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0001
  }
}
````

## 配置文件的使用

读取配置文件后，可以通过访问 `mmengine.Config` 的属性来访问/修改配置文件中定义的变量，也支持字典形式方式访问/修改变量。配置文件中 `dict` 类型的变量会被解析成  `mmegine.ConfigDict`。其余的 `python` 内置类型变量不发生改变。

```python
cfg = Config.fromfile('/path/to/config.py')

cfg.test_int # int: 1
cfg.test_list # int: [1, 2, 3]
cfg.optimizer # ConfigDict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
# accessed by __getitem__
cfg['test_int'] # int: 1
# change config attr
cfg.test_int = 2
cfg['test_int'] = 3
```

 对于 `mmengine` 或用户注册的实例，例如 `torch.optim.SGD`，可以调用 `mmengine.build_from_cfg` 来构建实例：

```python
from mmengine import Config, build_from_cfg
from mmengine.Registry import OPTIMIZER

cfg = Config.fromfile('/path/to/config.py')
optimizer = build_from_cfg(cfg.optimizer, OPTIMIZER)
# optimizer: torch.optim.SGD，cfg.optimizer: ConfigDict
```

## 配置文件进阶用法

### 解析预定义字段

`Config` 能解析预定义的字段，目前支持以下四种（变量名引用自[VS Code](https://code.visualstudio.com/docs/editor/variables-reference)）：

`{{ fileDirname }}` - 当前打开文件的目录名，例如 `/home/your-username/your-project/folder`

`{{ fileBasename }}` - 当前打开文件的文件名，例如 `file.ext`

`{{ fileBasenameNoExtension }}` - 当前打开文件不包含扩展名的文件名，例如 file

`{{ fileExtname }}` - 当前打开文件的扩展名，例如 `.py`

示例如下：

`config_predefined_var.py`：

```Python
a = 1
b = './work_dir/{{ fileBasenameNoExtension }}'
c = '{{ fileExtname }}'
```

`parse_config.py`：

```Python
cfg = Config.fromfile('./config_a.py')
# dict(a=1, b='./work_dir/config_a', c='.py')
```

### 解析外部模块

如果需要批量导入系统环境变量或者注册自定义模块，可以将需要导入的模块写入配置文件的 `custom_imports` 。`Config` 会在解析阶段导入 `custom_imports` 中定义的模块。

`config_custom_module.py`：

```Python
custom_imports = dict(imports=['path.to.module.my_module'], allow_failed_imports=False)
```

`my_module.py`：

```python
from mmcv.cnn import CONV_LAYERS
import os

os.environ["TEST_VALUE"] = 'test'
@CONV_LAYERS.register_module()
class NewConv1:
    pass

@CONV_LAYERS.register_module()
class NewConv2:
    pass

```

`parse_cfg.py`：

```Python
cfg = Config.fromfile('/path/to/config_custom_module.py', import_custom_modules=True)
assert os.environ["TEST_VALUE"] == 'test'
assert 'NewConv1' in CONV_LAYERS.module_dict
assert 'NewConv2' in CONV_LAYERS.module_dict
```

使用 `custom_imports` 能够非侵入式导入自定义注册模块。

### 继承式的配置文件

下游 `codebase` 一般会有通用配置文件，如 `default_runtime.py`，`default_schedule.py`。继承各种类型的通用配置可以减少具体任务的配置流程。以 `optimizer_cfg.py` 为例：

`optimizer_cfg.py`：

```Python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
max_epoch = 12
gpu_ids = [0, 1]
```

- **字段完全继承**

具体任务的配置文件不定义字段，直接继承 `default_runtime_cfg` 中的参数：

`task_config.py`：

```Python
_base_ = ['path/to/default_runtime_cfg.py']
```

- **修改字段**

有时候需要修改继承过来字段的值，对于 `int`、`list` 类型的字段，配置文件重新定义变量就能完成覆盖。对于 `dict` 类型的字段，完全覆盖需要加上 `_delete_` 关键字。

`task_config.py`：

```python
_base_ = ['path/to/default_runtime_cfg.py']
optimizer = dict(type='SGD', lr=0.1)
max_epoch = 24
gpus_ids = [0]
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
```

- **使用 base 文件中定义的变量**

有时需要重复利用 `_base_` 中定义的字段，可以通过 `{{}}` 获取来获取对应变量的拷贝。

`task_config.py`：

```python
_base_ = ['path/to/default_runtime_cfg.py']
a = {{_base_.optimizer}}
# Equivalent to： a = dict(type='SGD', lr=0.1)
```

:::{tip}
_base_ 的各个文件不能定义重名变量
:::

## 构建最简 runner

以 `.py` 格式的配置文件为例，要想基于 MMEngine 构建最简训练流程，需要给 `runner` 配置最低限度的参数，包括 `model` 、`optimzer`、`runner`、`env_cfg`、`scheduler` 和 `train_dataloader`，示例如下：

```Python
model = dict(type='custom_model')

# default EpochBasedRunner
max_epoch = 12
# config optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
# config dataloader
# Transform pipeline
train_transforms = [dict(type=...), dict(type=...)]
# training dataset
train_dataset = dict(type=..., transforms=train_transforms)
train_dataloader = dict(dataset=train_dataset,
                        batch_size=8)
scheduler = dict(type=...)
env_cfg = dict(backend='nccl')
```

其余组件，例如 `val_dataloader` 、`evaluator` 等为可选配置组件（允许训练阶段不起作用）。`optimizer_cfg`、`log_cfg`为默认配置组件（初始化阶段默认构造，无需配置，训练阶段起作用）。按照 配置文件的一级字段划分，总结如下：

![image](https://user-images.githubusercontent.com/57566630/153580725-18fe12df-9068-40c8-8cf1-7257f674a951.png)

每个模块的配置方法可见具体模块的用户文档。
