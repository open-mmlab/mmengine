# 配置（Config）

- [配置（Config）](#配置config)
  - [配置文件读取](#配置文件读取)
  - [配置文件的使用](#配置文件的使用)
  - [配置文件的继承](#配置文件的继承)
    - [继承机制概述](#继承机制概述)
    - [修改继承字段](#修改继承字段)
    - [删除字典中的 key](#删除字典中的-key)
    - [引用被继承文件中的变量](#引用被继承文件中的变量)
  - [配置文件的导出](#配置文件的导出)
  - [其他进阶用法](#其他进阶用法)
    - [预定义字段](#预定义字段)
    - [命令行修改配置](#命令行修改配置)
    - [使用环境变量替换配置](#使用环境变量替换配置)
    - [导入自定义 Python 模块](#导入自定义-python-模块)
    - [跨项目继承配置文件](#跨项目继承配置文件)
    - [跨项目获取配置文件](#跨项目获取配置文件)
  - [纯 Python 风格的配置文件（Beta）](#纯-python-风格的配置文件beta)
    - [基本语法](#基本语法)
      - [模块构建](#模块构建)
      - [继承](#继承)
      - [配置文件的导出](#配置文件的导出-1)
    - [什么是 lazy import](#什么是-lazy-import)
    - [功能限制](#功能限制)
    - [迁移指南](#迁移指南)

MMEngine 实现了抽象的配置类（Config），为用户提供统一的配置访问接口。配置类能够支持不同格式的配置文件，包括 `python`，`json`，`yaml`，用户可以根据需求选择自己偏好的格式。配置类提供了类似字典或者 Python 对象属性的访问接口，用户可以十分自然地进行配置字段的读取和修改。为了方便算法框架管理配置文件，配置类也实现了一些特性，例如配置文件的字段继承等。

在开始教程之前，我们先将教程中需要用到的配置文件下载到本地（建议在临时目录下执行，方便后续删除示例配置文件）：

```bash
wget https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/config_sgd.py
wget https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/cross_repo.py
wget https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/custom_imports.py
wget https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/demo_train.py
wget https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/example.py
wget https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/learn_read_config.py
wget https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/my_module.py
wget https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/optimizer_cfg.py
wget https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/predefined_var.py
wget https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/refer_base_var.py
wget https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/replace_data_root.py
wget https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/replace_num_classes.py
wget https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/resnet50_delete_key.py
wget https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/resnet50_lr0.01.py
wget https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/resnet50_runtime.py
wget https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/resnet50.py
wget https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/runtime_cfg.py
wget https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/resources/config/modify_base_var.py
```

```{note}
配置类支持两种风格的配置文件，即纯文本风格的配置文件和纯 Python 风格的配置文件（v0.8.0 的新特性），二者在调用接口统一的前提下各有特色。对于尚且不了解配置类基本用法用户，建议从[配置文件读取](#配置文件读取) 一节开始阅读，以了解配置类的功能和纯文本配置文件的语法。在一些情况下，纯文本风格的配置文件写法更加简洁，语法兼容性更好（`json`、`yaml` 通用）。如果你希望配置文件的写法可以更加灵活，建议阅读并使用[纯 Python 风格的配置文件](#纯-python-风格的配置文件beta)（beta）
```

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

对于以上三种格式的文件，假设文件名分别为 `config.py`，`config.json`，`config.yml`，调用 `Config.fromfile('config.xxx')` 接口加载这三个文件都会得到相同的结果，构造了包含 3 个字段的配置对象。我们以 `config.py` 为例，我们先将示例配置文件下载到本地：

然后通过配置类的 `fromfile` 接口读取配置文件：

```python
from mmengine.config import Config

cfg = Config.fromfile('learn_read_config.py')
print(cfg)
```

```
Config (path: learn_read_config.py): {'test_int': 1, 'test_list': [1, 2, 3], 'test_dict': {'key1': 'value1', 'key2': 0.1}}
```

## 配置文件的使用

通过读取配置文件来初始化配置对象后，就可以像使用普通字典或者 Python 类一样来使用这个变量了。我们提供了两种访问接口，即类似字典的接口 `cfg['key']` 或者类似 Python 对象属性的接口 `cfg.key`。这两种接口都支持读写。

```python
print(cfg.test_int)
print(cfg.test_list)
print(cfg.test_dict)
cfg.test_int = 2

print(cfg['test_int'])
print(cfg['test_list'])
print(cfg['test_dict'])
cfg['test_list'][1] = 3
print(cfg['test_list'])
```

```
1
[1, 2, 3]
{'key1': 'value1', 'key2': 0.1}
2
[1, 2, 3]
{'key1': 'value1', 'key2': 0.1}
[1, 3, 3]
```

注意，配置文件中定义的嵌套字段（即类似字典的字段），在 Config 中会将其转化为 ConfigDict 类，该类继承了 Python 内置字典类型的全部接口，同时也支持以对象属性的方式访问数据。

在算法库中，可以将配置与注册器结合起来使用，达到通过配置文件来控制模块构造的目的。这里举一个在配置文件中定义优化器的例子。

假设我们已经定义了一个优化器的注册器 OPTIMIZERS，包括了各种优化器。那么首先写一个 `config_sgd.py`：

```python
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
```

然后在算法库中可以通过如下代码构造优化器对象。

```python
from mmengine import Config, optim
from mmengine.registry import OPTIMIZERS

import torch.nn as nn

cfg = Config.fromfile('config_sgd.py')

model = nn.Conv2d(1, 1, 1)
cfg.optimizer.params = model.parameters()
optimizer = OPTIMIZERS.build(cfg.optimizer)
print(optimizer)
```

```
SGD (
Parameter Group 0
    dampening: 0
    foreach: None
    lr: 0.1
    maximize: False
    momentum: 0.9
    nesterov: False
    weight_decay: 0.0001
)
```

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
print(cfg.optimizer)
```

```
{'type': 'SGD', 'lr': 0.02, 'momentum': 0.9, 'weight_decay': 0.0001}
```

这里 `_base_` 是配置文件的保留字段，指定了该配置文件的继承来源。支持继承多个文件，将同时获得这多个文件中的所有字段，但是要求继承的多个文件中**没有**相同名称的字段，否则会报错。

`runtime_cfg.py`：

```python
gpu_ids = [0, 1]
```

`resnet50_runtime.py`：

```python
_base_ = ['optimizer_cfg.py', 'runtime_cfg.py']
model = dict(type='ResNet', depth=50)
```

这时，读取配置文件 `resnet50_runtime.py` 会获得 3 个字段 `model`，`optimizer`，`gpu_ids`。

```python
cfg = Config.fromfile('resnet50_runtime.py')
print(cfg.optimizer)
```

```
{'type': 'SGD', 'lr': 0.02, 'momentum': 0.9, 'weight_decay': 0.0001}
```

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
print(cfg.optimizer)
```

```
{'type': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0001}
```

对于非字典类型的字段，例如整数，字符串，列表等，重新定义即可完全覆盖，例如下面的写法就将 `gpu_ids` 这个字段的值修改成了 `[0]`。

```python
_base_ = ['optimizer_cfg.py', 'runtime_cfg.py']
model = dict(type='ResNet', depth=50)
gpu_ids = [0]
```

### 删除字典中的 key

有时候我们对于继承过来的字典类型字段，不仅仅是想修改其中某些 key，可能还需要删除其中的一些 key。这时候在重新定义这个字典时，需要指定 `_delete_=True`，表示将没有在新定义的字典中出现的 key 全部删除。

`resnet50_delete_key.py`：

```python
_base_ = ['optimizer_cfg.py', 'runtime_cfg.py']
model = dict(type='ResNet', depth=50)
optimizer = dict(_delete_=True, type='SGD', lr=0.01)
```

这时候，`optimizer` 这个字典中就只有 `type` 和 `lr` 这两个 key，`momentum` 和 `weight_decay` 将不再被继承。

```python
cfg = Config.fromfile('resnet50_delete_key.py')
print(cfg.optimizer)
```

```
{'type': 'SGD', 'lr': 0.01}
```

### 引用被继承文件中的变量

有时我们想重复利用 `_base_` 中定义的字段内容，就可以通过 `{{_base_.xxxx}}` 获取来获取对应变量的拷贝。例如：

`refer_base_var.py`

```python
_base_ = ['resnet50.py']
a = {{_base_.model}}
```

解析后发现，`a` 的值变成了 `resnet50.py` 中定义的 `model`

```python
cfg = Config.fromfile('refer_base_var.py')
print(cfg.a)
```

```
{'type': 'ResNet', 'depth': 50}
```

我们可以在 `json`、`yaml`、`python` 三种类型的配置文件中，使用这种方式来获取 `_base_` 中定义的变量。

尽管这种获取 `_base_` 中定义变量的方式非常通用，但是在语法上存在一些限制，无法充分利用 `python` 类配置文件的动态特性。比如我们想在 `python` 类配置文件中，修改 `_base_` 中定义的变量：

```python
_base_ = ['resnet50.py']
a = {{_base_.model}}
a['type'] = 'MobileNet'
```

配置类是无法解析这样的配置文件的（解析时报错）。配置类提供了一种更 `pythonic` 的方式，让我们能够在 `python` 类配置文件中修改 `_base_` 中定义的变量（`python` 类配置文件专属特性，目前不支持在 `json`、`yaml` 配置文件中修改 `_base_` 中定义的变量）。

`modify_base_var.py`：

```python
_base_ = ['resnet50.py']
a = _base_.model
a.type = 'MobileNet'
```

```python
cfg = Config.fromfile('modify_base_var.py')
print(cfg.a)
```

```
{'type': 'MobileNet', 'depth': 50}
```

解析后发现，`a` 的 type 变成了 `MobileNet`。

## 配置文件的导出

在启动训练脚本时，用户可能通过传参的方式来修改配置文件的部分字段，为此我们提供了 `dump` 接口来导出更改后的配置文件。与读取配置文件类似，用户可以通过 `cfg.dump('config.xxx')` 来选择导出文件的格式。`dump` 同样可以导出有继承关系的配置文件，导出的文件可以被独立使用，不再依赖于 `_base_` 中定义的文件。

基于继承一节定义的 `resnet50.py`，我们将其加载后导出：

```python
cfg = Config.fromfile('resnet50.py')
cfg.dump('resnet50_dump.py')
```

`resnet50_dump.py`

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
model = dict(type='ResNet', depth=50)
```

类似的，我们可以导出 json、yaml 格式的配置文件

`resnet50_dump.yaml`

```yaml
model:
  depth: 50
  type: ResNet
optimizer:
  lr: 0.02
  momentum: 0.9
  type: SGD
  weight_decay: 0.0001
```

`resnet50_dump.json`

```json
{"optimizer": {"type": "SGD", "lr": 0.02, "momentum": 0.9, "weight_decay": 0.0001}, "model": {"type": "ResNet", "depth": 50}}
```

此外，`dump` 不仅能导出加载自文件的 `cfg`，还能导出加载自字典的 `cfg`

```python
cfg = Config(dict(a=1, b=2))
cfg.dump('dump_dict.py')
```

`dump_dict.py`

```python
a=1
b=2
```

## 其他进阶用法

这里介绍一下配置类的进阶用法，这些小技巧可能使用户开发和使用算法库更简单方便。

```{note}
需要注意的是，如果你用的是纯 Python 风格的配置文件，只有“命令行修改配置”一节中提到功能是有效的。
```

### 预定义字段

```{note}
该用法仅适用于非 `lazy_import` 模式，具体见纯 Python 风格的配置文件一节
```

有时候我们希望配置文件中的一些字段和当前路径或者文件名等相关，这里举一个典型使用场景的例子。在训练模型时，我们会在配置文件中定义一个工作目录，存放这组实验配置的模型和日志，那么对于不同的配置文件，我们期望定义不同的工作目录。用户的一种常见选择是，直接使用配置文件名作为工作目录名的一部分，例如对于配置文件 `predefined_var.py`，工作目录就是 `./work_dir/predefined_var`。

使用预定义字段可以方便地实现这种需求，在配置文件 `predefined_var.py` 中可以这样写：

```Python
work_dir = './work_dir/{{fileBasenameNoExtension}}'
```

这里 `{{fileBasenameNoExtension}}` 表示该配置文件的文件名（不含拓展名），在配置类读取配置文件的时候，会将这种用双花括号包起来的字符串自动解析为对应的实际值。

```python
cfg = Config.fromfile('./predefined_var.py')
print(cfg.work_dir)
```

```
./work_dir/predefined_var
```

目前支持的预定义字段有以下四种，变量名参考自 [VS Code](https://code.visualstudio.com/docs/editor/variables-reference) 中的相关字段：

- `{{fileDirname}}` - 当前文件的目录名，例如 `/home/your-username/your-project/folder`
- `{{fileBasename}}` - 当前文件的文件名，例如 `file.py`
- `{{fileBasenameNoExtension}}` - 当前文件不包含扩展名的文件名，例如 `file`
- `{{fileExtname}}` - 当前文件的扩展名，例如 `.py`

### 命令行修改配置

有时候我们只希望修改部分配置，而不想修改配置文件本身，例如实验过程中想更换学习率，但是又不想重新写一个配置文件，常用的做法是在命令行传入参数来覆盖相关配置。考虑到我们想修改的配置通常是一些内层参数，如优化器的学习率、模型卷积层的通道数等，因此 MMEngine 提供了一套标准的流程，让我们能够在命令行里轻松修改配置文件中任意层级的参数。

1. 使用 `argparse` 解析脚本运行的参数
2. 使用 `argparse.ArgumentParser.add_argument` 方法时，让 `action` 参数的值为 [DictAction](mmengine.config.DictAction)，用它来进一步解析命令行参数中用于修改配置文件的参数
3. 使用配置类的 `merge_from_dict` 方法来更新配置

启动脚本示例如下：

`demo_train.py`

```python
import argparse

from mmengine.config import Config, DictAction


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    print(cfg)


if __name__ == '__main__':
    main()
```

示例配置文件如下：

`example.py`

```python
model = dict(type='CustomModel', in_channels=[1, 2, 3])
optimizer = dict(type='SGD', lr=0.01)
```

我们在命令行里通过 `.` 的方式来访问配置文件中的深层配置，例如我们想修改学习率，只需要在命令行执行：

```bash
python demo_train.py ./example.py --cfg-options optimizer.lr=0.1
```

```
Config (path: ./example.py): {'model': {'type': 'CustomModel', 'in_channels': [1, 2, 3]}, 'optimizer': {'type': 'SGD', 'lr': 0.1}}
```

我们成功地把学习率从 0.01 修改成 0.1。如果想改变列表、元组类型的配置，如上例中的 `in_channels`，则需要在命令行赋值时给 `()`，`[]` 外加上双引号：

```bash
python demo_train.py ./example.py --cfg-options model.in_channels="[1, 1, 1]"
```

```
Config (path: ./example.py): {'model': {'type': 'CustomModel', 'in_channels': [1, 1, 1]}, 'optimizer': {'type': 'SGD', 'lr': 0.01}}
```

`model.in_channels` 已经从 \[1, 2, 3\] 修改成 \[1, 1, 1\]。

```{note}
上述流程只支持在命令行里修改字符串、整型、浮点型、布尔型、None、列表、元组类型的配置项。对于列表、元组类型的配置，里面每个元素的类型也必须为上述七种类型之一。
```

:::{note}
`DictAction` 的行为与 `"extend"` 相似，支持多次传递，并保存在同一个列表中。如

```bash
python demo_train.py ./example.py --cfg-options optimizer.type="Adam" --cfg-options model.in_channels="[1, 1, 1]"
```

```
Config (path: ./example.py): {'model': {'type': 'CustomModel', 'in_channels': [1, 1, 1]}, 'optimizer': {'type': 'Adam', 'lr': 0.01}}
```

:::

### 使用环境变量替换配置

当要修改的配置嵌套很深时，我们在命令行中需要加上很长的前缀来进行定位。为了更方便地在命令行中修改配置，MMEngine 提供了一套通过环境变量来替换配置的方法。

在解析配置文件之前，MMEngine 会搜索所有的 `{{$ENV_VAR:DEF_VAL}}` 字段，并使用特定的环境变量来替换这一部分。这里 `ENV_VAR` 为替换这一部分所用的环境变量，`DEF_VAL` 为没有设置环境变量时的默认值。

例如，当我们想在命令行中修改数据集路径时，我们可以在配置文件 `replace_data_root.py` 中这样写：

```python
dataset_type = 'CocoDataset'
data_root = '{{$DATASET:/data/coco/}}'
dataset=dict(ann_file= data_root + 'train.json')
```

当我们运行 `demo_train.py` 来读取这个配置文件时：

```bash
python demo_train.py replace_data_root.py
```

```
Config (path: replace_data_root.py): {'dataset_type': 'CocoDataset', 'data_root': '/data/coco/', 'dataset': {'ann_file': '/data/coco/train.json'}}
```

这里没有设置环境变量 `DATASET`, 程序直接使用默认值 `/data/coco/` 来替换 `{{$DATASET:/data/coco/}}`。如果在命令行前设置设置环境变量则会有如下结果：

```bash
DATASET=/new/dataset/path/ python demo_train.py replace_data_root.py
```

```
Config (path: replace_data_root.py): {'dataset_type': 'CocoDataset', 'data_root': '/new/dataset/path/', 'dataset': {'ann_file': '/new/dataset/path/train.json'}}
```

`data_root` 被替换成了环境变量 `DATASET` 的值 `/new/dataset/path/`。

值得注意的是，`--cfg-options` 与 `{{$ENV_VAR:DEF_VAL}}` 都可以在命令行改变配置文件的值，但他们还有一些区别。环境变量的替换发生在配置文件解析之前。如果该配置还参与到其他配置的定义时，环境变量替换也会影响到其他配置，而 `--cfg-options` 只会改变要修改的配置文件的值。

我们以 `demo_train.py` 与 `replace_data_root.py` 为例。 如果我们通过配置 `--cfg-options data_root='/new/dataset/path'` 来修改 `data_root`：

```bash
python demo_train.py replace_data_root.py --cfg-options data_root='/new/dataset/path/'
```

```
Config (path: replace_data_root.py): {'dataset_type': 'CocoDataset', 'data_root': '/new/dataset/path/', 'dataset': {'ann_file': '/data/coco/train.json'}}
```

从输出结果上看，只有 `data_root` 被修改为新的值。`dataset.ann_file` 依然保持原始值。

作为对比，如果我们通过配置 `DATASET=/new/dataset/path` 来修改 `data_root`:

```bash
DATASET=/new/dataset/path/ python demo_train.py replace_data_root.py
```

```
Config (path: replace_data_root.py): {'dataset_type': 'CocoDataset', 'data_root': '/new/dataset/path/', 'dataset': {'ann_file': '/new/dataset/path/train.json'}}
```

`data_root` 与 `dataset.ann_file` 同时被修改了。

环境变量也可以用来替换字符串以外的配置，这时可以使用 `{{'$ENV_VAR:DEF_VAL'}}` 或者 `{{"$ENV_VAR:DEF_VAL"}}` 格式。`''` 与 `""` 用来保证配置文件合乎 python 语法。

例如，当我们想替换模型预测的类别数时，可以在配置文件 `replace_num_classes.py` 中这样写：

```
model=dict(
    bbox_head=dict(
        num_classes={{'$NUM_CLASSES:80'}}))
```

当我们运行 `demo_train.py` 来读取这个配置文件时：

```bash
python demo_train.py replace_num_classes.py
```

```
Config (path: replace_num_classes.py): {'model': {'bbox_head': {'num_classes': 80}}}
```

当设置 `NUM_CLASSES` 环境变量后：

```bash
NUM_CLASSES=20 python demo_train.py replace_num_classes.py
```

```
Config (path: replace_num_classes.py): {'model': {'bbox_head': {'num_classes': 20}}}
```

### 导入自定义 Python 模块

将配置与注册器结合起来使用时，如果我们往注册器中注册了一些自定义的类，就可能会遇到一些问题。因为读取配置文件的时候，这部分代码可能还没有被执行到，所以并未完成注册过程，从而导致构建自定义类的时候报错。

例如我们新实现了一种优化器 `CustomOptim`，相应代码在 `my_module.py` 中。

```python
from mmengine.registry import OPTIMIZERS

@OPTIMIZERS.register_module()
class CustomOptim:
    pass
```

我们为这个优化器的使用写了一个新的配置文件 `custom_imports.py`：

```python
optimizer = dict(type='CustomOptim')
```

那么就需要在读取配置文件和构造优化器之前，增加一行 `import my_module` 来保证将自定义的类 `CustomOptim` 注册到 OPTIMIZERS 注册器中：为了解决这个问题，我们给配置文件定义了一个保留字段 `custom_imports`，用于将需要提前导入的 Python 模块，直接写在配置文件中。对于上述例子，就可以将配置文件写成如下：

`custom_imports.py`

```python
custom_imports = dict(imports=['my_module'], allow_failed_imports=False)
optimizer = dict(type='CustomOptim')
```

这样我们就不用在训练代码中增加对应的 import 语句，只需要修改配置文件就可以实现非侵入式导入自定义注册模块。

```python
cfg = Config.fromfile('custom_imports.py')

from mmengine.registry import OPTIMIZERS

custom_optim = OPTIMIZERS.build(cfg.optimizer)
print(custom_optim)
```

```
<my_module.CustomOptim object at 0x7f6983a87970>
```

### 跨项目继承配置文件

为了避免基于已有算法库开发新项目时需要复制大量的配置文件，MMEngine 的配置类支持配置文件的跨项目继承。例如我们基于 MMDetection 开发新的算法库，需要使用以下 MMDetection 的配置文件：

```text
configs/_base_/schedules/schedule_1x.py
configs/_base_/datasets.coco_instance.py
configs/_base_/default_runtime.py
configs/_base_/models/faster-rcnn_r50_fpn.py
```

如果没有配置文件跨项目继承的功能，我们就需要把 MMDetection 的配置文件拷贝到当前项目，而我们现在只需要安装 MMDetection（如使用 `mim install mmdet`），在新项目的配置文件中按照以下方式继承 MMDetection 的配置文件：

`cross_repo.py`

```python
_base_ = [
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/datasets/coco_instance.py',
    'mmdet::_base_/default_runtime.py',
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
]
```

我们可以像加载普通配置文件一样加载 `cross_repo.py`

```python
cfg = Config.fromfile('cross_repo.py')
print(cfg.train_cfg)
```

```
{'type': 'EpochBasedTrainLoop', 'max_epochs': 12, 'val_interval': 1, '_scope_': 'mmdet'}
```

通过指定 `mmdet::`，Config 类会去检索 mmdet 包中的配置文件目录，并继承指定的配置文件。实际上，只要算法库的 `setup.py` 文件符合 [MMEngine 安装规范](todo)，在正确安装算法库以后，新的项目就可以使用上述用法去继承已有算法库的配置文件而无需拷贝。

### 跨项目获取配置文件

MMEngine 还提供了 `get_config` 和 `get_model` 两个接口，支持对符合 [MMEngine 安装规范](todo) 的算法库中的模型和配置文件做索引并进行 API 调用。通过 `get_model` 接口可以获得构建好的模型。通过 `get_config` 接口可以获得配置文件。

`get_model` 的使用样例如下所示，使用和跨项目继承配置文件相同的语法，指定 `mmdet::`，即可在 mmdet 包中检索对应的配置文件并构建和初始化相应模型。用户可以通过指定 `pretrained=True` 获得已经加载预训练权重的模型以进行训练或者推理。

```python
from mmengine.hub import get_model

model = get_model(
    'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py', pretrained=True)
print(type(model))
```

```
http loads checkpoint from path: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
<class 'mmdet.models.detectors.faster_rcnn.FasterRCNN'>
```

`get_config` 的使用样例如下所示，使用和跨项目继承配置文件相同的语法，指定 `mmdet::`，即可实现去 mmdet 包中检索并加载对应的配置文件。用户可以基于这样得到的配置文件进行推理修改并自定义自己的算法模型。同时，如果用户指定 `pretrained=True`，得到的配置文件中会新增 `model_path` 字段，指定了对应模型预训练权重的路径。

```python
from mmengine.hub import get_config

cfg = get_config(
    'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py', pretrained=True)
print(cfg.model_path)

```

```
https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
```

## 纯 Python 风格的配置文件（Beta）

在之前的教程里，我们介绍了如何使用配置文件，搭配注册器来构建模块；如何使用 `_base_` 来继承配置文件。这些纯文本风格的配置文件固然能够满足我们平时开发的大部分需求，并且一些模块的 alias 也大大简化了配置文件（例如 `ResNet` 就能指代 `mmcls.models.ResNet`）。但是也存在一些弊端：

1. 配置文件中，type 字段是通过字符串来指定的，在 IDE 中无法直接跳转到对应的类定义处，不利于代码阅读和跳转
2. 配置文件的继承，也是通过字符串来指定的，IDE 无法直接跳转到被继承的文件中，当配置文件继承结构复杂时，不利于配置文件的阅读和跳转
3. 继承规则较为隐式，初学者很难理解配置文件是如何对相同字段的变量进行融合，且衍生出 `_delete_` 这类特殊语法，学习成本较高
4. 用户忘记注册模块时，容易发生 module not found 的 error
5. 在尚且没有提到的跨库继承中，scope 的引入导致配置文件的继承规则更加复杂，初学者很难理解

综上所述，尽管纯文本风格的配置文件能够为 `python`、`json`、`yaml` 格式的配置提供相同的语法规则，但是当配置文件变得复杂时，纯文本风格的配置文件会显得力不从心。为此，我们提供了纯 Python 风格的配置文件，即 `lazy import` 模式，它能够充分利用 Python 的语法规则，解决上述问题。与此同时，纯 Python 风格的配置文件也支持导出成 `json` 和 `yaml` 格式。

### 基本语法

之前的教程分别介绍了基于纯文本风格配置文件的模块构建、继承和导出，本节将基于这三个方面来介绍纯 Python 风格的配置文件。

#### 模块构建

我们通过一个简单的例子来对比纯 Python 风格和纯文本风格的配置文件：

```{eval-rst}
.. tabs::
    .. tabs::

        .. code-tab:: python 纯 Python 风格

            # 无需注册

        .. code-tab:: python 纯文本风格

            # 注册流程
            from torch.optim import SGD
            from mmengine.registry import OPTIMIZERS

            OPTIMIZERS.register_module(module=SGD, name='SGD')

    .. tabs::

        .. code-tab:: python 纯 Python 风格

            # 配置文件写法
            from torch.optim import SGD


            optimizer = dict(type=SGD, lr=0.1)

        .. code-tab:: python 纯文本风格

            # 配置文件写法
            optimizer = dict(type='SGD', lr=0.1)

    .. tabs::

        .. code-tab:: python 纯 Python 风格

            # 构建流程完全一致
            import torch.nn as nn
            from mmengine.registry import OPTIMIZERS


            cfg = Config.fromfile('optimizer.py')
            model = nn.Conv2d(1, 1, 1)
            cfg.optimizer.params = model.parameters()
            optimizer = OPTIMIZERS.build(cfg.optimizer)

        .. code-tab:: python 纯文本风格

            # 构建流程完全一致
            import torch.nn as nn
            from mmengine.registry import OPTIMIZERS


            cfg = Config.fromfile('optimizer.py')
            model = nn.Conv2d(1, 1, 1)
            cfg.optimizer.params = model.parameters()
            optimizer = OPTIMIZERS.build(cfg.optimizer)
```

从上面的例子可以看出，纯 Python 风格的配置文件和纯文本风格的配置文件的区别在于：

1. 纯 Python 风格的配置文件无需注册模块
2. 纯 Python 风格的配置文件中，type 字段不再是字符串，而是直接指代模块。相应的配置文件需要多出 import 语法

需要注意的是，OpenMMLab 系列算法库在新增模块时仍会保留注册过程，用户基于 MMEngine 构建自己的项目时，如果使用纯 Python 风格的配置文件，则无需注册。看到这你会或许会好奇，这样没有安装 PyTorch 的环境不就没法解析样例配置文件了么，这样的配置文件还叫配置文件么？不要着急，这部分的内容我们会在后面介绍。

#### 继承

纯 Python 风格的配置文件继承语法有所不同：

```{eval-rst}
.. tabs::

    .. code-tab:: python 纯 Python 风格继承

        from mmengine.config import read_base


        with read_base():
            from .optimizer import *

    .. code-tab:: python 纯文本风格继承

        _base_ = [./optimizer.py]

```

纯 Python 风格的配置文件通过 import 语法来实现继承，这样做的好处是，我们可以直接跳转到被继承的配置文件中，方便阅读和跳转。变量的继承规则（增删改查）完全对齐 Python 语法，例如我想修改 base 配置文件中 optimizer 的学习率：

```python
from mmengine.config import read_base


with read_base():
    from .optimizer import *

# optimizer 为 base 配置文件定义的变量
optimizer.update(
    lr=0.01,
)
```

当然了，如果你已经习惯了纯文本风格的继承规则，且该变量在 _base_ 配置文件中为 `dict` 类型，也可以通过 merge 语法来实现和纯文本风格配置文件一致的继承规则：

```python
from mmengine.config import read_base


with read_base():
    from .optimizer import *

# optimizer 为 base 配置文件定义的变量
optimizer.merge(
    _delete_=True,
    lr=0.01,
    type='SGD'
)

# 等价的 python 风格写法如下，与 Python 的 import 规则完全一致
# optimizer = dict(
#     lr=0.01,
#     type='SGD'
# )
```

````{note}
需要注意的是，纯 Python 风格的配置文件中，字典的 `update` 方法与 `dict.update` 稍有不同。纯 Python 风格的 update 会递归地去更新字典中的内容，例如：

```python
x = dict(a=1, b=dict(c=2, d=3))

x.update(dict(b=dict(d=4)))
# 配置文件中的 update 规则：
# {a: 1, b: {c: 2, d: 4}}
# 普通 dict 的 update 规则：
# {a: 1, b: {d: 4}}
```

可见在配置文件中使用 update 方法会递归地去更新字段，而不是简单的覆盖。
````

与纯文本风格的配置文件相比，纯 Python 风格的配置文件的继承规则完全对齐 import 语法，更容易理解，且支持配置文件之间的跳转。你或许会好奇既然继承和模块的导入都使用了 import 语法，为什么继承配置文件还需要额外的 `with read_base():`  这个上下文管理器呢？一方面这样可以提升配置文件的可读性，可以让继承的配置文件更加突出，另一方面也是受限于 lazy_import 的规则，这个会在后面讲到。

#### 配置文件的导出

纯 python 风格配置文件也通过 dump 接口导出，使用上没有任何区别，但是导出的内容会有所不同：

```{eval-rst}
.. tabs::

    .. tabs::

        .. code-tab:: python 纯 Python 风格导出

            optimizer = dict(type='torch.optim.SGD', lr=0.1)

        .. code-tab:: python 纯文本风格导出

            optimizer = dict(type='SGD', lr=0.1)

    .. tabs::

        .. code-tab:: yaml 纯 Python 风格导出

            optimizer:
                type: torch.optim.SGD
                lr: 0.1

        .. code-tab:: yaml 纯文本风格导出

            optimizer:
                type: SGD
                lr: 0.1

    .. tabs::

        .. code-tab:: json 纯 Python 风格导出

            {"optimizer": "torch.optim.SGD", "lr": 0.1}

        .. code-tab:: json 纯文本风格导出

            {"optimizer": "SGD", "lr": 0.1}
```

可以看到，纯 Python 风格导出的 type 字段会包含模块的全量信息。导出的配置文件也可以被直接加载，通过注册器来构建实例。

### 什么是 lazy import

看到这你可能会吐槽，这纯 Python 风格的配置文件感觉就像是用纯 Python 语法来组织配置文件嘛。这样我哪还需要配置类，直接用 Python 语法来导入配置文件不就好了。如果你有这样的感受，那真是一件值得庆祝的事，因为这正是我们想要的效果。

正如前面所提到的，解析配置文件需要依赖配置文件中引用的三方库，这其实是一件非常不合理的事。例如我基于 MMagic 训练了一个模型，想使用 MMDeploy 的 onnxruntime 后端部署。由于部署环境中没有 torch，而配置文件解析过程中需要 torch，这就导致了我无法直接使用 MMagic 的配置文件作为部署的配置，这是非常不方便的。为了解决这个问题，我们引入了 lazy_import 的概念。

要聊 lazy_import 的具体实现是一件比较复杂的事，在此我们仅对其功能做简要介绍。lazy_import 的核心思想是，将配置文件中的 import 语句延迟到配置文件被解析时才执行，这样就可以避免配置文件中的 import 语句导致的三方库依赖问题。配置文件解析过程时，Python 解释器实际执行的等效代码如下

```{eval-rst}
.. tabs::
    .. code-tab:: python 原始配置文件

        from torch.optim import SGD


        optimizer = dict(type=SGD)

    .. code-tab:: python 通过配置类，Python 解释器实际执行的代码

        lazy_obj = LazyObject('torch.optim', 'SGD')

        optimizer = dict(type=lazy_obj)
```

LazyObject 作为 `Config` 模块的內部类型，无法被用户直接访问。用户在访问 type 字段时，会经过一系列的转换，将 `LazyObject` 转化成真正的 `torch.optim.SGD` 类型。这样一来，配置文件的解析不会触发三方库的导入，而用户使用配置文件时，又可以正常访问三方库的类型。

要想访问 `LazyObject` 的内部类型，可以通过 `Config.to_dict` 接口：

```python
cfg = Config.fromfile('optimizer.py').to_dict()
print(type(cfg['optimizer']['type']))
# mmengine.config.lazy.LazyObject
```

此时得到的 type 就是 `LazyObject` 类型。

然而对于 base 文件的继承（导入，import），我们不能够采取 lazy import 的策略，这是因为我们希望解析后的配置文件能够包含 base 配置文件定义的字段，需要真正的触发 import。因此我们对 base 文件的导入加了一层限制，即必须在 `with read_base()'` 的上下文中导入。

### 功能限制

1. 不能在配置文件中定义函数、类等
2. 配置文件名必须符合 Python 模块名的命名规范，即只能包含字母、数字、下划线，且不能以数字开头
3. 导入 base 配置文件中的变量时，例如 `from ._base_.alpha import beta`，此处的 `alpha` 必须是模块（module）名，即 Python 文件，而不能是含有 `__init__.py` 的包（package）名
4. 不支持在 absolute import 语句中同时导入多个变量，例如 `import torch, numpy, os`。需要通过多个 import 语句来实现，例如 `import torch; import numpy; import os`

### 迁移指南

从纯文本风格的配置文件迁移到纯 Python 风格的配置文件，需要遵守以下规则：

1. type 从字符串替换成具体的类：

   - 代码不依赖 type 字段是字符串，且没有对 type 字段做特殊处理，则可以将字符串类型的 type 替换成具体的类，并在配置文件的开头导入该类
   - 代码依赖 type 字段是字符串，则需要修改代码，或保持原有的字符串格式的 type

2. 重命名配置文件，配置文件命名需要符合 Python 模块名的命名规范，即只能包含字母、数字、下划线，且不能以数字开头

3. 删除 scope 相关配置。纯 Python 风格的配置文件不再需要通过 scope 来跨库调用模块，直接通过 import 导入即可。出于兼容性方面的考虑，我们仍然让 Runner 的 default_scope 参数为 `mmengine`，用户需要将其手动设置为 `None`

4. 对于注册器中存在别名的（alias）的模块，将其别名替换成其对应的真实模块即可，以下是常用的别名替换表：

   <table class="docutils">
    <thead>
    <tr>
        <th>模块</th>
        <th>别名</th>
        <th>注意事项</th>
    <thead>
    <tbody>
    <tr>
        <th>nearest</th>
        <th>torch.nn.modules.upsampling.Upsample</th>
        <th>将 type 替换成 Upsample 后，需要额外将 mode 参数指定为 'nearest'</th>
    </tr>
    <tr>
        <th>bilinear</th>
        <th>torch.nn.modules.upsampling.Upsample</th>
        <th>将 type 替换成 Upsample 后，需要额外将 mode 参数指定为 'bilinear'</th>
    </tr>
    <tr>
        <th>Clip</th>
        <th>mmcv.cnn.bricks.activation.Clamp</th>
        <th>无</th>
    </tr>
    <tr>
        <th>Conv</th>
        <th>mmcv.cnn.bricks.wrappers.Conv2d</th>
        <th>无</th>
    </tr>
    <tr>
        <th>BN</th>
        <th>torch.nn.modules.batchnorm.BatchNorm2d</th>
        <th>无</th>
    </tr>
    <tr>
        <th>BN1d</th>
        <th>torch.nn.modules.batchnorm.BatchNorm1d</th>
        <th>无</th>
    </tr>
    <tr>
        <th>BN2d</th>
        <th>torch.nn.modules.batchnorm.BatchNorm2d</th>
        <th>无</th>
    </tr>
    <tr>
        <th>BN3d</th>
        <th>torch.nn.modules.batchnorm.BatchNorm3d</th>
        <th>无</th>
    </tr>
    <tr>
        <th>SyncBN</th>
        <th>torch.nn.SyncBatchNorm</th>
        <th>无</th>
    </tr>
    <tr>
        <th>GN</th>
        <th>torch.nn.modules.normalization.GroupNorm</th>
        <th>无</th>
    </tr>
    <tr>
        <th>LN</th>
        <th>torch.nn.modules.normalization.LayerNorm</th>
        <th>无</th>
    </tr>
    <tr>
        <th>IN</th>
        <th>torch.nn.modules.instancenorm.InstanceNorm2d</th>
        <th>无</th>
    </tr>
    <tr>
        <th>IN1d</th>
        <th>torch.nn.modules.instancenorm.InstanceNorm1d</th>
        <th>无</th>
    </tr>
    <tr>
        <th>IN2d</th>
        <th>torch.nn.modules.instancenorm.InstanceNorm2d</th>
        <th>无</th>
    </tr>
    <tr>
        <th>IN3d</th>
        <th>torch.nn.modules.instancenorm.InstanceNorm3d</th>
        <th>无</th>
    </tr>
    <tr>
        <th>zero</th>
        <th>torch.nn.modules.padding.ZeroPad2d</th>
        <th>无</th>
    </tr>
    <tr>
        <th>reflect</th>
        <th>torch.nn.modules.padding.ReflectionPad2d</th>
        <th>无</th>
    </tr>
    <tr>
        <th>replicate</th>
        <th>torch.nn.modules.padding.ReplicationPad2d</th>
        <th>无</th>
    </tr>
    <tr>
        <th>ConvWS</th>
        <th>mmcv.cnn.bricks.conv_ws.ConvWS2d</th>
        <th>无</th>
    </tr>
    <tr>
        <th>ConvAWS</th>
        <th>mmcv.cnn.bricks.conv_ws.ConvAWS2d</th>
        <th>无</th>
    </tr>
    <tr>
        <th>HSwish</th>
        <th>torch.nn.modules.activation.Hardswish</th>
        <th>无</th>
    </tr>
    <tr>
        <th>pixel_shuffle</th>
        <th>mmcv.cnn.bricks.upsample.PixelShufflePack</th>
        <th>无</th>
    </tr>
    <tr>
        <th>deconv</th>
        <th>mmcv.cnn.bricks.wrappers.ConvTranspose2d</th>
        <th>无</th>
    </tr>
    <tr>
        <th>deconv3d</th>
        <th>mmcv.cnn.bricks.wrappers.ConvTranspose3d</th>
        <th>无</th>
    </tr>
    <tr>
        <th>Constant</th>
        <th>mmengine.model.weight_init.ConstantInit</th>
        <th>无</th>
    </tr>
    <tr>
        <th>Xavier</th>
        <th>mmengine.model.weight_init.XavierInit</th>
        <th>无</th>
    </tr>
    <tr>
        <th>Normal</th>
        <th>mmengine.model.weight_init.NormalInit</th>
        <th>无</th>
    </tr>
    <tr>
        <th>TruncNormal</th>
        <th>mmengine.model.weight_init.TruncNormalInit</th>
        <th>无</th>
    </tr>
    <tr>
        <th>Uniform</th>
        <th>mmengine.model.weight_init.UniformInit</th>
        <th>无</th>
    </tr>
    <tr>
        <th>Kaiming</th>
        <th>mmengine.model.weight_init.KaimingInit</th>
        <th>无</th>
    </tr>
    <tr>
        <th>Caffe2Xavier</th>
        <th>mmengine.model.weight_init.Caffe2XavierInit</th>
        <th>无</th>
    </tr>
    <tr>
        <th>Pretrained</th>
        <th>mmengine.model.weight_init.PretrainedInit</th>
        <th>无</th>
    </tr>
    </tbody>
    </table>
