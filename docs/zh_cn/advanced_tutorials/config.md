# 配置（Config）

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

### 预定义字段

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
