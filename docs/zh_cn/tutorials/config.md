# **Config**

MMEngine 使用配置文件为训练过程的各个组件提供实例化参数，目前支持 .py，.json，.yaml格式的配置文件。

## **通过配置文件传递参数**

### **解析 Python 内置类型**

程序运行时会使用 `Config.fromfile` 解析配置文件定义的变量。`.py` 格式的配置文件一般只含 python 的内置类型。其中 `dict` 在运行时会被解析成 `mmengine.ConfigDict` ，其余基础类型（str，list，tuple）保持类型不变 。对于含有 `type` 字段的 `dict` 可以使用 `build_from_cfg` 完成对象的实例化。

`train_config.py`：

```Python
test_int = 1
test_list = [1, 2, 3]
# include type, optimizer can be initiated by build_from_cfg
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
```

`parse_config.py`：

```Python
from mmengine.registry import OPTIMIZER
from mmengine import build_from_cfg, Config


if __name__ == '__main__':
    cfg_path = '/path/to/train_config.py'
    cfg = Config.fromfile(cfg_path)
    test_int = cfg.test_int
    # test_int: int
    test_list = cfg.test_list
    # test_list: list
    optimizer = build_from_cfg(cfg.optimizer, OPTIMIZER)
    # optimizer: torch.optim.SGD，cfg.optimizer: ConfigDict
```

### **解析预定义变量**

`Config` 还能解析预定义的变量，目前支持以下四种（变量名引用自[VS Code](https://code.visualstudio.com/docs/editor/variables-reference)）：

`{{ fileDirname }}` - 当前打开文件的目录名，例如 /home/your-username/your-project/folder

`{{ fileBasename }}` - 当前打开文件的文件名，例如 file.ext

`{{ fileBasenameNoExtension }}` - 当前打开文件不包含扩展名的文件名，例如 file

`{{ fileExtname }}` - 当前打开文件的扩展名，例如 .py

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
print(cfg)
# dict(a=1, b='./work_dir/config_a', c='.py')
```

### **解析外部模块**

如果需要批量导入系统环境变量或者注册自定义模块，可以将需要导入的文件写入配问文件的 `custom_imports` 字段。`Config` 会解析 `custom_imports` 关键字在运行时导入该模块修改的环境变量。

`config.py`：

```Python
custom_imports = dict(imports=['path.to.module.env_cfg'], allow_failed_imports=False)
env_cfg.py
os.environ["TEST_VALUE"] = 'test'
```

`parse_cfg.py`：

```Python
cfg = Config.fromfile(cfg_file, import_custom_modules=True)
print(os.environ["TEST_VALUE"]) 
# 'test', defined in env_cfg module
```

### **继承式的配置文件**

下游 codebase 一般会有通用配置文件，如 `default_runtime.py`，`default_schedule.py`。继承各种类型的通用配置文件可以减少具体任务的配置流程。以 `optimizer_cfg.py` 为例：

`default_runtime_cfg`：

```Python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
```

#### **子配置文件未定义变量，完全继承**

具体任务的配置文件可以不定义变量，直接继承 `default_runtime_cfg` 中的配置参数：

`task_sample.py`：

```Python
_base_ = ['path/to/default_runtime_cfg.py'] 
```

这样 `task_sample.py` 就获得了 `optimizer` 和 `optimizer_config` 的配置参数

#### **子配置文件修改变量**

有时候需要修改继承过来的变量，根据修改变量的字段层级，可分以下两种情况：

- 子文件需要修改的变量类型不是 `dict` ;或者需要 修改的变量类型是 `dict`，并且`_base_`中该字段类型同为`dict`。子文件中直接覆盖需要修改的变量即可。

`task_sample.py`：

```Apache
_base_ = ['path/to/default_runtime_cfg.py']
# lr: float -> float 
optimizer = dict(type='SGD', lr=0.1)
```

- 子文件需要修改的变量类型是 `dict`，而 `_base_` 中对应字段的类型不为 `dict`，需要使用 `_delete_` 关键字。

`task_sample.py`：

```Python
_base_ = ['path/to/default_runtime_cfg.py']
# grad_clip: None -> dict
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
```

#### 使用 base 文件中定义的变量

`task_sample.py`：

```python
_base_ = ['path/to/default_runtime_cfg.py']
a = {{_base_.optimizer}}
```

\```{tip} `_base_` 的各个文件不能定义重名变量 ```

### **json、yaml配置文件示例：**

`optimizer_cfg.yaml`：

```YAML
checkpoint_config:
  interval: 1
env_cfg:
  backend: nccl
load_from: null
log_config:
  hooks:
  - type: TextLoggerHook
  interval: 100
log_level: INFO
optimizer:
  lr: 0.02
  momentum: 0.9
  type: SGD
  weight_decay: 0.0001
optimizer_config:

  grad_clip: null
```

`optimizer_cfg.json`：

```JSON
{
  "checkpoint_config": {
    "interval": 1
  },
  "log_config": {
    "interval": 100,
    "hooks": [
      {
        "type": "TextLoggerHook"
      }
    ]
  },
  "optimizer": {
    "type": "SGD",
    "lr": 0.02,
    "momentum": 0.9,
    "weight_decay": 0.0001
  },
  "optimizer_config": {
    "grad_clip": null
  },
  "env_cfg": {
    "backend": "nccl"
  },
  "log_level": "INFO",
  "load_from": null
}
```

## **构建最简 runner**

以 .py 格式的配置文件为例，要想基于 MMEngine 构建最简训练流程，则需要给 `runner` 配置最低限度的参数，包括 `model` 、`optimzer`、`runner`、`env_cfg`、`scheduler` 和 `train_dataloader` 示例如下：

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

![runner模块分布](../../en/_static/runner_module.png)

每个模块的配置方法可见具体模块的用户文档。

