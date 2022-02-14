# 数据变换（Data Transform）

从数据集读出数据，到将数据投喂给模型之间，通常需要我们对数据做一系列的处理，包
括数据加载、格式化和增强等。与 PyTorch 类似，我们使用**数据变换**类来对数据进行
各种操作。进而将一系列数据变换组合为数据流水线（data pipeline）。

数据流水线中的每一个数据变换都定义了一种对数据字典的操作，因此每个数据变换类都
接受一个字典作为输入，同时输出一个字典用于接下来的变换。

在 MMEngine 中，我们提供了一个数据变换的基类 `BaseTransform` 和一些实用的变换包
装，来帮助我们构建灵活而强大的数据流水线。

## BaseTransform

数据变换的基类 `BaseTransform` 是一个抽象类，它只定义了数据变换类的接口。

一个新的数据变换类，只需要继承 `BaseTransform`，并实现 `transform` 函数即可。这
里，我们使用一个简单的随机翻转变换（`RandomFlip`）作为示例：

```python
import random
import mmcv
from mmengine.data import BaseTransform

class RandomFlip(BaseTransform):
    def transform(self, results: dict) -> dict:
        img = results['img']
        flip = True if random.random() > 0.5 else False
        if flip:
            results['img'] = mmcv.imflip(img)
        return results
```

进而，我们可以实例化一个 `RandomFlip` 对象，并将之作为一个可调用对象，来处理我
们的数据字典。

```python
import numpy as np
transform = RandomFlip()
data_info = dict(img=np.random.rand(224, 224, 3))
data_info = transform(data_info)
processed_img = data_info['img']
```

## 变换包装

变换包装是一种特殊的数据变换类，他们本身并不操作数据字典中的图像、标签等信息，
而是对其中定义的数据变换的行为进行增强。

### 字段映射（Remap）

字段映射包装（`Remap`）用于对数据字典中的字段进行映射。例如，一般的图像处理变换
都从数据字典中的 `"img"` 字段获得值。但有些时候，我们希望这些变换处理数据字典中
其他字段中的图像，比如 `"gt_img"` 字段。

如果配合注册器和配置文件使用的话，在配置文件中数据集的 `pipeline` 中如下例使用
字段映射包装：

```python
pipeline = [
    ...
    dict(type='Remap',
        input_mapping={'img': 'gt_img'},  # 将 "gt_img" 字段映射至 "img" 字段
        inplace=True,  # 在完成变换后，将 "img" 重映射回 "gt_img" 字段
        transforms=[
            # 在 `RandomFlip` 变换类中，我们只需要操作 "img" 字段即可
            dict(type='RandomFlip'),
        ])
    ...
]
```

利用字段映射包装，我们在实现数据变换类时，不需要考虑在 `transform` 方法中考虑各
种可能的输入字段名，只需要处理默认的字段即可。

### 随机选择（RandomChoice）

随机选择包装（`RandomChoice`）用于从一系列数据变换组合中随机应用一个数据变换组
合。利用这一包装，我们可以简单地实现一些数据增强功能，比如 AutoAugment。

如果配合注册器和配置文件使用的话，在配置文件中数据集的 `pipeline` 中如下例使用
随机选择包装：

```python
pipeline = [
    ...
    dict(type='RandomChoice',
        pipelines=[
            [
                dict(type='Posterize', bits=4),
                dict(type='Rotate', angle=30.)
            ],  # 第一种随机变化组合
            [
                dict(type='Equalize'),
                dict(type='Rotate', angle=30)
            ],  # 第二种随机变换组合
        ],
        pipeline_probs=[0.4, 0.6]  # 两种随机变换组合各自的选用概率
        )
    ...
]
```

### 多目标扩展（ApplyToMultiple）

通常，一个数据变换类只会从一个固定的字段读取操作目标。虽然我们也可以使用
`Remap` 来改变读取的字段，但无法将变换一次性应用于多个字段的数据。为了实现这一
功能，我们需要借助多目标扩展包装（`ApplyToMultiple`）。

多目标扩展包装（`ApplyToMultiple`）有两个用法，一是将数据变换作用于指定的多个
字段，二是将数据变换作用于某个字段下的一组目标中。

1. 应用于多个字段

   假设我们需要将数据变换应用于 `"lq"` (low-quanlity) 和 `"gt"` (ground-truth)
   两个字段中的图像上。

   ```python
   pipeline = [
       dict(type='ApplyToMultiple',
           # 分别应用于 "lq" 和 "gt" 两个字段，并将二者应设置 "img" 字段
           input_mapping={'img': ['lq', 'gt']},
           # 在完成变换后，将 "img" 字段重映射回原先的字段
           inplace=True,
           # 是否在对各目标的变换中共享随机变量
           # 更多介绍参加后续章节（随机变量共享）
           share_random_param=True,
           transforms=[
               # 在 `RandomFlip` 变换类中，我们只需要操作 "img" 字段即可
               dict(type='RandomFlip'),
           ])
   ]
   ```

2. 应用于一个字段的一组目标

   假设我们需要将数据变换应用于 `"images"` 字段中一个 list 的图像。

   ```python
   pipeline = [
       dict(type='ApplyToMultiple',
           # 将 "images" 字段下的每张图片映射至 "img" 字段
           input_mapping={'img': 'images'},
           # 在完成变换后，将 "img" 字段下的图片重映射回 "images" 字段的列表中
           inplace=True,
           # 是否在对各目标的变换中共享随机变量
           # 更多介绍参加后续章节（随机变量共享）
           share_random_param=True,
           transforms=[
               # 在 `RandomFlip` 变换类中，我们只需要操作 "img" 字段即可
               dict(type='RandomFlip'),
           ])
   ]
   ```

## 随机变量共享

有些情况下，我们希望在多次数据变换中共享随机状态。例如，在超分辨率任务中，我们
希望将随机变换**同步**作用于低分辨率图像和原始图像。

在 `ApplyToMultiple` 中，我们提供了 `share_random_param` 选项来启用这一功能。而
为了使这一功能生效，我们需要在数据变换类中标注哪些随机变量是支持共享的。

以上文中的 `RandomFlip` 为例：

```python
from mmengine.data.utils import cacheable_method

class RandomFlip(BaseTransform):
    @cacheable_method  # 标注该方法的输出为可共享的随机变量
    def do_flip(self):
        flip = True if random.random() > 0.5 else False
        return flip

    def transform(self, results: dict) -> dict:
        img = results['img']
        if self.do_flip():
            results['img'] = mmcv.imflip(img)
        return results
```

通过 `cacheable_method` 装饰器，方法返回值 `flip` 被标注为一个支持共享的随机变
量。进而，在 `ApplyToMultiple` 对多个目标的变换中，这一变量的值都会保持一致。

如果你对 `ApplyToMultiple` 是如何开关这一功能感到好奇，深入了解我们会发现，它使
用了上下文管理器 `cache_random_params` 来在特定范围内启用数据变换的随机变量共享。
我们可以通过一个小例子来体验这一功能。

```python
>>> import random
>>> from mmengine.data import BaseTransform
>>> from mmengine.data.utils import cacheable_method, cache_random_params
>>>
>>> class RandomNumber(BaseTransform):
...     @cacheable_method  # 标注该方法的输出为可共享的随机变量
...     def get_cached_random(self):
...         return random.random()
... 
...     def transform(self, results: dict) -> dict:
...         results['cache'] = self.get_cached_random()
...         results['no_cache'] = random.random()
...         return results
>>>
>>> transform = RandomNumber()
>>> # 没有 `cache_random_params` 时，被标注的随机变量也不会在多次调用中共享
>>> for i in range(3):
...     data_dict = transform({})
...     print(f'{data_dict["cache"]:.4f}, {data_dict["no_cache"]:.4f}')
0.7994, 0.1712
0.5317, 0.5089
0.6758, 0.0542
>>> # 在 `cache_random_params` 中，只有被标注的随机变量会在多次调用中共享
>>> with cache_random_params(transform):
...     for i in range(3):
...         data_dict = transform({})
...         print(f'{data_dict["cache"]:.4f}, {data_dict["no_cache"]:.4f}')
0.9899, 0.5399
0.9899, 0.4246
0.9899, 0.9384
```
