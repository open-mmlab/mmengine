# 数据集基类

**MMEngine** 提供了数据集基类来满足各种任务对数据集的基础需求。

## 基本用法

### 数据标注文件规范

数据集基类（BaseDataset）假定数据标注文件满足 **OpenMMLab 2.0 数据集格式规范**。简而言之，标注文件是一个字典，必须包含 `metadata` 和 `data_infos` 两个字段。其中 `metadata` 是一个字典，里面包含数据集的元信息； `data_infos` 是一个列表，列表中每个元素是一个字典，该字典定义了一个原始数据（raw data），每个原始数据包含一个或若干个训练/测试样本。以下是一个标注文件的例子（该例子中每个原始数据只包含一个训练/测试样本）:

```python

{
    'metadata':
        {
            'classes': ('cat', 'dog'),
            ...
        },
    'data_infos':
        [
            {
                'img_path': "xxx/xxx_0.jpg",
                'img_label': 0,
                ...
            },
            {
                'img_path': "xxx/xxx_1.jpg",
                'img_label': 1,
                ...
            },
            ...
        ]
}
```

### 定义图像的数据集类

数据集基类是一个抽象类，它有且只有一个抽象方法 `_parse_raw_data()` 来解析标注文件里的每个原始数据，因此用户需要基于数据集基类实现 `_parse_raw_data()` 方法才可以实例化对象。

`_parse_raw_data()` 定义了将一个原始数据处理成一个或若干个训练/测试样本的方法。以下是一个使用数据集基类来实现某一具体数据集的例子。

```python
import os.path as osp

from mmengine.data import BaseDataset


class ToyDataset(BaseDataset):

    # 以上面标注文件为例，在这里 raw_data 代表 `data_infos` 对应列表里的某个字典：
    # {
    #    'img_path': "xxx/xxx_0.jpg",
    #    'img_label': 0,
    #    ...
    # }
    def _parse_raw_data(self, raw_data):
        img_prefix = self.data_prefix.get('img', None)
        if img_prefix is not None:
            raw_data['img_path'] = osp.join(
                img_prefix, raw_data['img_path')
        return raw_data

```

### 使用自定义数据集类

假设数据存放路径如下：
```
data
├── annotations
│   ├── train.json
├── train
│   ├── xxx/xxx_0.jpg
│   ├── xxx/xxx_1.jpg
│   ├── ...
```

则可以通过如下配置实例化 `ToyDataset`：

```python
pipeline = [
    dict(type='xxx', ...),
    dict(type='yyy', ...),
    ...
]

toy_dataset = ToyDataset(
    data_root='data/',
    data_prefix=dict(img='train/'),
    ann_file='annotations/train.json',
    pipeline=pipeline)
```

`toy_dataset` 主要提供了 `meta`, `get_data_info(idx)`, `__len__()`, `__getitem__()` 接口来访问具体的数据信息：
```python
# 获得 toy_dataset 的元信息，返回值为字典
toy_dataset.meta

# 获取 toy_dataset 中某个样本的全量信息，返回值为字典：
# {
#     'img_path': "data/train/xxx/xxx_0.jpg",
#     'img_label': 0,
#     ...
# }
toy_dataset.get_data_info(0)

# 获取 toy_dataset 的长度，即样本总数量，返回值为整数型
len(toy_dataset)

# 获取 toy_dataset 中某个样本经过 pipeline 之后的结果（也就是送入模型的数据），返回值为字典
toy_dataset[0]
```

经过以上步骤，可以了解基于数据集基类如何自定义新的数据集类，以及如何使用自定义数据集类。

### 定义视频的数据集类

在上面的例子中，标注文件的每个原始数据只包含一个训练/测试样本（通常是图像领域）。如果每个原始数据包含若干个训练/测试样本（通常是视频领域），则只需保证 `_parse_raw_data()` 的返回值为 `list[dict]` 即可：

```python
from mmengine.data import BaseDataset


class ToyVideoDataset(BaseDataset):

    # raw_data 仍为一个字典，但它包含了多个样本
    def _parse_raw_data(self, raw_data):
        data_infos = []

        ...

        for ... :

            data_info = dict()

            ...

            data_infos.append(data_info)

        return data_infos

```

`ToyVideoDataset` 使用方法与 `ToyDataset` 类似，在此不做赘述。

## 其它特性

数据集基类还包含以下特性：

### lazy init

在数据集类实例化时，需要读取并解析标注文件，因此会消耗一定时间。然而在某些情况比如预测可视化时，往往只需要数据集类的元信息（meta），并不需要读取与解析标注文件。为了节省这种情况下数据集类实例化的时间，我们定义了 lazy init：

```python
pipeline = [
    dict(type='xxx', ...),
    dict(type='yyy', ...),
    ...
]

toy_dataset = ToyDataset(
    data_root='data/',
    data_prefix=dict(img='train/'),
    ann_file='annotations/train.json',
    pipeline=pipeline,
    # 在这里传入 lazy_init 变量
    lazy_init=True)
```

此时 `toy_dataset` 并未被完全初始化，因为 `toy_dataset` 并不会读取与解析标注文件，只会设置数据集类的元信息（meta）。

自然的，如果之后需要访问具体的数据信息，可以调用 `toy_dataset.full_init()` 接口来执行完整的初始化过程，在这个过程中标注文件将被读取与解析。调用 `get_data_info(idx)`, `__len__()`, `__getitem__()` 接口也会执行完整的初始化过程。

**值得注意的是**, 通过调用 `__getitem__()` 接口来执行完整初始化会带来一定风险：如果一个数据集类首先通过设置 `lazy_init=True` 未进行完全初始化，然后直接送入数据加载器（dataloader）中，在后续读取数据的过程中，不同的 worker 会同时读取与解析标注文件，这会消耗大量的时间与内存。**因此，建议在需要访问具体数据之前，仅通过 `full_init()` 接口来执行完整的初始化过程。**

以上通过设置 `lazy_init=True` 未进行完全初始化，之后根据需求再进行完整初始化的方式，称为 lazy init。

### 节省内存

在具体的读取数据过程中，数据加载器（dataloader）通常会起多个 worker 来预取数据，多个 worker 都拥有完整的数据集类备份，因此内存中会存在多份相同的 `data_infos`，为了节省这部分内存消耗，数据集基类可以提前将 `data_infos` 序列化存入内存中，使得多个 worker 可以共享同一份 `data_infos`，以达到节省内存的目的。

数据集基类通过 `serialize_data` 变量（默认为 `True`）来控制是否提前将 `data_infos` 序列化存入内存中：

```python
pipeline = [
    dict(type='xxx', ...),
    dict(type='yyy', ...),
    ...
]

toy_dataset = ToyDataset(
    data_root='data/',
    data_prefix=dict(img='train/'),
    ann_file='annotations/train.json',
    pipeline=pipeline,
    # 在这里传入 serialize_data 变量
    serialize_data=False)
```

上面例子不会提前将 `data_infos` 序列化存入内存中，不建议使用这种方式实例化数据集类。

# 数据集基类包装

除了数据集基类，MMEngine 也提供了若干个数据集基类包装：`BaseConcatDataset`, `BaseRepeatDataset`, `BaseClassBalancedDataset`。这些数据集基类包装同样也支持 lazy init 与拥有节省内存的特性。

## BaseConcatDataset

MMEngine 提供了 `BaseConcatDataset` 包装来连接多个数据集，使用方法如下：

```python
from mmengine.data import BaseConcatDataset

pipeline = [
    dict(type='xxx', ...),
    dict(type='yyy', ...),
    ...
]

toy_dataset_1 = ToyDataset(
    data_root='data/',
    data_prefix=dict(img='train/'),
    ann_file='annotations/train.json',
    pipeline=pipeline)

toy_dataset_2 = ToyDataset(
    data_root='data/',
    data_prefix=dict(img='val/'),
    ann_file='annotations/val.json',
    pipeline=pipeline)

toy_dataset_12 = BaseConcatDataset(datasets=[toy_dataset_1, toy_dataset_2])

```

上述例子将数据集的 `train` 部分与 `val` 部分合成一个大的数据集。

## BaseRepeatDataset

MMEngine 提供了 `BaseRepeatDataset` 包装来重复采样某个数据集若干次，使用方法如下：

```python
from mmengine.data import BaseRepeatDataset

pipeline = [
    dict(type='xxx', ...),
    dict(type='yyy', ...),
    ...
]

toy_dataset = ToyDataset(
    data_root='data/',
    data_prefix=dict(img='train/'),
    ann_file='annotations/train.json',
    pipeline=pipeline)

toy_dataset_repeat = BaseRepeatDataset(dataset=toy_dataset, times=5)

```

上述例子将数据集的 `train` 部分重复采样了 5 次。

## BaseClassBalancedDataset

MMEngine 提供了 `BaseClassBalancedDataset` 包装来基于数据集中类别出现频率重复采样相应样本，**请注意，** `BaseClassBalancedDataset` 包装需要被包装的数据集必须支持 `get_cat_ids(idx)` 方法，`get_cat_ids(idx)` 方法返回一个列表，该列表包含了 `idx` 对应的 `data_info` 包含的样本类别，使用方法如下：

```python
from mmengine.data import BaseDataset, BaseClassBalancedDataset

class ToyDataset(BaseDataset):

    def _parse_raw_data(self, raw_data):
        img_prefix = self.data_prefix.get('img', None)
        if img_prefix is not None:
            raw_data['img_path'] = osp.join(
                img_prefix, raw_data['img_path')
        return raw_data

    def get_cat_ids(self, idx):
        data_info = self.get_data_info(idx)
        return [int(data_info['img_label'])]

pipeline = [
    dict(type='xxx', ...),
    dict(type='yyy', ...),
    ...
]

toy_dataset = ToyDataset(
    data_root='data/',
    data_prefix=dict(img='train/'),
    ann_file='annotations/train.json',
    pipeline=pipeline)

toy_dataset_repeat = BaseClassBalancedDataset(dataset=toy_dataset, oversample_thr=1e-3)

```

上述例子将数据集以 `oversample_thr=1e-3` 重新采样 `toy_dataset`。
