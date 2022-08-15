# 抽象数据接口

MMEngine 提供了类字典的 `BaseDataElement` 抽象数据结构。同时基于 `BaseDataElement` 还实现了 `InstanceData`， `PixelData`， `LabelData` 三个典型的子类,用以封装
算法过程中具有相似属性的数据，并针对他们的数据特性支持了一些额外的功能。

- `InstanceData`：假定它封装的数据具有相同的长度 N，N 代表实例的个数，并基于此假定对数据进行校验、支持索引和拼接功能。
- `PixelData`：假定它封装的数据有相同的长度和宽度，最后两维度为被封装数据的长宽，`PixelData` 基于此假定对数据进行校验、支持对实例进行空间维度的索引。
- `LabelData`：封装标签数据，如场景分类标签等。提供了 onehot 与 label互转。

## BaseDataElement

`BaseDataElement` 中存在两种类型的数据，一种是 `data` 类型，如标注框、框的标签、和实例掩码等；另一种是 `metainfo` 类型，包含数据的元信息以确保数据的完整性，如 `img_shape`, `img_id` 等数据所在图片的一些基本信息，方便可视化等情况下对数据进行恢复和使用。用户在创建 `BaseDataElement` 的过程中需要对这两类属性的数据进行显式地区分和声明。

两种类型的抽象数据接口都可以作为 Python 类去使用和操作他们的属性。同时，因为他们封装的数据大多是 Tensor，他们也提供了类似 Tensor 的基础操作。

### 1. 数据元素的创建

`BaseDataElement` 的 data 参数可以直接通过 `key=value` 的方式自由添加，metainfo 的字段需要显式通过关键字 `metainfo` 指定。

```python
import torch
from mmengine import BaseDataElement
# 可以声明一个空的 object
data_element = BaseDataElement()

bboxes = torch.rand((5, 4))  # 假定 bboxes 是一个 Nx4 维的 tensor，N 代表框的个数
scores = torch.rand((5,))  # 假定框的分数是一个 N 维的 tensor，N 代表框的个数
img_id = 0  # 图像的 ID
H = 800  # 图像的高度
W = 1333  # 图像的宽度

# 直接设置 BaseDataElement 的 data 参数
data_element = BaseDataElement(bboxes=bboxes, scores=scores)

# 显式声明来设置 BaseDataElement 的参数 metainfo
data_element = BaseDataElement(
    bboxes=bboxes,
    scores=scores,
    metainfo=dict(img_id=img_id, img_shape=(H, W)))
```

### 2. `new` 与 `clone` 函数

用户可以使用 `new()` 函数通过已有的数据接口创建一个具有相同状态和数据的抽象数据接口。用户可以在创建新 `BaseDataElement` 时设置 metainfo 和 data，用与创建仅 data 或 metainfo
具有相同状态和数据的抽象接口。比如 `new(metainfo=xx)` 使得新的 BaseDataElement 与被clone 的 `BaseDataElement` 包含相同状态和数据的 `data` 内容，但 `metainfo` 为新设置的内容。
也可以直接使用 `clone()` 来获得一份深拷贝，`clone()` 函数的行为与 PyTorch 中 Tensor 的 `clone()` 参数保持一致。

```python
import torch
from mmengine import BaseDataElement
data_element = BaseDataElement(
    bboxes=torch.rand((5, 4)),
    scores=torch.rand((5,)),
    metainfo=dict(img_id=1, img_shape=(640, 640)))

# 可以在创建新 `BaseDataElement` 时设置 metainfo 和 data，使得新的 BaseDataElement 有相同未被设置的数据
data_element1 = data_element.new(metainfo=dict(img_id=2, img_shape=(320, 320)))
print('bbox' in data_element1) # True
print((data_element1.bbox == data_element.bbox).all()) # True
print(data_element1.img_id == 2) # True

data_element2 = data_element.new(label=torch.rand(5,))
print('bbox' not in data_element2) # True
print(data_element2.img_id == data_element.img_id)  # True
print('label' in data_element2)

# 也可以通过 `clone` 构建一个新的 object，新的 object 会拥有和 data_element 相同的 data 和 metainfo 内容以及状态。
data_element2 = data_element1.clone()
```

### 3. 属性的增加与查询

用户可以像增加类属性那样增加 `BaseDataElement` 的属性，此时数据会被**当作 data 类型**增加到 `BaseDataElement` 中。
如果需要增加 metainfo 属性，用户应当使用 `set_metainfo`。
用户可以可以通过 `keys`，`values`，和 `items` 来访问只存在于 data 中的键值，也可以通过 `metainfo_keys`，`metainfo_values`，和`metainfo_items` 来访问只存在于 metainfo 中的键值。
用户还能通过 `all_keys`，`all_values`， `all_items` 来访问 `BaseDataElement` 的所有的属性并且不区分他们的类型。

**注意：**

1. `BaseDataElement` 不支持 metainfo 和 data 属性中有同名的字段，所以用户应当避免 metainfo 和 data 属性中设置相同的字段，否则 `BaseDataElement` 会报错。
2. 考虑到 `InstanceData` 和 `PixelData` 支持对数据进行切片操作，为了避免 `[]` 用法的不一致，同时减少同种需求的不同方法，`BaseDataElement` 不支持像字典那样访问和设置它的属性，所以类似 `BaseDataElement[name]` 的取值赋值操作是不被支持的。

```python
import torch
from mmengine import BaseDataElement
data_element = BaseDataElement()
# 设置 data_element 的 meta 字段，img_id 和 img_shape 会被作为 metainfo 的字段成为 data_element 的属性
data_element.set_metainfo(dict(img_id=9, img_shape=(100, 100)))
print('img_shape' in data_element.metainfo_keys()) # True
# 'img_shape' 是 data_element 的属性
print('img_shape' in data_element) # True
# img_shape 不是 data_element 的 data 字段
print('img_shape' not in data_element.keys()) # True
# 通过 all_keys 来访问所有属性
print('img_shape' in data_element.all_keys()) # True
# 访问类属性一样访问 'img_shape'
print(data_element.img_shape) # (100, 100)

# 直接设置 data_element 的 scores 属性，默认该数据属于 data
data_element.scores = torch.rand((5,))
print('scores' in data_element.keys()) # True
# 'scores' 是 data_element 的属性
print('scores' in data_element)  # True
# 通过 all_keys 来访问所有属性
print('scores' in data_element.all_keys()) # True
# scores 不是 data_element 的 metainfo 字段
print('scores' not in data_element.metainfo_keys()) # True
# 访问类属性一样访问 'scores'
print(data_element.scores) # tensor([0.5112, 0.3111, 0.5239, 0.0580, 0.2831])

# 设置 data_element 的 data 字段 bboxes
data_element.bboxes = torch.rand((5, 4))
print('bboxes' in data_element.items()) # True
# 'bboxes' 是 data_element 的属性
print('bboxes' in data_element)  # True
# 通过 all_keys 来访问所有属性
print('bboxes' in data_element.all_keys()) # True
# bboxes 不是 data_element 的 metainfo 字段
print('bboxes' not in data_element.metainfo_keys()) # True
# 访问类属性一样访问 'bboxes'
print(data_element.bboxes)
# tensor([[0.1741, 0.7271, 0.6782, 0.2834],
#         [0.0641, 0.0768, 0.8332, 0.0974],
#         [0.7939, 0.1719, 0.2888, 0.2681],
#         [0.4588, 0.1208, 0.5458, 0.3537],
#         [0.5942, 0.5481, 0.6155, 0.9495]])

for k, v in data_element.all_items():
    print(f'{k}: {v}')  # 包含 img_shapes， img_id， bboxes，scores
# img_id: 9
# img_shape: (100, 100)
# scores: tensor([0.5112, 0.3111, 0.5239, 0.0580, 0.2831])
# bboxes: tensor([[0.1741, 0.7271, 0.6782, 0.2834],
#         [0.0641, 0.0768, 0.8332, 0.0974],
#         [0.7939, 0.1719, 0.2888, 0.2681],
#         [0.4588, 0.1208, 0.5458, 0.3537],
#         [0.5942, 0.5481, 0.6155, 0.9495]])

for k, v in data_element.metainfo_items():
    print(f'{k}: {v}')  # 包含 img_shapes， img_id
# img_id: 9
# img_shape: (100, 100)

for k, v in data_element.items():
    print(f'{k}: {v}')  # 包含 bboxes，scores
scores: tensor([0.5112, 0.3111, 0.5239, 0.0580, 0.2831])
# bboxes: tensor([[0.1741, 0.7271, 0.6782, 0.2834],
#         [0.0641, 0.0768, 0.8332, 0.0974],
#         [0.7939, 0.1719, 0.2888, 0.2681],
#         [0.4588, 0.1208, 0.5458, 0.3537],
#         [0.5942, 0.5481, 0.6155, 0.9495]])
```

### 4. 属性的删改

`BaseDataElement` 支持用户可以像使用一个类一样对它的`data`进行删改, 如果需要修改 metainfo 属性，用户应当使用 `set_metainfo`。
同时， `BaseDataElement` 支持 `get` 来允许在访问不到变量时设置默认值，也支持 `pop` 在在访问属性后删除属性。

```python
import torch
from mmengine import BaseDataElement
data_element = BaseDataElement(
    bboxes=torch.rand((6, 4)), scores=torch.rand((6,)),
    metainfo=dict(img_id=0, img_shape=(640, 640))
)

# 对 data 进行修改
data_element.bboxes = data_element.bboxes * 2

# 对 metainfo 进行修改
data_element.set_metainfo(dict(img_shape = (1280, 1280)))
print(data_element.img_shape)  # (1280, 1280)

# 提供了可设置默认值的获取方式 get
print(data_element.get('img_shape', None))  # (1280, 1280)
print(data_element.get('bboxes', None))    # 6x4 tensor

# 属性的删除
del data_element.img_shape
del data_element.bboxes
print('img_shape' not in data_element) # True
print('bboxes' not in data_element)  # True

# 提供了便捷的属性删除和访问操作 pop
data_element.pop('img_shape', None)  # None
data_element.pop('bboxes', None)  # None
```

### 5. 类张量操作

用户可以像 torch.Tensor 那样对 `BaseDataElement` 的 data 进行状态转换，目前支持 `cuda`， `cpu`， `to`， `numpy` 等操作。
其中，`to` 函数拥有和 `torch.Tensor.to()` 相同的接口，使得用户可以灵活地将被封装的 tensor 进行状态转换。
**注意：** 这些接口只会处理类型为 np.array，torch.Tensor，或者数字的序列，其他属性的数据（如字符串）会被跳过处理。

```python
import torch
from mmengine import BaseDataElement
data_element = BaseDataElement(
    bboxes=torch.rand((6, 4)), scores=torch.rand((6,)),
    metainfo=dict(img_id=0, img_shape=(640, 640))
)
# 将所有 data 转移到 GPU 上
cuda_element = data_element.cuda()
print(cuda_element.bboxes.device)  # cuda:0
cuda_element = data_element.to('cuda:0')
print(cuda_element.bboxes.device)  # cuda:0

# 将所有 data 转移到 cpu 上
cpu_element = cuda_element.cpu()
print(cpu_element.bboxes.device)  # cpu
cpu_element = cuda_element.to('cpu')
print(cpu_element.bboxes.device)  # cpu

# 将所有 data 变成 FP16
fp16_instances = cuda_element.to(
    device=None, dtype=torch.float16, non_blocking=False, copy=False,
    memory_format=torch.preserve_format)
print(fp16_instances.bboxes.dtype)  # torch.float16

# 阻断所有 data 的梯度
cpu_element = cuda_element.detach()
print(cpu_element.bboxes.requires_grad)  # False

# 转移 data 到 numpy array
np_instances = cpu_element.numpy()
print(type(np_instances.bboxes))  # <class 'numpy.ndarray'>
```

### 6. 属性的展示

`BaseDataElement` 还实现了 `__repr__`，因此，用户可以直接通过 `print` 函数看到其中的所有数据信息。
同时，为了便捷开发者 debug，`BaseDataElement` 中的属性都会添加进 `__dict__` 中，方便用户在 IDE 界面可以直观看到 `BaseDataElement` 中的内容。
一个完整的属性展示如下

```python
import torch
from mmengine import BaseDataElement
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
instance_data = BaseDataElement(metainfo=img_meta)
instance_data.det_labels = torch.LongTensor([0, 1, 2, 3])
instance_data.det_scores = torch.Tensor([0.01, 0.1, 0.2, 0.3])
print(instance_data)
# <BaseDataElement(

#     META INFORMATION
#     img_shape: (800, 1196, 3)
#     pad_shape: (800, 1216, 3)

#     DATA FIELDS
#     det_labels: tensor([0, 1, 2, 3])
#     det_scores: tensor([0.0100, 0.1000, 0.2000, 0.3000])
# ) at 0x7f346930b9d0>
```

## InstanceData

`InstanceData` 在 `BaseDataElement` 的基础上，对 data 中存储的数据做了限制，即存储在 data 中的数据的长度均相同。比如在目标检测中, 假设一张图像中有 N 个目标(instance)，可以将图像的所有边界框(bbox)，类别(label)等存储在 `InstanceData` 中, `InstanceData` 的 bbox 和label 的长度相同
均为 N。
基于上述假定对 `InstanceData`进行了扩展，包括：

- 对 `InstanceData` 中 data 所存储的数据进行了长度校验
- data 部分支持类字典访问和设置它的属性
- 支持基础索引，切片以及高级索引功能
- 支持具有**相同的 `key`** 但是不同 `InstanceData` 的拼接功能。
  这些扩展功能除了支持基础的数据结构， 比如`torch.tensor`, `numpy.dnarray`, `list`, `str`, `tuple`, 也可以是自定义的数据结构，只要自定义数据结构实现了 `__len__`, `__getitem__` and `cat`.

### 数据校验

`InstanceData` 中 data 的数据长度要保持一致，如果传入不同长度的新数据，将会报错。

```python
from mmengine.data import InstanceData
import torch
import numpy as np
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
instance_data = InstanceData(metainfo=img_meta)
print('img_shape' in instance_data)
True
instance_data.det_labels = torch.LongTensor([2, 3])
instance_data.det_scores. = torch.Tensor([0.8, 0.7])
instance_data.bboxes = torch.rand((2, 4))
print(len(instance_data))  # 2

instance_data.bboxes = torch.rand((3, 4))
#AssertionError: the length of values 3 is not consistent with the length of this :obj:`InstanceData` 2
```

### 类字典访问和设置属性

`InstanceData` 支持类似字典的操作访问和设置其 **data** 属性。

```python
from mmengine.data import InstanceData
import torch
import numpy as np
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
instance_data = InstanceData(metainfo=img_meta)
instance_data["det_labels"] = torch.LongTensor([2, 3])
instance_data["det_scores"] = torch.Tensor([0.8, 0.7])
instance_data.bboxes = torch.rand((2, 4))
print(instance_data["det_scores"])
# tensor([0.8000, 0.7000])

print(instance_data["bboxes"])
# tensor([[0.4016, 0.9736, 0.4230, 0.1427],
#         [0.6779, 0.2111, 0.0488, 0.3284]])
```

### 索引与切片

`InstanceData` 支持 Python 中类似列表的索引与切片，同时也支持类似 numpy 的高级索引操作。

```python
from mmengine.data import InstanceData
import torch
import numpy as np
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
instance_data = InstanceData(metainfo=img_meta)
instance_data.det_labels = torch.LongTensor([2, 3])
instance_data.det_scores. = torch.Tensor([0.8, 0.7])
instance_data.bboxes = torch.rand((2, 4))
print(instance_data)
# <InstanceData(

#     META INFORMATION
#     pad_shape: (800, 1216, 3)
#     img_shape: (800, 1196, 3)

#     DATA FIELDS
#     det_scores: tensor([0.8000, 0.7000])
#     det_labels: tensor([2, 3])
#     bboxes: tensor([[0.4016, 0.9736, 0.4230, 0.1427],
#                 [0.6779, 0.2111, 0.0488, 0.3284]])
# ) at 0x7f3691aee250>

# 索引
print(instance_data[1])
# <InstanceData(

#     META INFORMATION
#     pad_shape: (800, 1216, 3)
#     img_shape: (800, 1196, 3)

#     DATA FIELDS
#     det_scores: tensor([0.7000])
#     det_labels: tensor([3])
#     bboxes: tensor([[0.6779, 0.2111, 0.0488, 0.3284]])
# ) at 0x7f3576097370>

# 切片
print(instance_data[0:1])
# <InstanceData(

#     META INFORMATION
#     pad_shape: (800, 1216, 3)
#     img_shape: (800, 1196, 3)

#     DATA FIELDS
#     det_scores: tensor([0.8000])
#     det_labels: tensor([2])
#     bboxes: tensor([[0.4016, 0.9736, 0.4230, 0.1427]])
# ) at 0x7f3691af7910>

# 高级索引
#1. 列表索引
sorted_results = instance_data[instance_data.det_scores.sort().indices]
print(sorted_results)
# <InstanceData(

#     META INFORMATION
#     pad_shape: (800, 1216, 3)
#     img_shape: (800, 1196, 3)

#     DATA FIELDS
#     det_scores: tensor([0.7000, 0.8000])
#     det_labels: tensor([3, 2])
#     bboxes: tensor([[0.6779, 0.2111, 0.0488, 0.3284],
#                 [0.4016, 0.9736, 0.4230, 0.1427]])
# ) at 0x7f3691af7910>

#2. 布尔索引
filter_results = instance_data[instance_data.det_scores > 0.75]
print(filter_results)
# <InstanceData(
#     META INFORMATION
#     pad_shape: (800, 1216, 3)
#     img_shape: (800, 1196, 3)
#     DATA FIELDS
#     det_labels: tensor([2])
#     masks: [[11, 21, 31, 41]]
#     det_scores: tensor([0.8000])
#     bboxes: tensor([[0.9308, 0.4000, 0.6077, 0.5554]])
#     polygons: [[1, 2, 3, 4]]
# ) at 0x7f64ecf0ec40>

# 结果为空情况
empty_results = instance_data[instance_data.det_scores > 1]
print(empty_results)
# <InstanceData(

#     META INFORMATION
#     pad_shape: (800, 1216, 3)
#     img_shape: (800, 1196, 3)

#     DATA FIELDS
#     det_scores: tensor([])
#     det_labels: tensor([], dtype=torch.int64)
#     bboxes: tensor([], size=(0, 4))
# ) at 0x7f3691af7ee0>

```

### 拼接(cat)

`InstanceData` 支持具有**相同的 `key`** 的 `InstanceData` 的拼接功能。对于长度分别为 N 和 M 的两个 `InstanceData`， 拼接后为长度 N + M 的新的 `InstanceData`

```python
from mmengine.data import InstanceData
import torch
import numpy as np
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
instance_data = InstanceData(metainfo=img_meta)
instance_data.det_labels = torch.LongTensor([2, 3])
instance_data.det_scores. = torch.Tensor([0.8, 0.7])
instance_data.bboxes = torch.rand((2, 4))
print(len(instance_data))
# 2

cat_results = instance_data.cat([instance_data, instance_data])
print(cat_results)
# <InstanceData(

#     META INFORMATION
#     pad_shape: (800, 1216, 3)
#     img_shape: (800, 1196, 3)

#     DATA FIELDS
#     det_scores: tensor([0.8000, 0.7000, 0.8000, 0.7000])
#     det_labels: tensor([2, 3, 2, 3])
#     bboxes: tensor([[0.4016, 0.9736, 0.4230, 0.1427],
#                 [0.6779, 0.2111, 0.0488, 0.3284],
#                 [0.4016, 0.9736, 0.4230, 0.1427],
#                 [0.6779, 0.2111, 0.0488, 0.3284]])
# ) at 0x7f3691b42130>

print(len(cat_results))
# 4
```

### 自定义数据结构

对于自定义结构如果想使用上述扩展要求需要实现`__len__`, `__getitem__` 和 `cat`三个接口.

```python
from mmengine.data import InstanceData
import numpy as np

class TmpObject:
    def __init__(self, tmp) -> None:
        assert isinstance(tmp, list)
        self.tmp = tmp
    def __len__(self):
        return len(self.tmp)
    def __getitem__(self, item):
        if type(item) == int:
            if item >= len(self) or item < -len(self):  # type:ignore
                raise IndexError(f'Index {item} out of range!')
            else:
                # keep the dimension
                item = slice(item, None, len(self))
        return TmpObject(self.tmp[item])
    @staticmethod
    def cat(tmp_objs):
        assert all(isinstance(results, TmpObject) for results in tmp_objs)
        if len(tmp_objs) == 1:
            return tmp_objs[0]
        tmp_list = [tmp_obj.tmp for tmp_obj in tmp_objs]
        tmp_list = list(itertools.chain(*tmp_list))
        new_data = TmpObject(tmp_list)
        return new_data
    def __repr__(self):
        return str(self.tmp)

img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
instance_data = InstanceData(metainfo=img_meta)
instance_data.det_labels = torch.LongTensor([2, 3])
instance_data["det_scores"] = torch.Tensor([0.8, 0.7])
instance_data.bboxes = torch.rand((2, 4))
instance_data.polygons = TmpObject([[1, 2, 3, 4], [5, 6, 7, 8]])
print(instance_data)
# <InstanceData(
#     META INFORMATION
#     pad_shape: (800, 1196, 3)
#     img_shape: (800, 1216, 3)
#     DATA FIELDS
#     det_labels: tensor([2, 3])
#     det_scores: tensor([0.8, 0.7000])
#     bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188],
#         [0.8101, 0.3105, 0.5123, 0.6263]])
#     polygons: [[1, 2, 3, 4], [5, 6, 7, 8]]
#     ) at 0x7fb492de6280>

# 高级索引
print(instance_data[instance_data.det_scores > 0.75])
# <InstanceData(
#     META INFORMATION
#     pad_shape: (800, 1216, 3)
#     img_shape: (800, 1196, 3)
#     DATA FIELDS
#     det_labels: tensor([2])
#     det_scores: tensor([0.8000])
#     bboxes: tensor([[0.9308, 0.4000, 0.6077, 0.5554]])
#     polygons: [[1, 2, 3, 4]]
# ) at 0x7f64ecf0ec40>

# 拼接
print(instance_data.cat([instance_data, instance_data]))
# <InstanceData(
#     META INFORMATION
#     img_shape: (800, 1196, 3)
#     pad_shape: (800, 1216, 3)
#     DATA FIELDS
#     det_labels: tensor([2, 3, 2, 3])
#     bboxes: tensor([[0.7404, 0.6332, 0.1684, 0.9961],
#                 [0.2837, 0.8112, 0.5416, 0.2810],
#                 [0.7404, 0.6332, 0.1684, 0.9961],
#                 [0.2837, 0.8112, 0.5416, 0.2810]])
#     det_scores: tensor([0.8000, 0.7000, 0.8000, 0.7000])
#     polygons: [[1, 2, 3, 4], [5, 6, 7, 8],
#                 [1, 2, 3, 4], [5, 6, 7, 8]]
# ) at 0x7f203542feb0>
```

## PixelData

`PixelData` 在 `BaseDataElement` 的基础上，同样对对 data 中存储的数据做了限制:

- 所有 data 内的数据均为 3 维，并且顺序为 (通道，高， 宽)
- 所有在 data 内的数据要有相同的长和宽
  基于上述假定对 `PixelData`进行了扩展，包括：
- 对 `PixelData` 中 data 所存储的数据进行了尺寸的校验
- 支持对 data 部分的数据对实例进行空间维度的索引和切片。

### 数据校验

`PixelData` 会对传入到 data 的数据进行维度与长宽的校验。

```python
from mmengine import PixelData
import random
import torch
import numpy as np
metainfo = dict(
    img_id=random.randint(0, 100),
    img_shape=(random.randint(400, 600), random.randint(400, 600)))
image = np.random.randint(0, 255, (4, 20, 40))
featmap = torch.randint(0, 255, (10, 20, 40))
pixel_data = PixelData(metainfo=metainfo,
                       image=image,
                       featmap=featmap)
print(pixel_data.shape)
(20, 40)
# set
pixel_data.map3 = torch.randint(0, 255, (20, 40))
print(pixel_data.map3.shape)
# torch.Size([1, 20, 40])
pixel_data.map2 = torch.randint(0, 255, (3, 20, 30))
# AssertionError: the height and width of values (20, 30) is not consistent with the length of this :obj:`PixelData` (20, 40)
pixel_data.map2 = torch.randint(0, 255, (1, 3, 20, 40))
# AssertionError: The dim of value must be 2 or 3, but got 4
```

### 空间维度索引

`PixelData` 支持对 data 部分的数据对实例进行空间维度的索引和切片，只需传入长宽的索引即可。

```python
from mmengine import PixelData
import random
import torch
import numpy as np
metainfo = dict(
    img_id=random.randint(0, 100),
    img_shape=(random.randint(400, 600), random.randint(400, 600)))
image = np.random.randint(0, 255, (4, 20, 40))
featmap = torch.randint(0, 255, (10, 20, 40))
pixel_data = PixelData(metainfo=metainfo,
                       image=image,
                       featmap=featmap)
print(pixel_data.shape)
# (20, 40)

# 索引
slice_data = pixel_data[10, 20]
print(slice_data.shape)
# (1, 1)

# 切片
slice_data = pixel_data[10:20, 20:40]
print(slice_data.shape)
# (10, 20)
```

## LabelData

`LabelData` 主要用来封装标签数据，如场景分类标签，文字识别标签等。`LabelData` 没有对 data 做任何限制，只提供了两个额外功能：onehot 与 index 的转换。

```python
from mmengine import LabelData
import torch

item = torch.tensor([1], dtype=torch.int64)
num_classes = 10
onehot = LabelData.label_to_onehot(label=item, num_classes=num_classes)
print(onehot)
# tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])

index = LabelData.onehot_to_label(onehot=onehot)
print(index)
# tensor([1])
```
