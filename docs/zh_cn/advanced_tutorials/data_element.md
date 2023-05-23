# 抽象数据接口

在模型的训练/测试过程中，组件之间往往有大量的数据需要传递，不同的算法需要传递的数据经常是不一样的，例如，训练单阶段检测器需要获得数据集的标注框（ground truth bounding boxes）和标签（ground truth box labels），训练 Mask R-CNN 时还需要实例掩码（instance masks）。
训练这些模型时的代码如下所示

```python
for img, img_metas, gt_bboxes, gt_labels in data_loader:
    loss = retinanet(img, img_metas, gt_bboxes, gt_labels)
```

```python
for img, img_metas, gt_bboxes, gt_masks, gt_labels in data_loader:
    loss = mask_rcnn(img, img_metas, gt_bboxes, gt_masks, gt_labels)
```

可以发现，在不加封装的情况下，不同算法所需数据的不一致导致了不同算法模块之间接口的不一致，影响了算法库的拓展性，同时一个算法库内的模块为了保持兼容性往往在接口上存在冗余。
上述弊端在算法库之间会体现地更加明显，导致在实现多任务（同时进行如语义分割、检测、关键点检测等多个任务）感知模型时模块难以复用，接口难以拓展。

为了解决上述问题，MMEngine 定义了一套抽象的数据接口来封装模型运行过程中的各种数据。假设将上述不同的数据封装进 `data_sample` ，不同算法的训练都可以被抽象和统一成如下代码

```python
for img, data_sample in dataloader:
    loss = model(img, data_sample)
```

通过对各种数据提供统一的封装，抽象数据接口统一并简化了算法库中各个模块的接口，可以被用于算法库中 dataset，model，visualizer，和 evaluator 组件之间，或者 model 内各个模块之间的数据传递。
抽象数据接口实现了基本的增/删/改/查功能，同时支持不同设备之间的迁移，支持类字典和张量的操作，可以充分满足算法库对于这些数据的使用要求。
基于 MMEngine 的算法库可以继承这套抽象数据接口并实现自己的抽象数据接口来适应不同算法中数据的特点与实际需要，在保持统一接口的同时提高了算法模块的拓展性。

在实际实现过程中，算法库中的各个组件所具备的数据接口，一般为以下两种：

- 一个训练或测试样本（例如一张图像）的所有的标注信息和预测信息的集合，例如数据集的输出、模型以及可视化器的输入一般为单个训练或测试样本的所有信息。MMEngine 将其定义为数据样本（DataSample）
- 单一类型的预测或标注，一般是算法模型中某个子模块的输出, 例如二阶段检测中RPN的输出、语义分割模型的输出、关键点分支的输出，GAN中生成器的输出等。MMEngine 将其定义为数据元素（XXXData）

下边首先介绍一下数据样本与数据元素的基类 [BaseDataElement](mmengine.structures.BaseDataElement)。

## 数据基类(BaseDataElement)

`BaseDataElement` 中存在两种类型的数据，一种是 `data` 类型，如标注框、框的标签、和实例掩码等；另一种是 `metainfo` 类型，包含数据的元信息以确保数据的完整性，如 `img_shape`, `img_id` 等数据所在图片的一些基本信息，方便可视化等情况下对数据进行恢复和使用。用户在创建 `BaseDataElement` 的过程中需要对这两类属性的数据进行显式地区分和声明。

为了能够更加方便地使用 `BaseDataElement`，`data` 和 `metainfo` 中的数据均为 `BaseDataElement` 的属性。我们可以通过访问类属性的方式直接访问 `data` 和 `metainfo` 中的数据。此外，`BaseDataElement` 还提供了很多方法，方便我们操作 `data` 内的数据：

- 增/删/改/查 `data` 中不同字段的数据
- 将 `data` 迁移至目标设备
- 支持像访问字典/张量一样访问 data 内的数据以充分满足算法库对于这些数据的使用要求。

### 1. 数据元素的创建

`BaseDataElement` 的 data 参数可以直接通过 `key=value` 的方式自由添加，metainfo 的字段需要显式通过关键字 `metainfo` 指定。

```python
import torch
from mmengine.structures import BaseDataElement
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

用户可以使用 `new()` 方法基于已有的 `BaseDataElement` 创建一个具有相同 `data` 和 `metainfo` 的 `BaseDataElement`。用户也可以在调用 `new` 方法时传入新的 `data` 和 `metainfo`，例如 `new(metainfo=xx)` ，此时创建的 `BaseDataElement` 相较于已有的 `BaseDataElement`，`data` 完全一致 ，而 `metainfo` 则为新设置的内容。
也可以直接使用 `clone()` 来获得一份深拷贝，`clone()` 函数的行为与 PyTorch 中 Tensor 的 `clone()` 参数保持一致。

```python
data_element = BaseDataElement(
    bboxes=torch.rand((5, 4)),
    scores=torch.rand((5,)),
    metainfo=dict(img_id=1, img_shape=(640, 640)))

# 可以在创建新 `BaseDataElement` 时设置 metainfo 和 data，使得新的 BaseDataElement 有相同未被设置的数据
data_element1 = data_element.new(metainfo=dict(img_id=2, img_shape=(320, 320)))
print('bboxes is in data_element1:', 'bboxes' in data_element1) # True
print('bboxes in data_element1 is same as bbox in data_element', (data_element1.bboxes == data_element.bboxes).all())
print('img_id in data_element1 is', data_element1.img_id == 2) # True

data_element2 = data_element.new(label=torch.rand(5,))
print('bboxes is not in data_element2', 'bboxes' not in data_element2) # True
print('img_id in data_element2 is same as img_id in data_element', data_element2.img_id == data_element.img_id)
print('label in data_element2 is', 'label' in data_element2)

# 也可以通过 `clone` 构建一个新的 object，新的 object 会拥有和 data_element 相同的 data 和 metainfo 内容以及状态。
data_element2 = data_element1.clone()
```

```
bboxes is in data_element1: True
bboxes in data_element1 is same as bbox in data_element tensor(True)
img_id in data_element1 is True
bboxes is not in data_element2 True
img_id in data_element2 is same as img_id in data_element True
label in data_element2 is True
```

### 3. 属性的增加与查询

对增加属性而言，用户可以像增加类属性那样增加 `data` 内的属性；对 `metainfo` 而言，一般储存的为一些图像的元信息，一般情况下不会修改，如果需要增加，用户应当使用 `set_metainfo` 接口显示地修改。

对查询而言，用户可以可以通过 `keys`，`values`，和 `items` 来访问只存在于 data 中的键值，也可以通过 `metainfo_keys`，`metainfo_values`，和`metainfo_items` 来访问只存在于 metainfo 中的键值。
用户还能通过 `all_keys`，`all_values`， `all_items` 来访问 `BaseDataElement` 的所有的属性并且不区分他们的类型。

同时为了方便使用，用户可以像访问类属性一样访问 data 与 metainfo 内的数据，或着类字典方式通过 `get()` 接口访问数据。

**注意：**

1. `BaseDataElement` 不支持 metainfo 和 data 属性中有同名的字段，所以用户应当避免 metainfo 和 data 属性中设置相同的字段，否则 `BaseDataElement` 会报错。
2. 考虑到 `InstanceData` 和 `PixelData` 支持对数据进行切片操作，为了避免 `[]` 用法的不一致，同时减少同种需求的不同方法，`BaseDataElement` 不支持像字典那样访问和设置它的属性，所以类似 `BaseDataElement[name]` 的取值赋值操作是不被支持的。

```python
data_element = BaseDataElement()
# 通过 `set_metainfo`设置 data_element 的 metainfo 字段，
# 同时 img_id 和 img_shape 成为 data_element 的属性
data_element.set_metainfo(dict(img_id=9, img_shape=(100, 100)))
# 查看 metainfo 的 key, value 和 item
print("metainfo'keys are", data_element.metainfo_keys())
print("metainfo'values are", data_element.metainfo_values())
for k, v in data_element.metainfo_items():
    print(f'{k}: {v}')

print("通过类属性查看 img_id 和 img_shape")
print('img_id:', data_element.img_id)
print('img_shape:', data_element.img_shape)
```

```
metainfo'keys are ['img_id', 'img_shape']
metainfo'values are [9, (100, 100)]
img_id: 9
img_shape: (100, 100)
通过类属性查看 img_id 和 img_shape
img_id: 9
img_shape: (100, 100)
```

```python

# 通过类属性直接设置 BaseDataElement 中的 data 字段
data_element.scores = torch.rand((5,))
data_element.bboxes = torch.rand((5, 4))

print("data's key is:", data_element.keys())
print("data's value is:", data_element.values())
for k, v in data_element.items():
    print(f'{k}: {v}')

print("通过类属性查看 scores 和 bboxes")
print('scores:', data_element.scores)
print('bboxes:', data_element.bboxes)

print("通过 get() 查看 scores 和 bboxes")
print('scores:', data_element.get('scores', None))
print('bboxes:', data_element.get('bboxes', None))
print('fake:', data_element.get('fake', 'not exist'))
```

```
data's key is: ['scores', 'bboxes']
data's value is: [tensor([0.7937, 0.6307, 0.3682, 0.4425, 0.8515]), tensor([[0.9204, 0.2110, 0.2886, 0.7925],
        [0.7993, 0.8982, 0.5698, 0.4120],
        [0.7085, 0.7016, 0.3069, 0.3216],
        [0.0206, 0.5253, 0.1376, 0.9322],
        [0.2512, 0.7683, 0.3010, 0.2672]])]
scores: tensor([0.7937, 0.6307, 0.3682, 0.4425, 0.8515])
bboxes: tensor([[0.9204, 0.2110, 0.2886, 0.7925],
        [0.7993, 0.8982, 0.5698, 0.4120],
        [0.7085, 0.7016, 0.3069, 0.3216],
        [0.0206, 0.5253, 0.1376, 0.9322],
        [0.2512, 0.7683, 0.3010, 0.2672]])
通过类属性查看 scores 和 bboxes
scores: tensor([0.7937, 0.6307, 0.3682, 0.4425, 0.8515])
bboxes: tensor([[0.9204, 0.2110, 0.2886, 0.7925],
        [0.7993, 0.8982, 0.5698, 0.4120],
        [0.7085, 0.7016, 0.3069, 0.3216],
        [0.0206, 0.5253, 0.1376, 0.9322],
        [0.2512, 0.7683, 0.3010, 0.2672]])
通过 get() 查看 scores 和 bboxes
scores: tensor([0.7937, 0.6307, 0.3682, 0.4425, 0.8515])
bboxes: tensor([[0.9204, 0.2110, 0.2886, 0.7925],
        [0.7993, 0.8982, 0.5698, 0.4120],
        [0.7085, 0.7016, 0.3069, 0.3216],
        [0.0206, 0.5253, 0.1376, 0.9322],
        [0.2512, 0.7683, 0.3010, 0.2672]])
fake: not exist
```

```python

print("All key in data_element is:", data_element.all_keys())
print("The length of values in data_element is", len(data_element.all_values()))
for k, v in data_element.all_items():
    print(f'{k}: {v}')
```

```
All key in data_element is: ['img_id', 'img_shape', 'scores', 'bboxes']
The length of values in data_element is 4
img_id: 9
img_shape: (100, 100)
scores: tensor([0.7937, 0.6307, 0.3682, 0.4425, 0.8515])
bboxes: tensor([[0.9204, 0.2110, 0.2886, 0.7925],
        [0.7993, 0.8982, 0.5698, 0.4120],
        [0.7085, 0.7016, 0.3069, 0.3216],
        [0.0206, 0.5253, 0.1376, 0.9322],
        [0.2512, 0.7683, 0.3010, 0.2672]])
```

### 4. 属性的删改

用户可以像修改实例属性一样修改 `BaseDataElement` 的 `data`, 对`metainfo` 而言，一般储存的为一些图像的元信息，一般情况下不会修改，如果需要修改，用户应当使用 `set_metainfo` 接口显示的修改。

同时为了操作的便捷性，对 `data` 和 `metainfo` 中的数据可以通过 `del` 直接删除，也支持 `pop` 在访问属性后删除属性。

```python
data_element = BaseDataElement(
    bboxes=torch.rand((6, 4)), scores=torch.rand((6,)),
    metainfo=dict(img_id=0, img_shape=(640, 640))
)
for k, v in data_element.all_items():
    print(f'{k}: {v}')
```

```
img_id: 0
img_shape: (640, 640)
scores: tensor([0.8445, 0.6678, 0.8172, 0.9125, 0.7186, 0.5462])
bboxes: tensor([[0.5773, 0.0289, 0.4793, 0.7573],
        [0.8187, 0.8176, 0.3455, 0.3368],
        [0.6947, 0.5592, 0.7285, 0.0281],
        [0.7710, 0.9867, 0.7172, 0.5815],
        [0.3999, 0.9192, 0.7817, 0.2535],
        [0.2433, 0.0132, 0.1757, 0.6196]])
```

```python
# 对 data 进行修改
data_element.bboxes = data_element.bboxes * 2
data_element.scores = data_element.scores * -1
for k, v in data_element.items():
    print(f'{k}: {v}')

# 删除 data 中的属性
del data_element.bboxes
for k, v in data_element.items():
    print(f'{k}: {v}')

data_element.pop('scores', None)
print('The keys in data is', data_element.keys())
```

```
scores: tensor([-0.8445, -0.6678, -0.8172, -0.9125, -0.7186, -0.5462])
bboxes: tensor([[1.1546, 0.0578, 0.9586, 1.5146],
        [1.6374, 1.6352, 0.6911, 0.6735],
        [1.3893, 1.1185, 1.4569, 0.0562],
        [1.5420, 1.9734, 1.4344, 1.1630],
        [0.7999, 1.8384, 1.5635, 0.5070],
        [0.4867, 0.0264, 0.3514, 1.2392]])
scores: tensor([-0.8445, -0.6678, -0.8172, -0.9125, -0.7186, -0.5462])
The keys in data is []
```

```python
# 对 metainfo 进行修改
data_element.set_metainfo(dict(img_shape = (1280, 1280), img_id=10))
print(data_element.img_shape)  # (1280, 1280)
for k, v in data_element.metainfo_items():
    print(f'{k}: {v}')

# 提供了便捷的属性删除和访问操作 pop
del data_element.img_shape
for k, v in data_element.metainfo_items():
    print(f'{k}: {v}')

data_element.pop('img_id')
print('The keys in metainfo is', data_element.metainfo_keys())
```

```
(1280, 1280)
img_id: 10
img_shape: (1280, 1280)
img_id: 10
The keys in metainfo is []
```

### 5. 类张量操作

用户可以像 torch.Tensor 那样对 `BaseDataElement` 的 data 进行状态转换，目前支持 `cuda`， `cpu`， `to`， `numpy` 等操作。
其中，`to` 函数拥有和 `torch.Tensor.to()` 相同的接口，使得用户可以灵活地将被封装的 tensor 进行状态转换。
**注意：** 这些接口只会处理类型为 np.array，torch.Tensor，或者数字的序列，其他属性的数据（如字符串）会被跳过处理。

```python
data_element = BaseDataElement(
    bboxes=torch.rand((6, 4)), scores=torch.rand((6,)),
    metainfo=dict(img_id=0, img_shape=(640, 640))
)
# 将所有 data 转移到 GPU 上
cuda_element_1 = data_element.cuda()
print('cuda_element_1 is on the device of', cuda_element_1.bboxes.device)  # cuda:0
cuda_element_2 = data_element.to('cuda:0')
print('cuda_element_1 is on the device of', cuda_element_2.bboxes.device)  # cuda:0

# 将所有 data 转移到 cpu 上
cpu_element_1 = cuda_element_1.cpu()
print('cpu_element_1 is on the device of', cpu_element_1.bboxes.device)  # cpu
cpu_element_2 = cuda_element_2.to('cpu')
print('cpu_element_2 is on the device of', cpu_element_2.bboxes.device)  # cpu

# 将所有 data 变成 FP16
fp16_instances = cuda_element_1.to(
    device=None, dtype=torch.float16, non_blocking=False, copy=False,
    memory_format=torch.preserve_format)
print('The type of bboxes in fp16_instances is', fp16_instances.bboxes.dtype)  # torch.float16

# 阻断所有 data 的梯度
cuda_element_3 = cuda_element_2.detach()
print('The data in cuda_element_3 requires grad: ', cuda_element_3.bboxes.requires_grad)
# 转移 data 到 numpy array
np_instances = cpu_element_1.numpy()
print('The type of cpu_element_1 is convert to', type(np_instances.bboxes))
```

```
cuda_element_1 is on the device of cuda:0
cuda_element_1 is on the device of cuda:0
cpu_element_1 is on the device of cpu
cpu_element_2 is on the device of cpu
The type of bboxes in fp16_instances is torch.float16
The data in cuda_element_3 requires grad:  False
The type of cpu_element_1 is convert to <class 'numpy.ndarray'>
```

### 6. 属性的展示

`BaseDataElement` 还实现了 `__repr__`，因此，用户可以直接通过 `print` 函数看到其中的所有数据信息。
同时，为了便捷开发者 debug，`BaseDataElement` 中的属性都会添加进 `__dict__` 中，方便用户在 IDE 界面可以直观看到 `BaseDataElement` 中的内容。
一个完整的属性展示如下

```python
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
instance_data = BaseDataElement(metainfo=img_meta)
instance_data.det_labels = torch.LongTensor([0, 1, 2, 3])
instance_data.det_scores = torch.Tensor([0.01, 0.1, 0.2, 0.3])
print(instance_data)
```

```
<BaseDataElement(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([0, 1, 2, 3])
    det_scores: tensor([0.0100, 0.1000, 0.2000, 0.3000])
) at 0x7f9f339f85b0>
```

## 数据元素(xxxData)

MMEngine 将数据元素情况划分为三个类别：

- 实例数据 (InstanceData) : 主要针对的是上层任务 (high-level) 中，对图像中所有实例相关的数据进行封装，比如检测框 (bounding boxes)，物体类别 (box labels)，实例掩码 (instance masks)，关键点 (key points)，文字边界 (polygons)，跟踪 id (tracking ids) 等。所有实例相关的数据的**长度一致**，均为图像中实例的个数。
- 像素数据 (PixelData) : 主要针对底层任务 (low-level) 以及需要感知像素级别标签的部分上层任务。像素数据对像素级相关的数据进行封装，比如语义分割中的分割图 (segmentation map), 光流任务中的光流图 (flow map), 全景分割中的全景分割图 (panoptic seg map)；底层任务中生成的各种图像，比如超分辨图，去噪图，以及生成的各种风格图。这些数据的特点是都是三维或四维数组，最后两维度为数据的高度 (height) 和宽度 (width)，且具有相同的 height 和 width
- 标签数据 (LabelData) : 主要针对标签级别的数据进行封装，比如图像分类，多分类中的类别，图像生成中生成图像的类别内容，或者文字识别中的文本等。

### InstanceData

[`InstanceData`](mmengine.structures.InstanceData) 在 `BaseDataElement` 的基础上对 `data` 存储的数据做了限制，要求存储在 `data` 中的数据的长度一致。比如在目标检测中, 假设一张图像中有 N 个目标 (instance)，可以将图像的所有边界框 (bbox)，类别 (label) 等存储在 `InstanceData` 中, `InstanceData` 的 bbox 和 label 的长度相同。
MMEngine 对 `InstanceData` 加入了如下支持：

- 对 `InstanceData` 中 data 所存储的数据进行了长度校验
- data 部分支持类字典访问和设置它的属性
- 支持基础索引，切片以及高级索引功能
- 支持具有**相同的 `key`** 但是不同的 `InstanceData` 进行拼接的功能。

这些扩展功能除了支持基础的数据结构， 比如 `torch.tensor`, `numpy.dnarray`, `list`, `str` 和 `tuple`, 也可以是自定义的数据结构，只要自定义数据结构实现了 `__len__`, `__getitem__` 和 `cat` 方法。

#### 数据校验

`InstanceData` 中 data 的数据长度要保持一致，如果传入不同长度的新数据，将会报错。

```python
from mmengine.structures import InstanceData
import torch
import numpy as np

img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
instance_data = InstanceData(metainfo=img_meta)
instance_data.det_labels = torch.LongTensor([2, 3])
instance_data.det_scores = torch.Tensor([0.8, 0.7])
instance_data.bboxes = torch.rand((2, 4))
print('The length of instance_data is', len(instance_data))  # 2

instance_data.bboxes = torch.rand((3, 4))
```

```
The length of instance_data is 2
AssertionError: the length of values 3 is not consistent with the length of this :obj:`InstanceData` 2
```

#### 类字典访问和设置属性

`InstanceData` 支持类似字典的操作访问和设置其 **data** 属性。

```python
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
instance_data = InstanceData(metainfo=img_meta)
instance_data["det_labels"] = torch.LongTensor([2, 3])
instance_data["det_scores"] = torch.Tensor([0.8, 0.7])
instance_data.bboxes = torch.rand((2, 4))
print(instance_data)
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([2, 3])
    det_scores: tensor([0.8000, 0.7000])
    bboxes: tensor([[0.6576, 0.5435, 0.5253, 0.8273],
                [0.4533, 0.6848, 0.7230, 0.9279]])
) at 0x7f9f339f8ca0>
```

#### 索引与切片

`InstanceData` 支持 Python 中类似列表的索引与切片，同时也支持类似 numpy 的高级索引操作。

```python
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
instance_data = InstanceData(metainfo=img_meta)
instance_data.det_labels = torch.LongTensor([2, 3])
instance_data.det_scores = torch.Tensor([0.8, 0.7])
instance_data.bboxes = torch.rand((2, 4))
print(instance_data)
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([2, 3])
    det_scores: tensor([0.8000, 0.7000])
    bboxes: tensor([[0.1872, 0.1669, 0.7563, 0.8777],
                [0.3421, 0.7104, 0.6000, 0.1518]])
) at 0x7f9f312b4dc0>
```

1. 索引

```python
print(instance_data[1])
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([3])
    det_scores: tensor([0.7000])
    bboxes: tensor([[0.3421, 0.7104, 0.6000, 0.1518]])
) at 0x7f9f312b4610>
```

2. 切片

```python
print(instance_data[0:1])
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([2])
    det_scores: tensor([0.8000])
    bboxes: tensor([[0.1872, 0.1669, 0.7563, 0.8777]])
) at 0x7f9f312b4e20>
```

3. 高级索引

- 列表索引

```python
sorted_results = instance_data[instance_data.det_scores.sort().indices]
print(sorted_results)
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([3, 2])
    det_scores: tensor([0.7000, 0.8000])
    bboxes: tensor([[0.3421, 0.7104, 0.6000, 0.1518],
                [0.1872, 0.1669, 0.7563, 0.8777]])
) at 0x7f9f312b4a90>
```

- 布尔索引

```python
filter_results = instance_data[instance_data.det_scores > 0.75]
print(filter_results)
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([2])
    det_scores: tensor([0.8000])
    bboxes: tensor([[0.1872, 0.1669, 0.7563, 0.8777]])
) at 0x7fa061299dc0>
```

4. 结果为空

```python
empty_results = instance_data[instance_data.det_scores > 1]
print(empty_results)
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([], dtype=torch.int64)
    det_scores: tensor([])
    bboxes: tensor([], size=(0, 4))
) at 0x7f9f439cccd0>
```

#### 拼接(cat)

用户可以将两个具有相同 key 的 `InstanceData` 拼接成一个 `InstanceData`。对于长度分别为 N 和 M 的两个 `InstanceData`， 拼接后为长度 N + M 的新的 `InstanceData`

```python
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
instance_data = InstanceData(metainfo=img_meta)
instance_data.det_labels = torch.LongTensor([2, 3])
instance_data.det_scores = torch.Tensor([0.8, 0.7])
instance_data.bboxes = torch.rand((2, 4))
print('The length of instance_data is', len(instance_data))
cat_results = InstanceData.cat([instance_data, instance_data])
print('The length of instance_data is', len(cat_results))
print(cat_results)
```

```
The length of instance_data is 2
The length of instance_data is 4
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([2, 3, 2, 3])
    det_scores: tensor([0.8000, 0.7000, 0.8000, 0.7000])
    bboxes: tensor([[0.5341, 0.8962, 0.9043, 0.2824],
                [0.3864, 0.2215, 0.7610, 0.7060],
                [0.5341, 0.8962, 0.9043, 0.2824],
                [0.3864, 0.2215, 0.7610, 0.7060]])
) at 0x7fa061d4a9d0>
```

#### 自定义数据结构

对于自定义结构如果想使用上述扩展要求需要实现`__len__`, `__getitem__` 和 `cat`三个接口.

```python
import itertools

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
```

```python
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
instance_data = InstanceData(metainfo=img_meta)
instance_data.det_labels = torch.LongTensor([2, 3])
instance_data["det_scores"] = torch.Tensor([0.8, 0.7])
instance_data.bboxes = torch.rand((2, 4))
instance_data.polygons = TmpObject([[1, 2, 3, 4], [5, 6, 7, 8]])
print(instance_data)
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([2, 3])
    polygons: [[1, 2, 3, 4], [5, 6, 7, 8]]
    det_scores: tensor([0.8000, 0.7000])
    bboxes: tensor([[0.4207, 0.0778, 0.9959, 0.1967],
                [0.4679, 0.7934, 0.5372, 0.4655]])
) at 0x7fa061b5d2b0>
```

```python
# 高级索引
print(instance_data[instance_data.det_scores > 0.75])
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    bboxes: tensor([[0.4207, 0.0778, 0.9959, 0.1967]])
    det_labels: tensor([2])
    det_scores: tensor([0.8000])
    polygons: [[1, 2, 3, 4]]
) at 0x7f9f312716d0>
```

```python
# 拼接
print(InstanceData.cat([instance_data, instance_data]))
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    bboxes: tensor([[0.4207, 0.0778, 0.9959, 0.1967],
                [0.4679, 0.7934, 0.5372, 0.4655],
                [0.4207, 0.0778, 0.9959, 0.1967],
                [0.4679, 0.7934, 0.5372, 0.4655]])
    det_labels: tensor([2, 3, 2, 3])
    det_scores: tensor([0.8000, 0.7000, 0.8000, 0.7000])
    polygons: [[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]]
) at 0x7f9f31271490>
```

### PixelData

[`PixelData`](mmengine.structures.PixelData) 在 `BaseDataElement` 的基础上，同样对 data 中存储的数据做了限制:

- 所有 data 内的数据均为 3 维，并且顺序为 (通道，高，宽)
- 所有在 data 内的数据要有相同的长和宽

基于上述假定对 `PixelData`进行了扩展，包括：

- 对 `PixelData` 中 data 所存储的数据进行了尺寸的校验
- 支持对 data 部分的数据对实例进行空间维度的索引和切片。

#### 数据校验

`PixelData` 会对传入到 data 的数据进行维度与长宽的校验。

```python
from mmengine.structures import PixelData
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
print('The shape of pixel_data is', pixel_data.shape)
# set
pixel_data.map3 = torch.randint(0, 255, (20, 40))
print('The shape of pixel_data is', pixel_data.map3.shape)
```

```
The shape of pixel_data is (20, 40)
The shape of pixel_data is torch.Size([1, 20, 40])
```

```python
pixel_data.map2 = torch.randint(0, 255, (3, 20, 30))
# AssertionError: the height and width of values (20, 30) is not consistent with the length of this :obj:`PixelData` (20, 40)
```

```
AssertionError: the height and width of values (20, 30) is not consistent with the length of this :obj:`PixelData` (20, 40)
```

```python
pixel_data.map2 = torch.randint(0, 255, (1, 3, 20, 40))
# AssertionError: The dim of value must be 2 or 3, but got 4
```

```
AssertionError: The dim of value must be 2 or 3, but got 4
```

#### 空间维度索引

`PixelData` 支持对 data 部分的数据对实例进行空间维度的索引和切片，只需传入长宽的索引即可。

```python
metainfo = dict(
    img_id=random.randint(0, 100),
    img_shape=(random.randint(400, 600), random.randint(400, 600)))
image = np.random.randint(0, 255, (4, 20, 40))
featmap = torch.randint(0, 255, (10, 20, 40))
pixel_data = PixelData(metainfo=metainfo,
                       image=image,
                       featmap=featmap)
print('The shape of pixel_data is', pixel_data.shape)
```

```
The shape of pixel_data is (20, 40)
```

- 索引

```python
index_data = pixel_data[10, 20]
print('The shape of index_data is', index_data.shape)
```

```
The shape of index_data is (1, 1)
```

- 切片

```python
slice_data = pixel_data[10:20, 20:40]
print('The shape of slice_data is', slice_data.shape)
```

```
The shape of slice_data is (10, 20)
```

### LabelData

[`LabelData`](mmengine.structures.LabelData) 主要用来封装标签数据，如场景分类标签，文字识别标签等。`LabelData` 没有对 data 做任何限制，只提供了两个额外功能：onehot 与 index 的转换。

```python
from mmengine.structures import LabelData
import torch

item = torch.tensor([1], dtype=torch.int64)
num_classes = 10

```

```python
onehot = LabelData.label_to_onehot(label=item, num_classes=num_classes)
print(f'{num_classes} is convert to ', onehot)

index = LabelData.onehot_to_label(onehot=onehot)
print(f'{onehot} is convert to ', index)
```

```
10 is convert to  tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]) is convert to tensor([1])
```

## 数据样本(xxxDataSample)

一份样本中可能存在不同类型的标签，例如一张图片里可能同时存在实例级别的标签（Box），像素级别的标签（SegMap），因此在 PixelData、InstanceData、PixelData 之上，还会有一层更加高级封装，用来表示图像级别的标签。OpenMMLab 系列项目将这层封装命名为 `XXDataSample`。以检测任务为例，MMDet 就实现了 DetDataSample。训练过程中所有的标签都会被封装在 XXXDataSample 中，这样能够保证不同的深度学习任务能够保持统一的数据流和统一的数据操作方式。

### 下游库使用

以 MMDet 为例，说明下游库中数据样本的使用，以及数据样本字段的约束和命名。MMDet 中定义了 `DetDataSample`, 同时定义了 7 个字段，分别为：

- 标注信息
  - gt_instance(InstanceData): 实例标注信息，包括实例的类别、边界框等， 类型约束为 `InstanceData`。
  - gt_panoptic_seg(PixelData): 全景分割的标注信息，类型约束为 `PixelData`。
  - gt_semantic_seg(PixelData): 语义分割的标注信息， 类型约束为 `PixelData`。
- 预测结果
  - pred_instance(InstanceData): 实例预测结果，包括实例的类别、边界框等， 类型约束为 `InstanceData`。
  - pred_panoptic_seg(PixelData): 全景分割的预测结果，类型约束为 `PixelData`。
  - pred_semantic_seg(PixelData): 语义分割的预测结果， 类型约束为 `PixelData`。
- 中间结果
  - proposal(InstanceData): 主要为二阶段中 RPN 的预测结果， 类型约束为 `InstanceData`。

```python
from mmengine.structures import BaseDataElement
import torch

class DetDataSample(BaseDataElement):

    # 标注
    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def gt_panoptic_seg(self) -> PixelData:
        return self._gt_panoptic_seg

    @gt_panoptic_seg.setter
    def gt_panoptic_seg(self, value: PixelData):
        self.set_field(value, '_gt_panoptic_seg', dtype=PixelData)

    @gt_panoptic_seg.deleter
    def gt_panoptic_seg(self):
        del self._gt_panoptic_seg

    @property
    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: PixelData):
        self.set_field(value, '_gt_sem_seg', dtype=PixelData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self):
        del self._gt_sem_seg

    # 预测
    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    @property
    def pred_panoptic_seg(self) -> PixelData:
        return self._pred_panoptic_seg

    @pred_panoptic_seg.setter
    def pred_panoptic_seg(self, value: PixelData):
        self.set_field(value, '_pred_panoptic_seg', dtype=PixelData)

    @pred_panoptic_seg.deleter
    def pred_panoptic_seg(self):
        del self._pred_panoptic_seg

    # 中间结果
    @property
    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: PixelData):
        self.set_field(value, '_pred_sem_seg', dtype=PixelData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self):
        del self._pred_sem_seg

    @property
    def proposals(self) -> InstanceData:
        return self._proposals

    @proposals.setter
    def proposals(self, value: InstanceData):
        self.set_field(value, '_proposals', dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        del self._proposals

```

### 类型约束

DetDataSample 的用法如下所示，在数据类型不符合要求的时候(例如用 torch.Tensor 而非 InstanceData 定义 proposals 时)，DetDataSample 就会报错。

```python
data_sample = DetDataSample()

data_sample.proposals = InstanceData(data=dict(bboxes=torch.rand((5,4))))
print(data_sample)
```

```
<DetDataSample(

    META INFORMATION

    DATA FIELDS
    proposals: <InstanceData(

            META INFORMATION

            DATA FIELDS
            data:
                bboxes: tensor([[0.7513, 0.9275, 0.6169, 0.5581],
                            [0.6019, 0.6861, 0.7915, 0.0221],
                            [0.5977, 0.8987, 0.9541, 0.7877],
                            [0.0309, 0.1680, 0.1374, 0.0556],
                            [0.3842, 0.9965, 0.0747, 0.6546]])
        ) at 0x7f9f1c090310>
) at 0x7f9f1c090430>
```

```python
data_sample.proposals = torch.rand((5, 4))
```

```
AssertionError: tensor([[0.4370, 0.1661, 0.0902, 0.8421],
        [0.4947, 0.1668, 0.0083, 0.1111],
        [0.2041, 0.8663, 0.0563, 0.3279],
        [0.7817, 0.1938, 0.2499, 0.6748],
        [0.4524, 0.8265, 0.4262, 0.2215]]) should be a <class 'mmengine.data.instance_data.InstanceData'> but got <class 'torch.Tensor'>
```

## 接口的简化

下面以 MMDetection 为例更具体地说明 OpenMMLab 的算法库将如何迁移使用抽象数据接口，以简化模块和组件接口的。我们假定 MMDetection 和 MMEngine 中实现了 DetDataSample 和 InstanceData。

### 1. 组件接口的简化

检测器的外部接口可以得到显著的简化和统一。MMDet 2.X 中单阶段检测器和单阶段分割算法的接口如下。在训练过程中，`SingleStageDetector` 需要获取
`img`， `img_metas`， `gt_bboxes`， `gt_labels`， `gt_bboxes_ignore` 作为输入，但是 `SingleStageInstanceSegmentor` 还需要 `gt_masks`，导致 detector 的训练接口不一致，影响了代码的灵活性。

```python
class SingleStageDetector(BaseDetector):
    ...

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):


class SingleStageInstanceSegmentor(BaseDetector):
    ...

    def forward_train(self,
                      img,
                      img_metas,
                      gt_masks,
                      gt_labels,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
```

在 MMDet 3.0 中，所有检测器的训练接口都可以使用 DetDataSample 统一简化为 `img` 和 `data_samples`，不同模块可以根据需要去访问 `data_samples` 封装的各种所需要的属性。

```python
class SingleStageDetector(BaseDetector):
    ...

    def forward_train(self,
                      img,
                      data_samples):

class SingleStageInstanceSegmentor(BaseDetector):
    ...

    def forward_train(self,
                      img,
                      data_samples):

```

### 2. 模块接口的简化

MMDet 2.X 中 `HungarianAssigner` 和 `MaskHungarianAssigner` 分别用于在训练过程中将检测框和实例掩码和标注的实例进行匹配。他们内部的匹配逻辑实现是一样的，只是接口和损失函数的计算不同。
但是，接口的不同使得 `HungarianAssigner` 中的代码无法被复用，`MaskHungarianAssigner` 中重写了很多冗余的逻辑。

```python
class HungarianAssigner(BaseAssigner):

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):

class MaskHungarianAssigner(BaseAssigner):

    def assign(self,
               cls_pred,
               mask_pred,
               gt_labels,
               gt_mask,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
```

`InstanceData` 可以封装实例的框、分数、和掩码，将 `HungarianAssigner` 的核心参数简化成 `pred_instances`，`gt_instances`，和 `gt_instances_ignore`
使得 `HungarianAssigner` 和 `MaskHungarianAssigner` 可以合并成一个通用的 `HungarianAssigner`。

```python
class HungarianAssigner(BaseAssigner):

    def assign(self,
               pred_instances,
               gt_instancess,
               gt_instances_ignore=None,
               eps=1e-7):
```
