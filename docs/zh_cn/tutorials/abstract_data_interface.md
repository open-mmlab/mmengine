# 抽象数据接口

在模型的训练/测试过程中，组件之间往往有大量的数据需要传递，不同的算法需要传递的数据经常是不一样的
例如，在 MMDetection 中训练单阶段检测器时，模型需要获得数据集传出标注框（ground truth bounding boxes）和标签（ground truth box labels），训练 Mask R-CNN 时还需要实例掩码（instance masks），训练 Panoptic FPN 时还需要语义分割掩码（semantic segmentation maps）。
算法之间所需数据的不一致导致了不同算法模块之间接口的不一致，这种不一致性在算法库之间体现地更加明显，导致在实现多任务（同时进行如语义分割、检测、关键点检测等多个任务）感知模型时有明显的不方便，使得一个算法库内的模块为了保持兼容性在接口上也存在冗余，也影响了算法库的拓展性。

为了解决上述问题，MMEngine 定义了一套抽象的数据接口来封装模型运行过程中产生的各种数据。
抽象数据接口会被用于算法库中 dataset，model，visualizer，和 evaluator 组件之间，或者 model 内各个模块之间的数据传递。
抽象数据接口实现了基本的增/删/改/查功能，同时支持不同设备之间的迁移，支持类字典和张量的操作，可以充分满足算法库对于这些数据的使用要求。
通过对各种数据提供统一的封装，抽象数据接口统一并简化了算法库中各个模块的接口。
基于 MMEngine 的算法库可以继承这套抽象数据接口并实现自己的抽象数据接口来适应不同算法中数据的特点与实际需要，在保持统一接口的同时提高了算法模块的拓展性。

## 设计

一个算法库中的数据可以被归类成具有不同性质的基础数据元素。一个训练样本（如一张图片）的所有基础数据元素构成了一个训练样本的完整数据，称为样本数据。相应地，MMEngine 为基础数据元素和样本数据分别定义了一种封装。

1. 基础数据元素的封装： 基础的数据元素指的是某一算法任务上的预测数据或标注，例如检测框，实例掩码，语义分割掩码等。因为标注数据和预测数据往往具有相似的性质（例如模型的预测框和标注框具有相同的数据特性），MMEngine 使用相同的抽象数据接口来封装预测数据和标注数据，并推荐使用命名来区分他们，如使用 `gt_instances` 和 `pred_instances` 来区分标注和预测的实例数据。另外，我们将基础数据元素区分为实例级别，像素级别，和标签级别。这些类型各有自己的特点，因此，MMEngine 定义了基础数据元素的基类 `BaseDataElement`，并由此派生出了 3 类数据结构来封装不同类型的标注数据或者模型的预测结果：`InstanceData`, `PixelData`, 和 `LabelData`。这些接口将被用于模型内各个模块之间的数据传递。

2. 样本数据的封装：一个训练样本（例如一张图片）的所有标注和预测构成了一个样本数据。一般情况下，一张图片可以同时有多种类型的标注和/或预测（例如，同时拥有像素级别的语义分割标注，标签级的场景分类标注，和实例级别的检测框标注）。因此，MMEngine 定义了 `BaseDataSample`作为样本数据封装的基类。也就是说，`BaseDataSample` 的属性会是各种类型的基础数据元素，OpenMMLab 算法库将基于 `BaseDataSample` 实现自己的抽象数据接口，来封装一个算法库中单个样本的所有相关数据，作为 dataset，model，visualizer，和 evaluator 组件之间的数据接口。

为了保证抽象数据接口内数据的完整性，抽象数据接口内部有两种数据，除了被封装的数据本身，还有一种是数据的元信息，例如图片大小和 ID 等。
两种类型的抽象数据接口都可以像 Python 的基础数据结构 dict 一样被使用。同时，因为他们封装的数据大多是 Tensor，他们也提供了类似 Tensor 的基础操作。

## 用法

### BaseDataElement

`BaseDataElement` 被用于模型内部模块之间的数据交换，同时对一些可以归结到一起的元素进行了封装。例如，其派生的 `InstanceData` 封装了检测框、框对应的标签和实例掩码、甚至关键点等数据。
`BaseDataElement` 中存在两种类型的数据，一种是 `data` 类型，如标注框、框的标签、和实例掩码等；另一种是 `meta_info` 类型，包含数据的元信息以确保数据的完整性，如 `img_shape`, `img_id` 等数据所在图片的一些基本信息，方便可视化等情况下对数据进行恢复和使用。用户在创建 `BaseDataElement` 的过程中需要对这两类属性的数据进行显式地区分和声明。

1. 数据元素的创建

```python
# 可以声明一个空的 object
gt_instances = BaseDataElement()

bboxes = torch.rand((5, 4))  # 假定 bboxes 是一个 Nx4 维的 tensor，N 代表框的个数
scores = torch.rand((5,))  # 假定框的分数是一个 N 维的 tensor，N 代表框的个数
img_id = 0  # 图像的 ID
H = 800  # 图像的高度
W = 1333  # 图像的宽度

# 显式声明 BaseDataElement 的参数 meta_info 和 data
gt_instances = BaseDataElement(
    meta_info=dict(img_id=img_id, img_shape=(H, W)),
    data=dict(bboxes=bboxes, scores=scores))

# 不显式声明的时候，传入字典将设置 BaseDataElement 的参数 meta_info
gt_instances = BaseDataElement(dict(img_id=img_id, img_shape=(H, W)))
```

2. 使用 `new()` 函数通过已有的数据接口创建一个具有相同状态和数据的抽象数据接口。用户可以在创建新 `BaseDataElement` 时设置 meta_info 和 data，使得新的 BaseDataElement 有相同的状态但是不同的数据。
也可以直接使用 `new()` 来获得一份深拷贝。

```python
gt_instances = BaseDataElement()

# 可以在创建新 `BaseDataElement` 时设置 meta_info 和 data，使得新的 BaseDataElement 有不同的数据但是数据在相同的 device 上
gt_instances1 = gt_instance.new(
    meta_info=dict(img_id=1, img_shape=(640, 640)),
    data=dict(bboxes=torch.rand((5, 4)), scores=torch.rand((5,)))
)

# 也可以声明一个新的 object，新的 object 会拥有和 gt_instance 相同的 data 和 meta_info 内容
gt_instances2 = gt_instances1.new()
```

3. 属性的增加与查询：用户可以通过 `set_meta_info` 和 `set_data` 来增加 `BaseDataElement` 的属性，这种方式会显式声明了字段属于 `meta_info` 还是 `data` ，并将属性添加到对应的字段列表中。
用户也可以像增加类属性或者字典字段那样增加 `BaseDataElement` 的属性，此时数据会被**默认为 data 类型**增加到 `BaseDataElement` 中。
类似的，用户可以通过 `meta_info_keys`，`meta_info_values`，和`meta_info_items` 来访问只存在于 meta_info 中的键值，
也可以通过 `data_keys`，`data_values`，和 `data_items` 来访问只存在于 data 中的键值。
用户还能通过 `keys`，`values`， `items` 来访问 `BaseDataElement` 的所有的属性并且不区分他们的类型。

```python
gt_instances = BaseDataElement()
# 设置 gt_instances 的 meta 字段，img_id 和 img_shape 会被作为 meta_info 的字段成为 gt_instances 的属性
gt_instances.set_meta_info(dict(img_id=9, img_shape=(100, 100))
assert 'img_shape' in gt_instaces.meta_info_keys()
# img_shape 不是 gt_instances 的 data 字段
assert 'img_shape' not in gt_instaces.data_keys()
# 通过 keys 来访问所有属性
assert 'img_shape' in gt_instaces.keys()
assert 'img_shape' in gt_instaces

# 设置 gt_instances 的 data 字段，bboxes 会被作为 data 的字段成为 gt_instances 的属性
gt_instances.set_data(dict(bboxes=bboxes))
assert 'bboxes' in gt_instances.data_keys()
# 通过 keys 来访问所有属性
assert 'bboxes' in gt_instances.keys()
assert 'bboxes' in gt_instances
# bboxes 不是 gt_instances 的 meta_info 字段
assert 'bboxes' not in gt_instances.meta_info_skeys()

# 直接设置 gt_instance 的 scores 属性，默认该数据属于 data
gt_instances.scores = torch.rand((5,))
assert 'scores' in gt_instances.data_keys()
# 通过 keys 来访问所有属性
assert 'scores' in gt_instances.keys()
assert 'scores' in gt_instances
# scores 不是 gt_instances 的 meta_info 字段
assert 'scores' not in gt_instances.meta_info_skeys()

# 直接像字典一样设置 gt_instance 的 labels 属性，默认该数据属于 data
gt_instances['labels'] = torch.rand((5,))
assert 'labels' in gt_instances.data_keys()
# 通过 keys 来访问所有属性
assert 'labels' in gt_instances.keys()
assert 'labels' in gt_instances
# labels 不是 gt_instances 的 meta_info 字段
assert 'labels' not in gt_instances.meta_info_skeys()

for k, v in gt_instances.items():
    print(f'{k}: {v}')  # 包含 img_shapes， img_id， bboxes，scores，labels
```

4. `BaseDataElement` 支持类字典操作来对字段进行删改，同时用户也可以像使用一个类一样对它的属性进行删改。

```python
gt_instances = BaseDataElement(
    meta_info=dict(img_id=0, img_shape=(640, 640))，
    data=dict(bboxes=torch.rand((6, 4)), scores=torch.rand((6,))))

# 对类的属性进行修改
gt_instances.img_shape = (1280, 1280)
gt_instances.img_shape  # (1280, 1280)
gt_instances.bboxes = gt_instances.bboxes * 2

# 像字典一样对属性进行修改
gt_instances['img_shape'] = (640, 640)
gt_instances['img_shape']  # (640, 640)
gt_instances['bboxes'] = gt_instances['bboxes'] / 2

# 提供了类字典的访问方式
gt_instances.get('img_shape', None)  # (640， 640)
gt_instances.get('bboxes', None)    # 6x4 tensor

# 属性的删除
del gt_instances.img_shape
del gt_instances.bboxes
assert 'img_shape' not in gt_instances
assert 'bboxes' not in gt_instances
gt_instances.pop('img_shape', None)  # None
gt_instances.get('bboxes', None)  # None
```

5. 用户可以像 torch.Tensor 那样对 `BaseDataElement` 的 data 进行状态转换，目前支持 `cuda`， `cpu`， `to`， `numpy` 等操作。
其中，`to` 函数拥有和 `torch.Tensor.to()` 相同的接口，使得用户可以灵活地将被封装的 tensor 进行状态转换。

```python
# 将所有 data 转移到 GPU 上
cuda_instances = gt_instances.cuda()
cuda_instances = gt_instancess.to('cuda:0')

# 将所有 data 转移到 cpu 上
cpu_instances = cuda_instances.cpu()
cpu_instances = cuda_instances.to('cpu')

# 将所有 data 变成 FP16
fp16_instances = cuda_instances.to(
    device=None, dtype=None, non_blocking=False, copy=False,
    memory_format=torch.preserve_format)

# 阻断所有 data 的梯度
cpu_instances = cuda_instances.detach()

# 转移 data 到 numpy array
np_instances = cpu_instances.numpy()
```

6. `BaseDataElement` 还实现了 `__nice__` 和 `__repr__`，因此，用户可以直接通过 `print` 函数看到其中的所有数据信息。
同时，为了便捷开发者 debug，`BaseDataElement` 中的属性都会添加进 `__dict__` 中，方便用户在 IDE 界面可以直观看到 `BaseDataElement` 中的内容。
一个完整的用例展示如下

```python
>>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
>>> instance_data = BaseDataElement(meta_info=img_meta)
>>> img_shape in instance_data
True
>>> instance_data.det_labels = torch.LongTensor([0, 1, 2, 3])
>>> instance_data["det_scores"] = torch.Tensor([0.01, 0.1, 0.2, 0.3])
>>> print(results)
<BaseDataElement(
  META INFORMATION
img_shape: (800, 1196, 3)
pad_shape: (800, 1216, 3)
  DATA FIELDS
shape of det_labels: torch.Size([4])
shape of det_scores: torch.Size([4])
) at 0x7f84acd10f90>
>>> instance_data.det_scores
tensor([0.0100, 0.1000, 0.2000, 0.3000])
>>> instance_data.det_labels
tensor([0, 1, 2, 3])
>>> instance_data['det_labels']
tensor([0, 1, 2, 3])
>>> 'det_labels' in instance_data
True
>>> instance_data.img_shape
(800, 1196, 3)
>>> 'det_scores' in instance_data
True
>>> del instance_data.det_scores
>>> 'det_scores' in instance_data
False
>>> det_labels = instance_data.pop('det_labels', None)
>>> det_labels
tensor([0, 1, 2, 3])
>>> 'det_labels' in instance_data
>>> False
```

### BaseDataSample

`BaseDataSample` 是所有基础数据元素的封装，并作为一个算法库内 dataset，visualizer，evaluator，model 组件之间的数据接口进行流通。
因此 `BaseDataSample` 支持 `BaseDataElement` 的上述所有使用方式，唯一的不同点在于每个算法库内会对齐样本数据的字段进行规约和校验。

## 命名规约

为了保持不同任务数据之间的兼容性和统一性，我们建议抽象数据接口中相同的数据使用统一的字段命名，并在下文的命名列表中进行约束。
