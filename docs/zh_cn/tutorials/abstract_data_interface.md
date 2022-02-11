# 抽象数据接口

MMEngine 定义了一套抽象的数据接口来封装模型运行过程中产生的各种数据。
抽象数据接口会被用于算法库中 dataset，model，visualizer，和 evaluator 组件之间，或者 model 内各个模块之间的数据传递。
抽象数据接口实现了基本的增/删/改/查功能，同时支持不同设备之间的迁移。
基于 MMEngine 的算法库可以继承这套抽象数据接口并实现自己的抽象数据接口来适应不同算法中数据的特点与实际需要。
为了保持不同任务数据之间的兼容性和统一性，我们建议抽象数据接口中相同的数据使用统一的字段命名，并在本文档的命名列表中进行约束。

## 设计

一个算法库中的数据可以分成两种类型，一种是基础的数据元素，另一种是样本数据。相应地，MMEngine 为这两种数据类型分别提供了一个抽象的封装接口。

1. 基础数据元素的封装： 基础的数据元素指的是某一算法任务上的预测数据或标注。因为这些数据往往具有相似的性质（例如模型的预测框和标注框具有相同的数据特性），因此，MMEngine 使用相同的抽象数据接口来封装他们。 我们将数据标注或者模型的预测数据区分为实例级别，像素级别，和标签级别。这些类型各有自己的特点，因此，MMEngine 定义了基础数据元素的基类 `BaseDataType`，并由此派生出了 3 类数据结构来封装不同类型的标注数据或者模型的预测结果：`InstanceData`, `PixelData`, 和 `LabelData`。这些接口将被用于模型内哥哥模块之间的数据传递。

2. 样本数据的封装：一个训练样本（例如一张图片）的所有标注和预测被概称为样本数据。一般情况下，一张图片可以同时有多种类型的标注和/或预测（例如，同时拥有像素级别的语义分割标注，标签级的场景分类标注，和实例级别的检测框标注）。因此，MMEngine 定义了 `BaseDataSample`，OpenMMLab 算法库将基于 `BaseDataSample` 实现自己的抽象数据接口，来封装一个算法库中单个样本的所有相关数据，作为 dataset，model，visualizer，和 evaluator 组件之间的数据接口。

为了保证抽象数据接口内数据的完整性，抽象数据接口内部有两种数据，除了被封装的数据本身，还有一种是数据的元信息，例如图片大小和 ID 等。
两种类型的抽象数据接口都可以像 Python 的基础数据结构 dict 一样被使用。同时，因为他们封装的数据大多是 Tensor，他们也提供了类似 Tensor 的基础操作。

## 用法

### BaseDataType

以 `InstanceData` 为例，基础数据元素的使用有如下用法

1. 数据元素的创建

```python
bboxes = torch.tensor((5, 4))  # 假定 bboxes 是一个 Nx4 维的 tensor，N代表框的个数
scores = torch.tensor((5,))  # 假定框的分数是一个 N 维的 tensor，N代表框的个数
img_id = 0  # 图像的 ID
H = img_height  # 图像的高度
W = img_width  # 图像的宽度

# 显式声明 InstanceData 的参数 meta_info 和 data
gt_instances = InstanceData(
    meta_info=dict(img_id=img_id, img_shape=(H, W)),
    data=dict(bboxes=bboxes, scores=scores))

# 不显式声明的时候，传入字典将设置 InstanceData 的参数 meta_info
gt_instances = InstanceData(dict(img_id=img_id, img_shape=(H, W)))

# 也可以声明一个空的 object
gt_instances = InstanceData()
```

2. 通过已有的数据接口创建一个具有相同状态的抽象数据接口

```python
gt_instances = InstanceData()

gt_instances1 = gt_instance.new(
    meta_info=dict(img_id=img_id, img_shape=(H, W)),
    data=dict(bboxes=bboxes, scores=scores)
)

# 也可以声明一个新的 object，新的 object 会拥有和 gt_instance 相同的属性
gt_instances1 = gt_instances.new()
```

3. 属性的增删改查

```python
gt_instances = InstanceData()

# 设置 gt_instances 的 meta 信息
gt_instances.set_meta_info(dict(img_id=img_id, img_shape=(H, W)))

# 设置 gt_instances 的 data
gt_instances.set_data(dict(bboxes=bboxes))

# 直接设置 gt_instance 的 scores 属性，默认该数据属于 data
gt_instances.scores = scores

# 提供了类字典的访问方式
assert 'img_id' in gt_instances
print(gt_instances.img_shape)
print(gt_instances['img_shape'])
print(gt_instances.get('img_shape', None))
print(gt_instances.pop('img_shape', None))

# 属性修改
gt_instances.img_shape = (1280, 1280)
gt_instances['img_shape'] = (640, 640)
gt_instances.set_data(dict(labels=[1,2,3]))

# 属性的遍历：
assert 'bboxes' in gt_instances
assert 'bboxes' in gt_instances.keys()
assert 'scores' in gt_instances.keys()
assert (640, 640) in gt_instaces.meta_info_values()
assert 'img_shape' in gt_instaces.meta_info_keys()

# 属性的删除
del gt_instances['bboxes']
del gt_instances.scores
```

4. 属性的状态变化

```python
# 转移到 GPU 上
cuda_instances = gt_instances.cuda()
cuda_instances = gt_instancess.to('cuda:0')

# 转移到 cpu 上
cpu_instances = cuda_instances.cpu()
cpu_instances = cuda_instances.to('cpu')

# 阻断 梯度
cpu_instances = cuda_instances.detach()

# 转移到 numpy
np_instances = cpu_instances.numpy()
```

完整的用例展示包括

```
>>> from mmdet.core import GeneralData
>>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
>>> instance_data = GeneralData(meta_info=img_meta)
>>> img_shape in instance_data
True
>>> instance_data.det_labels = torch.LongTensor([0, 1, 2, 3])
>>> instance_data["det_scores"] = torch.Tensor([0.01, 0.1, 0.2, 0.3])
>>> print(results)
<GeneralData(
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
