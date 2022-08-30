# 抽象数据接口

在模型的训练/测试过程中，组件之间往往有大量的数据需要传递，不同的算法需要传递的数据经常是不一样的，
例如，训练单阶段检测器需要获得数据集的标注框（ground truth bounding boxes）和标签（ground truth box labels），训练 Mask R-CNN 时还需要实例掩码（instance masks）。
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

## 设计

一个算法库中的数据可以被归类成具有不同性质的数据元素 （data element）。一张图片的检测框标注、模型在这张图片上预测出的检测框、以及一张图片的所有标注信息（包含检测框、语义分割图等）都可以被抽象成数据元素。因此，MMEngine 定义了数据元素的基类 `BaseDataElement` 和它所提供的基本的增/删/改/查等基本功能。基于 MMEngine 的算法库可以定义由 `BaseDataElement` 派生而来的数据元素封装，作为该库的组件之间的抽象数据接口。

一种典型数据元素是某一算法任务上的预测数据或标注：例如检测框，实例掩码，语义分割掩码，和图像标签等。这些数据元素可以进一步区分为实例级别，像素级别，和标签级别。这些类型各有自己的特点。因此，MMEngine 基于 `BaseDataElement` 派生出了 3 类数据结构来封装不同类型的标注数据或者模型的预测结果：`InstanceData`, `PixelData`, 和 `LabelData`。这些接口可以被用于模型内各个模块之间的数据传递。因为标注数据和预测数据往往具有相似的性质（例如模型的预测框和标注框具有相同的性质），MMEngine 使用相同的抽象数据接口来封装预测数据和标注数据，并推荐使用命名来区分他们，如使用 `gt_instances` 和 `pred_instances` 来区分标注和预测的实例数据。

算法库中另一种常见的数据元素是一个训练样本（例如一张图片）的所有标注和预测构成的数据元素。一般情况下，一张图片可以同时有多种类型的标注和/或预测（例如，同时拥有像素级别的语义分割标注和实例级别的检测框标注）。 一个训练样本（例如一张图片）的所有标注和预测经常在 dataset，model，visualizer，和 evaluator 组件之间被传递。为了简化组件之间的接口，我们可以将他们当作一个大的数据元素并对他们进行封装，这类数据元素在 OpenMMLab 算法库中一般被称为 `XXDataSample`。
因此，类似于 `nn.Module` 的派生类内部可以拥有类型为 `nn.Module` 的属性，`BaseDataElement` 也允许封装 `BaseDataElement` 作为它的属性。这样的类一般在算法库中封装一个样本的全体数据， 并且**它的属性一般会是各种类型的数据元素**。例如，MMDetection 由 `BaseDataElement` 派生出了 `DetDataSample` 来封装该算法库中一个样本的标注与预测的全部数据元素，`DetDataSample` 的属性一般是 `InstanceData`。

他们的关系如下图所示

![data_element](https://user-images.githubusercontent.com/40779233/163569298-756ddb51-7e18-4979-88e7-60cdb99c7b08.jpeg)

为了保证抽象数据接口内数据的完整性，抽象数据接口内部有两种数据，除了被封装的数据（data）本身，还有一种是数据的元信息（metainfo），例如图片大小和 ID 等。
两种类型的抽象数据接口都可以作为 Python 类去使用和操作他们的属性。同时，因为他们封装的数据大多是 Tensor，他们也提供了类似 Tensor 的基础操作。

## 用法

### BaseDataElement

MMEngine 为数据元素的封装提供了一个基类 `BaseDataElement`。
基于 `BaseDataElement`，MMEngine 还实现了 `InstanceData`， `PixelData`， `LabelData` 三个典型的子类，封装了实例级别，像素级别，标签级别的数据元素，并针对他们的数据特性支持了一些额外的功能。

1. `InstanceData`：封装检测框、框对应的标签和实例掩码、甚至关键点等实例级别数据，`InstanceData` 假定它封装的数据具有相同的长度 N，N 代表实例的个数，并基于此假定对数据进行校验、支持对实例进行索引和拼接。
2. `PixelData`：封装逐像素级别的数据，如语义分割图和深度图等。`PixelData` 假定它封装的数据有相同的长度和宽度，第一和第二维为图片的长宽，第三维为通道数。`PixelData` 基于此假定对数据进行校验、支持对实例进行空间维度的索引和各维度的拼接。
3. `LabelData`：封装标签数据，如场景分类标签等。

`BaseDataElement` 中存在两种类型的数据，一种是 `data` 类型，如标注框、框的标签、和实例掩码等；另一种是 `metainfo` 类型，包含数据的元信息以确保数据的完整性，如 `img_shape`, `img_id` 等数据所在图片的一些基本信息，方便可视化等情况下对数据进行恢复和使用。用户在创建 `BaseDataElement` 的过程中需要对这两类属性的数据进行显式地区分和声明。

#### 1. 数据元素的创建

`BaseDataElement` 的 data 参数可以直接通过 `key=value` 的方式自由添加，metainfo 的字段需要显式通过关键字 `metainfo` 指定。
`BaseDataElement` 支持 `from_dict` 接口，支持从 dict 构建 `BaseDataElement`。

```python
# 可以声明一个空的 object
gt_instances = BaseDataElement()

bboxes = torch.rand((5, 4))  # 假定 bboxes 是一个 Nx4 维的 tensor，N 代表框的个数
scores = torch.rand((5,))  # 假定框的分数是一个 N 维的 tensor，N 代表框的个数
img_id = 0  # 图像的 ID
H = 800  # 图像的高度
W = 1333  # 图像的宽度

# 直接设置 BaseDataElement 的 data 参数
gt_instances = BaseDataElement(bboxes=bboxes, scores=scores)

# 显式声明来设置 BaseDataElement 的参数 metainfo
gt_instances = BaseDataElement(
    bboxes=bboxes,
    scores=scores,
    metainfo=dict(img_id=img_id, img_shape=(H, W)))

# 通过 from_dict，传入字典将设置 BaseDataElement 的参数 data
BaseDataElement.from_dict(dict(bboxes=bboxes, scores=scores))
BaseDataElement.from_dict(
    dict(bboxes=bboxes, scores=scores),
    metainfo=dict(img_id=img_id, img_shape=(H, W)))
```

#### 2. `new` 与 `clone` 函数

用户可以使用 `new()` 函数通过已有的数据接口创建一个具有相同状态和数据的抽象数据接口。用户可以在创建新 `BaseDataElement` 时设置 metainfo 和 data，使得新的 BaseDataElement 有相同的状态但是不同的数据。
也可以直接使用 `clone()` 来获得一份深拷贝，`clone()` 函数的行为与 PyTorch 中 Tensor 的 `clone()` 参数保持一致。

```python
gt_instances = BaseDataElement()

# 可以在创建新 `BaseDataElement` 时设置 metainfo 和 data，使得新的 BaseDataElement 有不同的数据但是数据在相同的 device 上
gt_instances1 = gt_instance.new(
    bboxes=torch.rand((5, 4)),
    scores=torch.rand((5,)),
    metainfo=dict(img_id=1, img_shape=(640, 640)),
)

# 也可以通过 `clone` 构建一个新的 object，新的 object 会拥有和 gt_instance 相同的 data 和 metainfo 内容以及状态。
gt_instances2 = gt_instances1.clone()
```

#### 3. 属性的增加与查询

用户可以像增加类属性那样增加 `BaseDataElement` 的属性，此时数据会被**当作 data 类型**增加到 `BaseDataElement` 中。
如果需要增加 metainfo 属性，用户应当使用 `set_metainfo`。
用户可以可以通过 `keys`，`values`，和 `items` 来访问只存在于 data 中的键值，也可以通过 `metainfo_keys`，`metainfo_values`，和`metainfo_items` 来访问只存在于 metainfo 中的键值。
用户还能通过 `all_keys`，`all_values`， `all_items` 来访问 `BaseDataElement` 的所有的属性并且不区分他们的类型。

**注意：**

1. `BaseDataElement` 不支持 metainfo 和 data 属性中有同名的字段，所以用户应当避免 metainfo 和 data 属性中设置相同的字段，否则 `BaseDataElement` 会报错。
2. 考虑到 `InstanceData` 和 `PixelData` 支持对数据进行切片操作，为了避免 `[]` 用法的不一致，同时减少同种需求的不同方法，`BaseDataElement` 不支持像字典那样访问和设置它的属性，所以类似 `BaseDataElement[name]` 的取值赋值操作是不被支持的。

```python
gt_instances = BaseDataElement()
# 设置 gt_instances 的 meta 字段，img_id 和 img_shape 会被作为 metainfo 的字段成为 gt_instances 的属性
gt_instances.set_metainfo(dict(img_id=9, img_shape=(100, 100))
assert 'img_shape' in gt_instances.metainfo_keys()
# 'img_shape' 是 gt_instances 的属性
assert 'img_shape' in gt_instances
# img_shape 不是 gt_instances 的 data 字段
assert 'img_shape' not in gt_instances.all_keys()
# 通过 all_keys 来访问所有属性
assert 'img_shape' in gt_instances.all_keys()
# 访问类属性一样访问 'img_shape'
print(gt_instances.img_shape)

# 直接设置 gt_instance 的 scores 属性，默认该数据属于 data
gt_instances.scores = torch.rand((5,))
assert 'scores' in gt_instances.items()
# 'scores' 是 gt_instances 的属性
assert 'scores' in gt_instances
# 通过 all_keys 来访问所有属性
assert 'scores' in gt_instances.all_keys()
# scores 不是 gt_instances 的 metainfo 字段
assert 'scores' not in gt_instances.metainfo_keys()
# 访问类属性一样访问 'scores'
print(gt_instances.scores)

# 设置 gt_instances 的 data 字段 bboxes
gt_instances.bboxes = torch.rand((5, 4))
assert 'bboxes' in gt_instances.items()
# 'bboxes' 是 gt_instances 的属性
assert 'bboxes' in gt_instances
# 通过 all_keys 来访问所有属性
assert 'bboxes' in gt_instances.all_keys()
# bboxes 不是 gt_instances 的 metainfo 字段
assert 'bboxes' not in gt_instances.metainfo_keys()
# 访问类属性一样访问 'bboxes'
print(gt_instances.bboxes)

for k, v in gt_instances.all_items():
    print(f'{k}: {v}')  # 包含 img_shapes， img_id， bboxes，scores

for k, v in gt_instances.metainfo_items():
    print(f'{k}: {v}')  # 包含 img_shapes， img_id

for k, v in gt_instances.items():
    print(f'{k}: {v}')  # 包含 bboxes，scores
```

#### 4. 属性的删改

`BaseDataElement` 支持用户可以像使用一个类一样对它的属性进行删改
同时， `BaseDataElement` 支持 `get` 来允许在访问不到变量时设置默认值，也支持 `pop` 在在访问属性后删除属性。

```python
gt_instances = BaseDataElement(
    bboxes=torch.rand((6, 4)), scores=torch.rand((6,)),
    metainfo=dict(img_id=0, img_shape=(640, 640))，
)

# 对类的属性进行修改
gt_instances.img_shape = (1280, 1280)
gt_instances.img_shape  # (1280, 1280)
gt_instances.bboxes = gt_instances.bboxes * 2

# 提供了可设置默认值的获取方式 get
gt_instances.get('img_shape', None)  # (640， 640)
gt_instances.get('bboxes', None)    # 6x4 tensor

# 属性的删除
del gt_instances.img_shape
del gt_instances.bboxes
assert 'img_shape' not in gt_instances
assert 'bboxes' not in gt_instances

# 提供了便捷的属性删除和访问操作 pop
gt_instances.pop('img_shape', None)  # None
gt_instances.pop('bboxes', None)  # None
```

#### 5. 类张量操作

用户可以像 torch.Tensor 那样对 `BaseDataElement` 的 data 进行状态转换，目前支持 `cuda`， `cpu`， `to`， `numpy` 等操作。
其中，`to` 函数拥有和 `torch.Tensor.to()` 相同的接口，使得用户可以灵活地将被封装的 tensor 进行状态转换。
**注意：** 这些接口只会处理类型为 np.array，torch.Tensor，或者数字的序列，其他属性的数据（如字符串）会被跳过处理。

```python
# 将所有 data 转移到 GPU 上
cuda_instances = gt_instances.cuda()
cuda_instances = gt_instancess.to('cuda:0')

# 将所有 data 转移到 cpu 上
cpu_instances = cuda_instances.cpu()
cpu_instances = cuda_instances.to('cpu')

# 将所有 data 变成 FP16
fp16_instances = cuda_instances.to(
    device=None, dtype=torch.float16, non_blocking=False, copy=False,
    memory_format=torch.preserve_format)

# 阻断所有 data 的梯度
cpu_instances = cuda_instances.detach()

# 转移 data 到 numpy array
np_instances = cpu_instances.numpy()
```

#### 6. 属性的展示

`BaseDataElement` 还实现了 `__repr__`，因此，用户可以直接通过 `print` 函数看到其中的所有数据信息。
同时，为了便捷开发者 debug，`BaseDataElement` 中的属性都会添加进 `__dict__` 中，方便用户在 IDE 界面可以直观看到 `BaseDataElement` 中的内容。
一个完整的属性展示如下

```python
>>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
>>> instance_data = BaseDataElement(metainfo=img_meta)
>>> instance_data.det_labels = torch.LongTensor([0, 1, 2, 3])
>>> instance_data.det_scores = torch.Tensor([0.01, 0.1, 0.2, 0.3])
>>> print(results)
<BaseDataElement(
  META INFORMATION
img_shape: (800, 1196, 3)
pad_shape: (800, 1216, 3)
  DATA FIELDS
shape of det_labels: torch.Size([4])
shape of det_scores: torch.Size([4])
) at 0x7f84acd10f90>
```

### DataSample

基于 `BaseDataElement`，下游算法库可以定义 `DetDataSample`，并且定义 3 个 property：proposals，gt_instances，pred_instances，并约束他们的类型。

```python
class DetDataSample(BaseDataElement):

    @property
    def proposals(self):
        return self._proposals

    @proposals.setter
    def proposals(self, value):
        self.set_field(value, '_proposals', dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        del self._proposals

    @property
    def gt_instances(self):
        """Ground truth instances of an image"""
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value):
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def pred_instances(self):
        """Predicted instances of an image"""
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value):
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    @property
    def proposals(self):
        """Region proposals"""
        return self._proposals

    @proposals.setter
    def proposals(self, value):
        self.set_field(value, '_proposals', dtype=InstanceData)

    @proposals.deleter
    def proposalss(self):
        del self._proposals

```

`DetDataSample` 的用法如下所示，在数据类型不符合要求的时候（例如用 `torch.Tensor` 而非 `InstanceData` 定义 proposals 时） ，`DetDataSample` 就会报错。

```python
a = DetDataSample()

a.proposals = InstanceData(data=dict(bboxes=torch.rand((5,4))))

assert 'proposals' in a
print(a.proposals)

del a.proposals
assert 'proposals' not in a
```

### 对接口的简化

下面以 MMDetection 为例更具体地说明 OpenMMLab 的算法库将如何迁移使用抽象数据接口，以简化模块和组件接口的。我们假定 MMDetection 和 MMEngine 中实现了 DetDataSample 和 InstanceData。

#### 1. 组件接口的简化

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

#### 2. 模块接口的简化

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

`InstanceData` 可以封装实例的框、分数、和掩码，将 `HungarianAssigner` 的核心参数简化成 `pred_instances`，`gt_instancess`，和 `gt_instances_ignore`
使得 `HungarianAssigner` 和 `MaskHungarianAssigner` 可以合并成一个通用的 `HungarianAssigner`。

```python
class HungarianAssigner(BaseAssigner):

    def assign(self,
               pred_instances,
               gt_instancess,
               gt_instances_ignore=None,
               eps=1e-7):
```

## 命名规约

为了保持不同任务数据之间的兼容性和统一性，我们建议抽象数据接口中对相同的数据使用统一的字段命名。
在本文档中，我们暂时性地在下文列举一些算法方向的样本数据封装及其属性约定，后续会有更全面的文档来描述命名规约。
用户在使用各算法库抽象接口的过程中，可以假定对应的数据（如有）在样本数据封装中是按照如下约定进行命名的。

### ClsDataSample

- gt_label (LabelData): 数据的分类标签
- pred_label (LabelData): 模型对数据的分类预测结果

### DetDataSample

- pred_instances (InstanceData): 模型预测的实例
- gt_instances (InstanceData): 标注的实例
- gt_sem_seg (PixelData): 语义分割的标注
- pred_sem_seg (PixelData): 语义分割任务的模型预测
- gt_panoptic_seg (PixelData): 全景分割的标注
- pred_panoptic_seg (PixelData): 全景分割任务的模型预测
- proposals (InstanceData): 用于双阶段检测器的候选框提名
- ignored_instances (InstanceData): 在训练中应当被忽视的实例

### SegDataSample

- gt_sem_seg (PixelData): 语义分割的标注
- pred_sem_seg (PixelData): 语义分割任务的模型预测
