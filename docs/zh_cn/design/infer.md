# 推理接口

基于 MMEngine 开发时，我们通常会为具体算法定义一个配置文件，根据配置文件去构建[执行器](./runner.md)，执行训练、测试流程，并保存训练好的权重。基于训练好的模型进行推理时，通常需要执行以下步骤：

1. 基于配置文件构建模型
2. 加载模型权重
3. 搭建数据预处理流程
4. 执行模型前向推理
5. 可视化推理结果
6. 输出推理结果

对于此类标准的推理流程，MMEngine 提供了统一的推理接口，并且建议用户基于这一套接口规范来开发推理代码。

## 使用样例

### 定义推理器

**基于 `BaseInferencer` 实现自定义的推理器**

```python
from mmengine.infer import BaseInferencer

class CustomInferencer(BaseInferencer)
    ...
```

具体细节参考[开发规范](#推理接口开发规范)

### 构建推理器

**基于配置文件路径构建推理器**

```python
cfg = 'path/to/config.py'
weight = 'path/to/weight.pth'

inferencer = CustomInferencer(model=cfg, weight=weight)
```

**基于配置类实例构建推理器**

```python
from mmengine import Config

cfg = Config.fromfile('path/to/config.py')
weight = 'path/to/weight.pth'

inferencer = CustomInferencer(model=cfg, weight=weight)
```

**基于 model-index 中定义的 model name 构建推理器**，以 MMDetection 中的 [atss 检测器为例](https://github.com/open-mmlab/mmdetection/blob/31c84958f54287a8be2b99cbf87a6dcf12e57753/configs/atss/metafile.yml#L22)，model name 为 `atss_r50_fpn_1x_coco`，由于 model-index 中已经定义了 weight 的路径，因此可以不配置 weight 参数。

```python
inferencer = CustomInferencer(model='atss_r50_fpn_1x_coco')
```

### 执行推理

**推理单张图片**

```python
# 输入为图片路径
img = 'path/to/img.jpg'
result = inferencer(img)

# 输入为读取的图片(类型为 np.ndarray)
img = cv2.imread('path/to/img.jpg')
result = inferencer(img)

# 输入为 url
img = 'https://xxx.com/img.jpg'
result = inferencer(img)
```

**推理多张图片**

```python
img_dir = 'path/to/directory'
result = inferencer(img_dir)
```

```{note}
OpenMMLab 系列算法库要求 `inferencer(img)` 输出一个 `dict`，其中包含 `visualization: list` 和 `predictions: list` 两个字段，分别对应可视化结果和预测结果。
```

## 推理接口开发规范

inferencer 执行推理时，通常会执行以下步骤：

1. preprocess：输入数据预处理，包括数据读取、数据预处理、数据格式转换等
2. forward: 模型前向推理
3. visualize：预测结果可视化
4. postprocess：预测结果后处理，包括结果格式转换、导出预测结果等

为了优化 inferencer 的使用体验，我们不希望使用者在执行推理时，需要为每个过程都配置一遍参数。换句话说，我们希望使用者可以在不感知上述流程的情况下，简单为 `__call__` 接口配置参数，即可完成推理。

`__call__` 接口会按照顺序执行上述步骤，但是本身却不知道使用者传入的参数需要分发给哪个步骤，因此开发者在实现 `CustomInferencer` 时，需要定义 `preprocess_kwargs`，`forward_kwargs`，`visualize_kwargs`，`postprocess_kwargs` 4 个类属性，每个属性均为一个字符集合（`Set[str]`），用于指定 `__call__` 接口中的参数对应哪个步骤：

```python
class CustomInferencer(BaseInferencer):
    preprocess_kwargs = {'a'}
    forward_kwargs = {'b'}
    visualize_kwargs = {'c'}
    postprocess_kwargs = {'d'}

    def preprocess(self, inputs, batch_size=1, a=None):
        pass

    def forward(self, inputs, b=None):
        pass

    def visualize(self, inputs, preds, show, c=None):
        pass

    def postprocess(self, preds, visualization, return_datasample=False, d=None):
        pass

    def __call__(
        self,
        inputs,
        batch_size=1,
        show=True,
        return_datasample=False,
        a=None,
        b=None,
        c=None,
        d=None):
        return super().__call__(
            inputs, batch_size, show, return_datasample, a=a, b=b, c=c, d=d)
```

上述代码中，`preprocess`，`forward`，`visualize`，`postprocess` 四个函数的 `a`，`b`，`c`，`d` 为用户可以传入的额外参数（`inputs`, `preds` 等参数在 `__call__` 的执行过程中会被自动填入），因此开发者需要在类属性 `preprocess_kwargs`，`forward_kwargs`，`visualize_kwargs`，`postprocess_kwargs` 中指定这些参数，这样 `__call__` 阶段用户传入的参数就可以被正确分发给对应的步骤。分发过程由 `BaseInferencer.__call__` 函数实现，开发者无需关心。

此外，我们需要将 `CustomInferencer` 注册到自定义注册器或者 MMEngine 的注册器中

```python
from mmseg.registry import INFERENCERS
# 也可以注册到 MMEngine 的注册中
# from mmengine.registry import INFERENCERS

@INFERENCERS.register_module()
class CustomInferencer(BaseInferencer):
    ...
```

```{note}
OpenMMLab 系列算法仓库必须将 Inferencer 注册到下游仓库的注册器，而不能注册到 MMEngine 的根注册器（避免重名）。
```

## 核心接口说明：

### `__init__()`

`BaseInferencer.__init__` 已经实现了[使用样例](#构建推理器)中构建推理器的逻辑，因此通常情况下不需要重写 `__init__` 函数。如果想实现自定义的加载配置文件、权重初始化、pipeline 初始化等逻辑，也可以重写 `__init__` 方法。

### `_init_pipeline()`

```{note}
抽象方法，子类必须实现
```

初始化并返回 inferencer 所需的 pipeline。pipeline 用于单张图片，类似于 OpenMMLab 系列算法库中定义的 `train_pipeline`，`test_pipeline`。使用者调用 `__call__` 接口传入的每个 `inputs`，都会经过 pipeline 处理，组成 batch data，然后传入 `forward` 方法。

### `_init_collate()`

初始化并返回 inferencer 所需的 `collate_fn`，其值等价于训练过程中 Dataloader 的 `collate_fn`。`BaseInferencer` 默认会从 `test_dataloader` 的配置中获取 `collate_fn`，因此通常情况下不需要重写 `_init_collate` 函数。

### `_init_visualizer()`

初始化并返回 inferencer 所需的 `visualizer`，其值等价于训练过程中 `visualizer`。`BaseInferencer` 默认会从 `visualizer` 的配置中获取 `visualizer`，因此通常情况下不需要重写 `_init_visualizer` 函数。

### `preprocess()`

入参：

- inputs：输入数据，由 `__call__` 传入，通常为图片路径或者图片数据组成的列表
- batch_size：batch 大小，由使用者在调用 `__call__` 时传入
- 其他参数：由用户传入，且在 `preprocess_kwargs` 中指定

返回值：

- 生成器，每次迭代返回一个 batch 的数据。

`preprocess` 默认是一个生成器函数，将 `pipeline` 及 `collate_fn` 应用于输入数据，生成器迭代返回的是组完 batch，预处理后的结果。通常情况下子类无需重写。

### `forward()`

入参：

- inputs：输入数据，由 `preprocess` 处理后的 batch data
- 其他参数：由用户传入，且在 `forward_kwargs` 中指定

返回值：

- 预测结果，默认类型为 `List[BaseDataElement]`

调用 `model.test_step` 执行前向推理，并返回推理结果。通常情况下子类无需重写。

### `visualize()`

```{note}
抽象方法，子类必须实现
```

入参：

- inputs：输入数据，未经过预处理的原始数据。
- preds：模型的预测结果
- show：是否可视化
- 其他参数：由用户传入，且在 `visualize_kwargs` 中指定

返回值：

- 可视化结果，类型通常为 `List[np.ndarray]`，以目标检测任务为例，列表中的每个元素应该是画完检测框后的图像，直接使用 `cv2.imshow` 就能可视化检测结果。不同任务的可视化流程有所不同，`visualize` 应该返回该领域内，适用于常见可视化流程的结果。

### `postprocess()`

```{note}
抽象方法，子类必须实现
```

入参：

- preds：模型预测结果，类型为 `list`，列表中的每个元素表示一个数据的预测结果。OpenMMLab 系列算法库中，预测结果中每个元素的类型均为 `BaseDataElement`
- visualization：可视化结果
- return_datasample：是否维持 datasample 返回。`False` 时转换成 `dict` 返回
- 其他参数：由用户传入，且在 `postprocess_kwargs` 中指定

返回值：

- 可视化结果和预测结果，类型为一个字典。OpenMMLab 系列算法库要求返回的字典包含 `predictions` 和 `visualization` 两个 key。

### `__call__()`

入参：

- inputs：输入数据，通常为图片路径、或者图片数据组成的列表。`inputs` 中的每个元素也可以是其他类型的数据，只需要保证数据能够被 [\_init_pipeline](#_init_pipeline) 返回的 `pipeline` 处理即可。当 `inputs` 只含一个推理数据时，它可以不是一个 `list`，`__call__` 会在内部将 `inputs` 包装成列表，以便于后续处理
- return_datasample：是否将 datasample 转换成 `dict` 返回
- batch_size：推理的 batch size，会被进一步传给 `preprocess` 函数
- 其他参数：分发给 `preprocess`、`forward`、`visualize`、`postprocess` 函数的额外参数

返回值：

- `postprocess` 返回的可视化结果和预测结果，类型为一个字典。OpenMMLab 系列算法库要求返回的字典包含 `predictions` 和 `visualization` 两个 key
