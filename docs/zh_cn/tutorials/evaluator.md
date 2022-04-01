# 评测器（Evaluator）

在模型验证和模型测试中，通常需要对模型精度做定量评测。在 MMEngine 中实现了[评测器](Todo:evaluator-doc-link)来完成这一功能。评测器可以根据模型的输入数据和预测结果，计算特定的评测指标（Metric）。评测器与数据集之间相互解耦，这使得用户可以任意组合所需的测试数据和评测器。如 [COCOEvaluator](Todo:coco-evaluator-doc-link) 可用于计算 COCO 数据集的 AP，AR 等评测指标，也可用于其他的目标检测数据集上。

## 模型精度评测

使用评测器计算模型精度的过程如下图所示。

测试数据通常会被划分为若干批次（batch）。通过一个循环，依次将每个批次的数据送入模型，得到对应的预测结果，并将预测结果连同模型的输入数据一起通过评测器的 `process()` 方法送入评测器。当循环结束后，再调用评测器的 `evaluate()` 方法，即可计算得到对应的评测指标。

在实际使用中，这些操作均由任务执行器完成。用户只需要在配置文件中选择要使用的评测器并配置相应参数即可。

<div align="center">
    <img src="https://user-images.githubusercontent.com/15977946/154652635-f4bda588-9f94-462f-b68f-b900690e6215.png"/>
</div>


### 在配置文件中配置评测器

在配置文件中配置评测器时，需要指定评测器的类别、参数以及调用方式等。其中，调用方式通常针对模型验证阶段，包括调用评测器的间隔时间单位（epoch 或 iteration）、间隔时间、主要评测指标（即筛选最佳 checkpoint 所依据的指标）等。

例如，用户希望在模型验证时使用 COCO 评测器，每 10 epoch 评测一次，并以 AP 作为主要评测指标，对应的配置文件部分如下：

```python
validation_cfg=dict(
    evaluator=dict(type='COCO'),  # 使用 COCO 评测器，无参数
    main_metric='AP',  # 主要评测指标为 AP
    interval=10,  # 每 10 epoch 评测一次
    by_epoch=True,
)
```

### 使用多个评测器

评测器支持组合使用。用户可以通过配置多个评测器，在模型验证或模型测试阶段同时计算多个评测指标。使用多个评测器时，只需要在配置文件里将所有评测器的配置写在一个列表里即可：

```python
validation_cfg=dict(
    evaluator=[
        dict(type='Accuracy', top_k=1),  # 使用分类正确率评测器
        dict(type='F1Score')  # 使用 F1_score 评测器
    ],
    main_metric='accuracy'
    interval=10,
    by_epoch=True,
)
```

使用多个评测器时，可能出现评测指标同名的情况。比如，在下面的例子中使用了 2 个 `COCOEvaluator` 分别对检测框和关键点的预测结果进行评测，它们的评测指标都包括 `AP`，`AR` 等。为了避免同名评测指标引发歧义，`Evaluator` 中支持通过 `prefix` 参数为评测指标名增加前缀。通常，一个 `Evaluator` 会有默认的前缀，用户也可以在配置文件中进行指定。

```python
validation_cfg=dict(
    evaluator=[
        dict(type='COCO', iou_type='bbox'),  # 使用默认前缀 `COCO`
        dict(type='COCO', iou_type='keypoints', prefix='COCOKpts')  # 自定义前缀 `COCOKpts`
    ],
    # 指定使用前缀为 COCO 的 AP 为主要评测指标
    # 在没有重名指标歧义的情况下，此处可以不写前缀，只写评测指标名
    main_metric='COCO/AP',
    interval=10,
    by_epoch=True,
)
```

## 增加自定义评测器

在 OpenMMLab 的各个算法库中，已经实现了对应方向的常用评测器。如 MMDetection 中提供了 COCO 评测器，MMClassification 中提供了 Accuracy、F1Score 等评测器等。

用户也可以根据自身需求，增加自定义的评测器。在实现自定义评测器时，用户需要继承 MMEngine 中提供的评测器基类 [BaseEvaluator](Todo:baseevaluator-doc-link)，并实现对应的抽象方法。

### 评测器基类

评测器基类 `BaseEvaluator` 是一个抽象类，具有以下 2 个抽象方法：

- `process()`: 处理每个批次的测试数据和模型预测结果。处理结果应存放在 `self.results` 列表中，用于在处理完所有测试数据后计算评测指标。
- `compute_metrics()`: 计算评测指标，并将所评测指标存放在一个字典中返回。

其中，`compute_metrics()` 会在 `evaluate()` 方法中被调用；后者在计算评测指标前，会在分布式测试时收集和汇总不同 rank 的中间处理结果。而 `process()` 和 `evaluate()` 都会由任务执行器调用。因此，用户只需要在继承 `BaseEvaluator` 后实现 `process()` 和 `compute_metrics()` 方法即可。

需要注意的是，`self.results` 中存放的具体类型取决于自定义评测器类的实现。例如，当测试样本或模型输出数据量较大（如语义分割、图像生成等任务），不宜全部存放在内存中时，可以在 `self.results` 中存放每个批次计算得到的指标，并在 `compute_metrics()` 中汇总；或将每个批次的中间结果存储到临时文件中，并在 `self.results` 中存放临时文件路径，最后由 `compute_metrics()` 从文件中读取数据并计算指标。

### 自定义评测器类

我们以实现分类正确率（Classification Accuracy）评测器为例，说明实现自定义评测器的方法。

首先，自定义评测器类应继承自 `BaseEvaluator`，并应加入注册器 `EVALUATORS` (关于注册器的说明请参考[相关文档](docs\zh_cn\tutorials\registry.md))。

 `process()` 方法有 2 个输入参数，分别是一个批次的测试数据样本 `data_batch` 和模型预测结果 `predictions`。我们从中分别取出样本类别标签和分类预测结果，并存放在 `self.results` 中。

`compute_metrics()` 方法有 1 个输入参数 `results`，里面存放了所有批次测试数据经过 `process()` 方法处理后得到的结果。从中取出样本类别标签和分类预测结果，即可计算得到分类正确率 `acc`。最终，将计算得到的评测指标以字典的形式返回。

此外，我们建议在子类中为类属性 `default_prefix` 赋值。如果在初始化参数（即 config 中）没有指定 `prefix`，则会自动使用 `default_prefix` 作为评测指标名的前缀。同时，应在 docstring 中说明该评测器的 `default_prefix` 值以及所有的评测指标。

具体的实现如下：

```python
from mmengine.evaluator import BaseEvaluator
from mmengine.registry import EVALUATORS

import numpy as np

@EVALUATORS.register_module()
class Accuracy(BaseEvaluator):
    """ Accuracy Evaluator

    Default prefix: ACC

    Metrics:
        - accuracy (float): classification accuracy
    """

    default_prefix = 'ACC'

    def process(self, data_batch: Sequence[Tuple[Any, BaseDataElement]],
                predictions: Sequence[BaseDataElement]):
        """Process one batch of data and predictions. The processed
        Results should be stored in `self.results`, which will be used
        to computed the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, BaseDataElement]]): A batch of data
                from the dataloader.
            predictions (Sequence[BaseDataElement]): A batch of outputs from
                the model.
        """

        # 取出分类预测结果和类别标签
        result = dict(
            'pred': predictions.pred_label,
            'gt': data_samples.gt_label
        )

        # 将当前 batch 的结果存进 self.results
        self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        # 汇总所有样本的分类预测结果和类别标签
        preds = np.concatenate([res['pred'] for res in results])
        gts = np.concatenate([res['gt'] for res in results])

        # 计算分类正确率
        acc = (preds == gts).sum() / preds.size

        # 返回评测指标结果
        return {'accuracy': acc}

```
