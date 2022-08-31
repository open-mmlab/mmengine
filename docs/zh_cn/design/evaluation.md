# 模型精度评测

在模型验证和模型测试中，通常需要对模型精度做定量评测。在 MMEngine 中实现了[评测指标](mmengine.evaluator.BaseMetric)和[评测器](mmengine.evaluator.Evaluator)来完成这一功能。

**评测指标** 根据模型的输入数据和预测结果，完成特定指标下模型精度的计算。评测指标与数据集之间相互解耦，这使得用户可以任意组合所需的测试数据和评测指标。如 [COCOMetric](Todo:coco-metric-doc-link) 可用于计算 COCO 数据集的 AP，AR 等评测指标，也可用于其他的目标检测数据集上。
**评测器** 是评测指标的上层模块，通常包含一个或多个评测指标。评测器的作用是在模型评测时完成必要的数据格式转换，并调用评测指标计算模型精度。评测器通常由[执行器](../tutorials/runner.md)或测试脚本构建，分别用于在线评测和离线评测。

## 模型精度评测流程

通常，模型精度评测的过程如下图所示。

**在线评测**：测试数据通常会被划分为若干批次（batch）。通过一个循环，依次将每个批次的数据送入模型，得到对应的预测结果，并将测试数据和模型预测结果送入评测器。评测器会调用评测指标的 `process()` 方法对数据和预测结果进行处理。当循环结束后，评测器会调用评测指标的 `evaluate()` 方法，可计算得到对应指标的模型精度。

**离线评测**：与在线评测过程类似，区别是直接读取预先保存的模型预测结果来进行评测。评测器提供了 `offline_evaluate` 接口，用于在离线方式下调用评测指标来计算模型精度。为了避免同时处理大量数据导致内存溢出，离线评测时会将测试数据和预测结果分成若干个块（chunk）进行处理，类似在线评测中的批次。

<div align="center">
    <img src="https://user-images.githubusercontent.com/15977946/187579113-279f097c-3530-40c4-9cd3-1bb0ce2fa452.png" width="500"/>
</div>

## 增加自定义评测指标

在 OpenMMLab 的各个算法库中，已经实现了对应方向的常用评测指标。如 MMDetection 中提供了 COCO 评测指标，MMClassification 中提供了 Accuracy、F1Score 等评测指标等。

用户也可以增加自定义的评测指标。在实现自定义评测指标时，需要继承 MMEngine 中提供的评测指标基类 [BaseMetric](mmengine.evaluator.BaseMetric)，并实现对应的抽象方法。

### 评测指标基类

评测指标基类 `BaseMetric` 是一个抽象类，具有以下 2 个抽象方法：

- `process()`: 处理每个批次的测试数据和模型预测结果。处理结果应存放在 `self.results` 列表中，用于在处理完所有测试数据后计算评测指标。
- `compute_metrics()`: 计算评测指标，并将所评测指标存放在一个字典中返回。

其中，`compute_metrics()` 会在 `evaluate()` 方法中被调用；后者在计算评测指标前，会在分布式测试时收集和汇总不同 rank 的中间处理结果。

需要注意的是，`self.results` 中存放的具体类型取决于评测指标子类的实现。例如，当测试样本或模型输出数据量较大（如语义分割、图像生成等任务），不宜全部存放在内存中时，可以在 `self.results` 中存放每个批次计算得到的指标，并在 `compute_metrics()` 中汇总；或将每个批次的中间结果存储到临时文件中，并在 `self.results` 中存放临时文件路径，最后由 `compute_metrics()` 从文件中读取数据并计算指标。

### 自定义评测指标类

我们以实现分类正确率（Classification Accuracy）评测指标为例，说明自定义评测指标的方法。

首先，评测指标类应继承自 `BaseMetric`，并应加入注册器 `METRICS` (关于注册器的说明请参考[相关文档](../tutorials/registry.md))。

`process()` 方法有 2 个输入参数，分别是一个批次的测试数据样本 `data_batch` 和模型预测结果 `predictions`。我们从中分别取出样本类别标签和分类预测结果，并存放在 `self.results` 中。

`compute_metrics()` 方法有 1 个输入参数 `results`，里面存放了所有批次测试数据经过 `process()` 方法处理后得到的结果。从中取出样本类别标签和分类预测结果，即可计算得到分类正确率 `acc`。最终，将计算得到的评测指标以字典的形式返回。

此外，我们建议在子类中为类属性 `default_prefix` 赋值。如果在初始化参数（即 config 中）没有指定 `prefix`，则会自动使用 `default_prefix` 作为评测指标名的前缀。同时，应在 docstring 中说明该评测指标类的 `default_prefix` 值以及所有的返回指标名称。

具体的实现如下：

```python
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

import numpy as np


@METRICS.register_module()  # 将 Accuracy 类注册到 METRICS 注册器
class Accuracy(BaseMetric):
    """ Accuracy Evaluator

    Default prefix: ACC

    Metrics:
        - accuracy (float): classification accuracy
    """

    default_prefix = 'ACC'  # 设置 default_prefix

    def process(self, data_batch: Sequence[dict], predictions: Sequence[dict]):
        """Process one batch of data and predictions. The processed
        Results should be stored in `self.results`, which will be used
        to computed the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from
                the model.
        """

        # 取出分类预测结果和类别标签
        result = {
            'pred': predictions['pred_label'],
            'gt': data_batch['data_sample']['gt_label']
        }

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
