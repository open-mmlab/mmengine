# 模型精度评测（Evaluation）

在模型验证和模型测试中，通常需要对模型精度做定量评测。我们可以通过在配置文件中指定评测指标（Metric）来实现这一功能。

## 在模型训练或测试中进行评测

### 使用单个评测指标

在基于 MMEngine 进行模型训练或测试时，用户只需要在配置文件中通过 `val_evaluator` 和 `test_evaluator` 2 个字段分别指定模型验证和测试阶段的评测指标即可。例如，用户在使用 [MMPretrain](https://github.com/open-mmlab/mmpretrain) 训练分类模型时，希望在模型验证阶段评测 top-1 和 top-5 分类正确率，可以按以下方式配置：

```python
val_evaluator = dict(type='Accuracy', top_k=(1, 5))  # 使用分类正确率评测指标
```

关于具体评测指标的参数设置，用户可以查阅相关算法库的文档。如上例中的 [Accuracy 文档](https://mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.evaluation.Accuracy.html#mmpretrain.evaluation.Accuracy)。

### 使用多个评测指标

如果需要同时评测多个指标，也可以将 `val_evaluator` 或 `test_evaluator` 设置为一个列表，其中每一项为一个评测指标的配置信息。例如，在使用 [MMDetection](https://github.com/open-mmlab/mmdetection) 训练全景分割模型时，希望在模型测试阶段同时评测模型的目标检测（COCO AP/AR）和全景分割精度，可以按以下方式配置：

```python
test_evaluator = [
    # 目标检测指标
    dict(
        type='CocoMetric',
        metric=['bbox', 'segm'],
        ann_file='annotations/instances_val2017.json',
    ),
    # 全景分割指标
    dict(
        type='CocoPanopticMetric',
        ann_file='annotations/panoptic_val2017.json',
        seg_prefix='annotations/panoptic_val2017',
    )
]
```

### 自定义评测指标

如果算法库中提供的常用评测指标无法满足需求，用户也可以增加自定义的评测指标。我们以简化的分类正确率为例，介绍实现自定义评测指标的方法：

1. 在定义新的评测指标类时，需要继承基类 [BaseMetric](mmengine.evaluator.BaseMetric)（关于该基类的介绍，可以参考[设计文档](../design/evaluation.md)）。此外，评测指标类需要用注册器 `METRICS` 进行注册（关于注册器的说明请参考 [Registry 文档](../advanced_tutorials/registry.md)）。

2. 实现 `process()` 方法。该方法有 2 个输入参数，分别是一个批次的测试数据样本 `data_batch` 和模型预测结果 `data_samples`。我们从中分别取出样本类别标签和分类预测结果，并存放在 `self.results` 中。

3. 实现 `compute_metrics()` 方法。该方法有 1 个输入参数 `results`，里面存放了所有批次测试数据经过 `process()` 方法处理后得到的结果。从中取出样本类别标签和分类预测结果，即可计算得到分类正确率 `acc`。最终，将计算得到的评测指标以字典的形式返回。

4. （可选）可以为类属性 `default_prefix` 赋值。该属性会自动作为输出的评测指标名前缀（如 `defaut_prefix='my_metric'`,则实际输出的评测指标名为 `'my_metric/acc'`），用以进一步区分不同的评测指标。该前缀也可以在配置文件中通过 `prefix` 参数改写。我们建议在 docstring 中说明该评测指标类的 `default_prefix` 值以及所有的返回指标名称。

具体实现如下：

```python
from typing import Sequence, List

from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

import numpy as np


@METRICS.register_module()  # 将 Accuracy 类注册到 METRICS 注册器
class SimpleAccuracy(BaseMetric):
    """ Accuracy Evaluator

    Default prefix: ACC

    Metrics:
        - accuracy (float): classification accuracy
    """

    default_prefix = 'ACC'  # 设置 default_prefix

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]):
        """Process one batch of data and predictions. The processed
        Results should be stored in `self.results`, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """

        # 取出分类预测结果和类别标签
        result = {
            'pred': data_samples['pred_label'],
            'gt': data_samples['data_sample']['gt_label']
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

## 使用离线结果进行评测

另一种常见的模型评测方式，是利用提前保存在文件中的模型预测结果进行离线评测。此时，用户需要手动构建**评测器**，并调用评测器的相应接口完成评测。关于离线评测的详细说明，以及评测器和评测指标的关系，可以参考[设计文档](../design/evaluation.md)。我们仅在此给出一个离线评测示例：

```python
from mmengine.evaluator import Evaluator
from mmengine.fileio import load

# 构建评测器。参数 `metrics` 为评测指标配置
evaluator = Evaluator(metrics=dict(type='Accuracy', top_k=(1, 5)))

# 从文件中读取测试数据。数据格式需要参考具使用的 metric。
data = load('test_data.pkl')

# 从文件中读取模型预测结果。该结果由待评测算法在测试数据集上推理得到。
# 数据格式需要参考具使用的 metric。
data_samples = load('prediction.pkl')

# 调用评测器离线评测接口，得到评测结果
# chunk_size 表示每次处理的样本数量，可根据内存大小调整
results = evaluator.offline_evaluate(data, data_samples, chunk_size=128)

```
