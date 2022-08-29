# 模型精度评测

在模型验证和模型测试中，通常需要对模型精度做定量评测。在 MMEngine 中实现了评测指标（Metric）和评测器（Evaluator）模块来完成这一功能：

- 评测指标： 用于根据测试数据和模型预测结果，完成模型特定精度指标的计算。在 OpenMMLab 各算法库中提供了对应任务的常用评测指标，如 [MMClassification](https://github.com/open-mmlab/mmclassification) 中提供了[分类正确率指标（Accuracy）](https://mmclassification.readthedocs.io/zh_CN/dev-1.x/generated/mmcls.evaluation.Accuracy.html) 用于计算分类模型的 Top-k 分类正确率。

- 评测器： 是评测指标的上层模块，用于在数据输入评测指标前完成必要的格式转换，并提供分布式支持。在模型训练和测试中，评测器由[执行器（Runner）](https://mmengine.readthedocs.io/zh_CN/latest/api/runner.html)自动构建。用户亦可根据需求手动创建评测器，进行离线评测。

## 在模型训练或测试中进行评测

### 评测指标配置

在基于 MMEngine 进行模型训练或测试时，执行器会自动构建评测器进行评测，用户只需要在配置文件中通过 `val_evaluator` 和 `test_evaluator` 2 个字段分别指定模型验证和测试阶段的评测指标即可。例如，用户在使用 [MMClassification](https://github.com/open-mmlab/mmclassification) 训练分类模型时，希望在模型验证阶段评测 top-1 和 top-5 分类正确率，可以按以下方式配置：

```python
val_evaluator = dict(type='Accuracy', top_k=(1, 5))  # 使用分类正确率评测指标
```

如果需要同时评测多个指标，也可以将 `val_evaluator` 或 `test_evaluator` 设置为一个列表，其中每一项为一个评测指标的配置信息。例如，在使用 [MMDetection](https://github.com/open-mmlab/mmdetection) 训练全景分割模型时，希望在模型测试阶段同时评测模型的目标检测（COCO AP/AR）和全景分割精度，可以按以下方式配置：

```python
test_evaluator = [
    # 目标检测指标
    dict(
        type='COCOMetric',
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

如果算法库中提供的常用评测指标无法满足需求，用户也可以增加自定义的评测指标。具体的方法可以参考[评测指标和评测器设计](/docs/zh_cn/design/metric_and_evaluator.md)。

## 使用离线结果进行评测

另一种常见的模型评测方式，是利用提前保存在文件中的模型预测结果进行离线评测。此时，由于不存在执行器，用户需要手动构建评测器，并调用评测器的相应接口完成评测。以下是一个离线评测示例：

```python
from mmengine.evaluator import Evaluator
from mmengine.fileio import load

# 构建评测器。参数 `metrics` 为评测指标配置
evaluator = Evaluator(metrics=dict(type='Accuracy', top_k=(1, 5)))

# 从文件中读取测试数据。数据格式需要参考具使用的 metric。
data = load('test_data.pkl')

# 从文件中读取模型预测结果。该结果由待评测算法在测试数据集上推理得到。
# 数据格式需要参考具使用的 metric。
predictions = load('prediction.pkl')

# 调用评测器离线评测接口，得到评测结果
# chunk_size 表示每次处理的样本数量，可根据内存大小调整
results = evaluator.offline_evaluate(data, predictions, chunk_size=128)

```
