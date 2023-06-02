# Evaluation

In model validation and testing, it is often necessary to make a quantitative evaluation of model accuracy. We can achieve this by specifying the metrics in the configuration file.

## Evaluation in model training or testing

### Using a single evaluation metric

When training or testing a model based on MMEngine, users only need to specify the evaluation metrics for the validation and testing stages through the `val_evaluator` and `test_evaluator` fields in the configuration file. For example, when using [MMPretrain](https://github.com/open-mmlab/mmpretrain) to train a classification model, if the user wants to evaluate the top-1 and top-5 classification accuracy during the model validation stage, they can configure it as follows:

```python
# using classification accuracy evaluation metric
val_evaluator = dict(type='Accuracy', top_k=(1, 5))
```

For specific parameter settings of evaluation metrics, users can refer to the documentation of the relevant algorithm libraries, such as the [Accuracy](https://mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.evaluation.Accuracy.html#mmpretrain.evaluation.Accuracy) documentation in the above example.

### Using multiple evaluation metrics

If multiple evaluation metrics need to be evaluated simultaneously, `val_evaluator` or `test_evaluator` can be set as a list, with each item being the configuration information for an evaluation metric. For example, when using [MMDetection](https://github.com/open-mmlab/mmdetection) to train a panoptic segmentation model, if the user wants to evaluate both the object detection (COCO AP/AR) and panoptic segmentation accuracy during the model testing stage, they can configure it as follows:

```python
test_evaluator = [
    # object detection metric
    dict(
        type='CocoMetric',
        metric=['bbox', 'segm'],
        ann_file='annotations/instances_val2017.json',
    ),
    # panoramic segmentation metric
    dict(
        type='CocoPanopticMetric',
        ann_file='annotations/panoptic_val2017.json',
        seg_prefix='annotations/panoptic_val2017',
    )
]
```

### Customizing evaluation metrics

If the common evaluation metrics provided in the algorithm library cannot meet the needs, users can also add custom evaluation metrics. As an example, we present the implementation of custom metrics with the simplified classification accuracy:

1. When defining a new evaluation metric class, you need to inherit the base class [BaseMetric](mmengine.evaluator.BaseMetric) (for an introduction to this base class, you can refer to the [design document](../design/evaluation.md)). In addition, the evaluation metric class needs to be registered with the registrar `METRICS` (for a description of the registrar, please refer to the [Registry documentation](../advanced_tutorials/registry.md)).

2. Implement the `process()` method. This method has two input parameters, which are a batch of test data samples, `data_batch`, and model prediction results, `data_samples`. We extract the sample category labels and the classification prediction results from them and store them in `self.results` respectively.

3. Implement the `compute_metrics()` method. This method has one input parameter `results`, which holds the results of all batches of test data processed by the `process()` method. The sample category labels and classification predictions are extracted from the results to calculate the classification accuracy (`acc`). Finally, the calculated evaluation metrics are returned in the form of a dictionary.

4. (Optional) You can assign a value to the class attribute `default_prefix`. This attribute is automatically prefixed to the output metric name (e.g. `defaut_prefix='my_metric'`, then the actual output metric name is `'my_metric/acc'`) to further distinguish the different metrics. This prefix can also be rewritten in the configuration file via the `prefix` parameter. We recommend describing the `default_prefix` value for the metric class and the names of all returned metrics in the docstring.

The specific implementation is as follows:

```python
from typing import Sequence, List

from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

import numpy as np


@METRICS.register_module()  # register the Accuracy class to the METRICS registry
class SimpleAccuracy(BaseMetric):
    """ Accuracy Evaluator

    Default prefix: ACC

    Metrics:
        - accuracy (float): classification accuracy
    """

    default_prefix = 'ACC'  # set default_prefix

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

        # fetch classification prediction results and category labels
        result = {
            'pred': data_samples['pred_label'],
            'gt': data_samples['data_sample']['gt_label']
        }

        # store the results of the current batch into self.results
        self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        # aggregate the classification prediction results and category labels for all samples
        preds = np.concatenate([res['pred'] for res in results])
        gts = np.concatenate([res['gt'] for res in results])

        # calculate the classification accuracy
        acc = (preds == gts).sum() / preds.size

        # return evaluation metric results
        return {'accuracy': acc}
```

## Using offline results for evaluation

Another common way of model evaluation is to perform offline evaluation using model prediction results saved in files in advance. In this case, the user needs to manually build **Evaluator** and call the corresponding interface of the evaluator to complete the evaluation. For more details about offline evaluation and the relationship between the evaluator and the metric, please refer to the [design document](../design/evaluation.md). We only give an example of offline evaluation here:

```python
from mmengine.evaluator import Evaluator
from mmengine.fileio import load

# Build the evaluator. The parameter `metrics` is the configuration of the evaluation metric
evaluator = Evaluator(metrics=dict(type='Accuracy', top_k=(1, 5)))

# Reads the test data from a file. The data format needs to refer to the metric used.
data = load('test_data.pkl')

# The model prediction result is read from the file. The result is inferred by the algorithm to be evaluated on the test dataset.
# The data format needs to refer to the metric used.
data_samples = load('prediction.pkl')

# Call the evaluator offline evaluation interface and get the evaluation results
# chunk_size indicates the number of samples processed at a time, which can be adjusted according to the memory size
results = evaluator.offline_evaluate(data, data_samples, chunk_size=128)
```
