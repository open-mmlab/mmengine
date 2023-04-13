# Evaluation

## Evaluation metrics and evaluators

In model validation and model testing, quantitative evaluation of model accuracy is usually required. `Metric` and `Evaluator` are implemented in MMEngine to perform this function.

- **Metric** is used to complete the calculation of specific model accuracy metrics based on test data and model prediction results. Common metrics for the corresponding tasks are provided in each of the OpenMMLab algorithm libraries, e.g. [Accuracy](https://mmclassification.readthedocs.io/en/1.x/api/generated/mmcls.evaluation.Accuracy.html#mmcls.evaluation.Accuracy) is provided in [MMClassification](https://github.com/open-mmlab/mmclassification) for calculating the top-k rate of correct classification of classification models; [COCOMetric](https://github.com/open-mmlab/mmdetection/blob/3.x/mmdet/evaluation/metrics/coco_metric.py) is provided in [MMDetection](https://github.com/open-mmlab/mmdetection) to calculate AP, AR, and other metrics for object detection models. The evaluation metrics is decoupled from the dataset, as COCOMetric can also be used on object detection datasets other than COCO.

- **Evaluator** is an upper-level module for Metric, usually containing one or more Metric. The role of the Evaluator is to complete the necessary data format conversions during model evaluation and to call the Metric to calculate the model accuracy. Evaluator is usually built from [Runner](../tutorials/runner.md) or test scripts for online and offline evaluations, respectively.

### The base class of Metric `BaseMetric`

`BaseMetric` is an abstract class with the following initialization parameters:

- `collect_device`: is used to synchronize the name of the device of results in distributed reviews, such as `'cpu'` or `'gpu'`.
- `prefix`: the prefix of the metric name which is used to distinguish multiple metrics with the same name. If this parameter is not given, then an attempt is made to use the class attribute `default_prefix` as the prefix.

```python
class BaseMetric(metaclass=ABCMeta):

    default_prefix: Optional[str] = None

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        ...
```

`BaseMetric` has the following two important methods that need to be overridden in the subclass:

- **`process()`** is used to process the test data and model prediction results for each batch. The processing results should be stored in the `self.results` list, which is used to calculate the metrics after all the test data has been processed. This method has the following two parameters:

  - `data_batch`: A sample of test data from a batch, usually directly from the dataloader
  - `data_samples`: Corresponding model prediction results
    This method has no return value. The function interface is defined as follows:

  ```python
  @abstractmethod
  def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
      """Process one batch of data samples and predictions. The processed
      results should be stored in ``self.results``, which will be used to
      compute the metrics when all batches have been processed.
      Args:
          data_batch (Any): A batch of data from the dataloader.
          data_samples (Sequence[dict]): A batch of outputs from the model.
      """
  ```

- **`compute_metrics()`** is used to calculate the metrics and return the metrics in a dictionary. This method has the following one parameter:

  - `results`: list type, which holds the results of all batches of test data processed by the `process()` method
    This method returns a dictionary that holds the names of the metrics and the corresponding values of the metrics. The function interface is defined as follows:

  ```python
  @abstractmethod
  def compute_metrics(self, results: list) -> dict:
      """Compute the metrics from processed results.

      Args:
          results (list): The processed results of each batch.

      Returns:
          dict: The computed metrics. The keys are the names of the metrics,
          and the values are corresponding results.
      """
  ```

Among them, `compute_metrics()` is called in the `evaluate()` method; the latter collects and aggregates intermediate processing results of different ranks during distributed test before calculating the metrics.

Note that the content of `self.results` depends on the implementation of the subclasses. For example, when the amount of test samples or model output data is large (such as semantic segmentation, image generation, and other tasks) and it is not appropriate to store them all in memory, you can store the metrics computed by each batch in `self.results` and collect them in `compute_metrics()`; or store the intermediate results of each batch in a temporary file, and store the temporary file path in `self .results`, and then collect them in `compute_metrics()` by reading the data from the file and calculates the metrics.

## Model accuracy evaluation process

Usually, the process of model accuracy evaluation is shown in the figure below.

**Online evaluation**: The test data is usually divided into batches. Through a loop, each batch is fed into the model in turn, yielding corresponding predictions, and the test data and model predictions are passed to the evaluator. The evaluator calls the `process()` method of the `Metric` to process the data and prediction results. When the loop ends, the evaluator calls the `evaluate()` method of the metrics to calculate the model accuracy of the corresponding metrics.

**Offline  evaluation**: Similar to the online evaluation process, the difference is that the pre-saved model predictions are read directly to perform the evaluation. The evaluator provides the `offline_evaluate` interface for calling the `Metric`s to calculate the model accuracy in an offline way. In order to avoid memory overflow caused by processing a large amount of data at the same time, the offline evaluation divides the test data and prediction results into chunks for processing, similar to the batches in online evaluation.

<div align="center">
    <img src="https://user-images.githubusercontent.com/15977946/187579113-279f097c-3530-40c4-9cd3-1bb0ce2fa452.png" width="500"/>
</div>

## Add custom evaluation metrics

In each algorithm library of OpenMMLab, common evaluation metrics have been implemented in the corresponding direction. For example, COCO metrics is provided in MMDetection and Accuracy, F1Score, etc. are provided in MMClassification.

Users can also add custom metrics. For details, please refer to the examples given in the [tutorial documentation](../tutorials/evaluation.md#Customizing evaluation metrics).
