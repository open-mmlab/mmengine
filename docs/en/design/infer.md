# Inference

When developing with MMEngine, we usually define a configuration file for a specific algorithm, use the file to build a [runner](./runner.md), execute the training and testing processes, and save the trained weights. When performing inference based on the trained model, the following steps are usually required:

1. Build the model based on the configuration file
2. Load the model weights
3. Set up the data preprocessing pipeline
4. Perform the forward inference of the model
5. Visualize the inference results
6. Return the inference results

For such standard inference workflow, MMEngine provides a unified inference interface and recommends that users develop inference applications based on this interface specification.

## Usage

### Defining an Inferencer

Implement a custom `inferencer` based on `BaseInferencer`

```python
from mmengine.infer import BaseInferencer

class CustomInferencer(BaseInferencer)
    ...
```

For specific details, please refer to the [Development Specification](#Development-Specification-of-Inference-Interface).

### Building an Inferencer

**Building an Inferencer Based on Configuration File**

```python
cfg = 'path/to/config.py'
weight = 'path/to/weight.pth'

inferencer = CustomInferencer(model=cfg, weight=weight)
```

**Building an Inferencer Based on Config object**

```python
from mmengine import Config

cfg = Config.fromfile('path/to/config.py')
weight = 'path/to/weight.pth'

inferencer = CustomInferencer(model=cfg, weight=weight)
```

**Building an Inferencer based on model name defined in model-index**. Take the [ATSS detector in MMDetection](https://github.com/open-mmlab/mmdetection/blob/31c84958f54287a8be2b99cbf87a6dcf12e57753/configs/atss/metafile.yml#L22) as an example, the model name is `atss_r50_fpn_1x_coco`. Since the path of weight has already been defined in the model-index, there is no need to configure the weight argument anymore.

```python
inferencer = CustomInferencer(model='atss_r50_fpn_1x_coco')
```

### Performing Inference

**Inferring on a Single Image**

```python
# Input as Image Path
img = 'path/to/img.jpg'
result = inferencer(img)

# Input as Loaded Image (Type: np.ndarray)
img = cv2.imread('path/to/img.jpg')
result = inferencer(img)

# Input as url
img = 'https://xxx.com/img.jpg'
result = inferencer(img)
```

**Inferring on Multiple Images**

```python
img_dir = 'path/to/directory'
result = inferencer(img_dir)
```

```{note}
OpenMMLab requires the `inferencer(img)` to output a `dict` containing two fields: `visualization: list` and `predictions: list`, representing the visualization results and prediction results, respectively.
```

## Development Specification of Inference Interface

When performing inference, the following steps are typically executed:

1. preprocess: Input data preprocessing, including data reading, data preprocessing, data format conversion, etc.
2. forward: Execute `model.forwward`
3. visualize: Visualization of predicted results.
4. postprocess: Post-processing of predicted results, including result format conversion, exporting predicted results, etc.

To improve the user experience of the inferencer,  we do not want users to have to configure parameters for each step when performing inference. In other words, we hope that users can simply configure parameters for the `__call__` interface without being aware of the above process and complete the inference.

The `__call__` interface will execute the aforementioned steps in order, but it is not aware of which step the parameters provided by the user should be assigned to. Therefore, when developing a `CustomInferencer`, developers need to define four class attributes: `preprocess_kwargs`, `forward_kwargs`, `visualize_kwargs`, and `postprocess_kwargs`. Each attribute is a set of strings that are used to specify which step the parameters in the `__call__` interface correspond to:

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

In the code above, `a`, `b`, `c`, and `d` in the `preprocess`, `forward`, `visualize`, and `postprocess` functions are additional parameters that can be passed in by the user (`inputs`, `preds`, and other parameters are automatically filled in during the execution of `__call__`). Therefore, developers need to specify these parameters in the `preprocess_kwargs`, `forward_kwargs`, `visualize_kwargs`, and `postprocess_kwargs` class attributes, so that the parameters passed in by the user in the `__call__` phase can be correctly assigned to the corresponding steps. The distribution process is implemented by the `BaseInferencer.__call__` function, which developers do not need to be concerned about.

In addition, we need to register the `CustomInferencer` to a custom registry or the MMEngine's registry.

```python
from mmseg.registry import INFERENCERS
# It can also be registered to the registry of MMEngine.
# from mmengine.registry import INFERENCERS

@INFERENCERS.register_module()
class CustomInferencer(BaseInferencer):
    ...
```

```{note}
In OpenMMLab's algorithm repositories, the Inferencer must be registered to the downstream repository's registry instead of the root registry of MMEngine to avoid naming conflicts.
```

## Core Interface Explanation:

### `__init__()`

The `BaseInferencer.__init__` method has already implemented the logic for building an inferencer as shown in the above [section](#Building-an-Inferencer), so in most cases, there is no need to override the `__init__` method. However, if there is a need to implement custom logic for loading configuration files, weight initialization, pipeline initialization, etc., the `__init__` method can be overridden.

### `_init_pipeline()`

```{note}
This is an abstract method that must be implemented by the subclass.
```

Initialize and return the pipeline required by the inferencer. The pipeline is used for a single image, similar to the `train_pipeline` and `test_pipeline` defined in the OpenMMLab series algorithm library. Each `inputs` passed in by the user when calling the `__call__` interface will be processed by the pipeline to form batch data, which will then be passed to the `forward` method. This is an abstract method that must be implemented by the subclass.

### `_init_collate()`

Initialize and return the `collate_fn` required by the inferencer, which is equivalent to the `collate_fn` of the Dataloader in the training process. `BaseInferencer` will obtain the `collate_fn` from the configuration of `test_dataloader` by default, so it is generally not necessary to override the `_init_collate` method.

### `_init_visualizer()`

Initializes and returns the `visualizer` required by the inferencer, which is equivalent to the `visualizer` used in the training process. By default, `BaseInferencer` obtains the `visualizer` from the configuration of the `visualizer`, so there is usually no need to override the `_init_visualizer` function.

### `preprocess()`

Input arguments:

- inputs: Input data, passed into `__call__`, usually a list of image paths or image data.
- batch_size: batch size, passed in by the user when calling `__call__`.
- Other parameters: Passed in by the user and specified in `preprocess_kwargs`.

Return:

- A generator that yields one batch of data at each iteration.

The `preprocess` function is a generator function by default, which applies the `pipeline` and `collate_fn` to the input data, and yields the preprocessed batch data. In general, subclasses do not need to override this function.

### `forward()`

Input arguments:

- inputs: The batch data processed by `preprocess` function.
- Other parameters: Passed in by the user and specified in `forward_kwargs`.

Return:

- Prediction result, default type is `List[BaseDataElement]`.

Calls `model.test_step` to perform forward inference and returns the inference result. Subclasses typically do not need to override this method.

### `visualize()`

```{note}
This is an abstract method that must be implemented by the subclass.
```

Input arguments:

- inputs: The input data, which is the raw data without preprocessing.
- preds: Predicted results of the model.
- show: Whether to visualize.
- Other parameters: Passed in by the user and specified in `visualize_kwargs`.

Return:

- Visualize the results, which are usually of type `List[np.ndarray]`. Taking object detection as an example, each element in the list should be an image with detection boxes drawn, which can be visualized using `cv2.imshow`. The visualization process may vary for different tasks, and `visualize` should return results that are suitable for common visualization processes in that field.

### `postprocess()`

```{note}
This is an abstract method that must be implemented by the subclass.
```

Input arguments:

- preds: The predicted results of the model, which is a `list` type. Each element in the list represents the prediction result for a single data item. In the OpenMMLab series of algorithm libraries, the type of each element in the prediction result is `BaseDataElement`.
- visualization: Visualization results
- return_datasample: Whether to maintain datasample for return. When set to `False`, the returned result is converted to a `dict`.
- Other parameters: Passed in by the user and specified in `postprocess_kwargs`.

Return:

- The type of the returned value is a dictionary containing both the visualization and prediction results. OpenMMLab requires the returned dictionary to have two keys: `predictions` and `visualization`.

### `__call__()`

Input arguments:

- inputs: The input data, usually a list of image paths or image data. Each element in `inputs` can also be other types of data as long as it can be processed by the `pipeline` returned by [init_pipeline](#_init_pipeline). When there is only one inference data in `inputs`, it does not have to be a `list`, `__call__` will internally wrap it into a list for further processing.
- return_datasample: Whether to convert datasample to dict for return.
- batch_size: Batch size for inference, which will be further passed to the `preprocess` function.
- Other parameters: Additional parameters assigned to `preprocess`, `forward`, `visualize`, and `postprocess` methods.

Return:

- The visualized and predicted results returned by `postprocess`, in the form of a dictionary. OpenMMLab requires the returned dictionary to contain two keys: `predictions` and `visualization`.
