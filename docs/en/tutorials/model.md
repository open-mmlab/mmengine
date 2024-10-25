# Model

## Runner and model

As mentioned in [basic dataflow](./runner.md#basic-dataflow), the dataflow between DataLoader, model and evaluator follows some rules. Don't remember clearly? Let's review it:

```python
# Training process
for data_batch in train_dataloader:
    data_batch = model.data_preprocessor(data_batch, training=True)
    if isinstance(data_batch, dict):
        losses = model(**data_batch, mode='loss')
    elif isinstance(data_batch, (list, tuple)):
        losses = model(*data_batch, mode='loss')
    else:
        raise TypeError()
# Validation process
for data_batch in val_dataloader:
    data_batch = model.data_preprocessor(data_batch, training=False)
    if isinstance(data_batch, dict):
        outputs = model(**data_batch, mode='predict')
    elif isinstance(data_batch, (list, tuple)):
        outputs = model(**data_batch, mode='predict')
    else:
        raise TypeError()
    evaluator.process(data_samples=outputs, data_batch=data_batch)
metrics = evaluator.evaluate(len(val_dataloader.dataset))
```

In [runner tutorial](../tutorials/runner.md), we simply mentioned the relationship between DataLoader, model and evaluator, and introduced the concept of `data_preprocessor`. You may have a certain understanding of the model. However, during the running of Runner, the situation is far more complex than the above pseudo-code.

In order to focus your attention on the algorithm itself, and ignore the complex relationship between the model, DataLoader and evaluator, we designed [BaseModel](mmengine.model.BaseModel). In most cases, the only thing you need to do is to make your model inherit from `BaseModel`, and implement the `forward` as required to perform the training, testing, and validation process.

Before continuing reading the model tutorial, let's throw out two questions that we hope you will find the answers after reading the model tutorial:

1. When do we update the parameters of model? and how to update the parameters by a custom optimization process?
2. Why is the concept of data_preprocessor necessary? What functions can it perform?

## Interface introduction

Usually, we should define a model to implement the body of the algorithm. In MMEngine, model will be managed by Runner, and need to implement some interfaces, such as `train_step`, `val_step`, and `test_step`. For high-level tasks like detection, classification, and segmentation, the interfaces mentioned above commonly implement a standard workflow. For example, `train_step` will calculate the loss and update the parameters of the model, and `val_step`/`test_step` will calculate the metrics and return the predictions. Therefore, MMEnine abstracts the [BaseModel](mmengine.model.BaseModel) to implement the common workflow.

Benefits from the `BaseModel`, we only need to make the model inherit from `BaseModel`, and implement the `forward` function to perform the training, testing, and validation process.

```{note}
BaseModel inherits from [BaseModule](../advanced_tutorials/initialize.md), which can be used to initialize the model parameters dynamically.
```

[**forward**](mmengine.model.BaseModel.forward): The arguments of `forward` need to match with the data given by [DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). If the DataLoader samples a tuple `data`, `forward` needs to accept the value of unpacked `*data`. If DataLoader returns a dict `data`, `forward` needs to accept the key-value of unpacked `**data`. `forward` also accepts `mode` parameter, which is used to control the running branch:

- `mode='loss'`: `loss` mode is enabled in training process, and `forward` returns a differentiable loss `dict`. Each key-value pair in loss `dict` will be used to log the training status and optimize the parameters of model. This branch will be called by `train_step`

- `mode='predict'`: `predict` mode is enabled in validation/testing process, and `forward` will return predictions, which matches with arguments of [process](mmengine.evaluator.Evaluator.process). Repositories of OpenMMLab have a more strict rules. The predictions must be a list and each element of it must be a [BaseDataElement](../advanced_tutorials/data_element.md). This branch will be called by `val_step`

- `mode='tensor'`: In `tensor` and `predict` modes, `forward` will return the predictions. The difference is that `forward` will return a `tensor` or a container or `tensor` which has not been processed by a series of post-process methods, such as non-maximum suppression (NMS). You can customize your post-process method after getting the result of `tensor` mode.

[**train_step**](mmengine.model.BaseModel.train_step): Get the loss `dict` by calling `forward` with `loss` mode. `BaseModel` implements a standard optimization process as follows:

```python
def train_step(self, data, optim_wrapper):
    # See details in the next section
    data = self.data_preprocessor(data, training=True)
    # `loss` mode, return a loss dict. Actually train_step accepts
    #  both tuple  dict input, and unpack it with ** or *
    loss = self(**data, mode='loss')
    # Parse the loss dict and return the parsed losses for optimization
    # and log_vars for logging
    parsed_losses, log_vars = self.parse_losses()
    optim_wrapper.update_params(parsed_losses)
    return log_vars
```

[**val_step**](mmengine.model.BaseModel.val_step): Get the predictions by calling `forward` with `predict` mode.

```python
def val_step(self, data, optim_wrapper):
    data = self.data_preprocessor(data, training=False)
    outputs = self(**data, mode='predict')
    return outputs
```

[**test_step**](mmengine.model.BaseModel.test_step): There is no difference between `val_step` and `test_step` in `BaseModel`. But we can customize it in the subclasses, for example, you can get validation loss in `val_step`.

Understand the interfaces of `BaseModel`, now we are able to come up with a more complete pseudo-code:

```python
# training
for data_batch in train_dataloader:
    loss_dict = model.train_step(data_batch)
# validation
for data_batch in val_dataloader:
    preds = model.test_step(data_batch)
    evaluator.process(data_samples=outputs, data_batch=data_batch)
metrics = evaluator.evaluate(len(val_dataloader.dataset))
```

Great!, ignoring `Hook`, the pseudo-code above almost implements the main logic in [loop](mmengine.runner.EpochBasedTrainLoop)! Let's go back to [15 minutes to get started with MMEngine](../get_started/15_minutes.md), we may truly understand what `MMResNet` has done:

```python
import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel

class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels

    # train_step, val_step and test_step have been implemented in BaseModel.
    # We list the equivalent code here for better understanding
    def train_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        loss = self(*data, mode='loss')
        parsed_losses, log_vars = self.parse_losses()
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        outputs = self(*data, mode='predict')
        return outputs

    def test_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        outputs = self(*data, mode='predict')
        return outputs
```

Now, you may have a deeper understanding of dataflow, and can answer the first question in [Runner and model](#runner-and-model).

`BaseModel.train_step` implements the standard optimization, and if we want to customize a new optimization process, we can override it in the subclass. However, it is important to note that we need to make sure that `train_step` returns a loss dict.

## DataPreprocessor

If your computer is equipped with a GPU (or other hardware that can accelerate training, such as MPS, IPU, etc.), when you run the [15 minutes tutorial](../get_started/15_minutes.md), you will see that the program is running on the GPU, but, when does `MMEngine` move the data and model from the CPU to the GPU?

In fact, the Runner will move the model to the specified device during the construction, while the data will be moved to the specified device at the `self.data_preprocessor(data)` mentioned in the code snippet of the previous section. The moved data will be further passed to the model.

Makes sense but it's weird, isn't it? At this point you may be wondering:

1. `MMResNet50` does not define `data_preprocessor`, but why it can still access `data_preprocessor` and move data to GPU?

2. Why `BaseModel` does not move data by `data = data.to(device)`, but needs the `DataPreprocessor` to move data?

The answer to the first question is that: `MMResNet50` inherit from `BaseModel`, and `super().__init__` will build a default `data_preprocessor` for it. The equivalent implementation of the default one is like this:

```python
class BaseDataPreprocessor(nn.Module):
    def forward(self, data, training=True):  # ignore the training parameter here
        # suppose data given by CIFAR10 is a tuple. Actually
        # BaseDataPreprocessor could move various type of data
        # to target device.
        return tuple(_data.cuda() for _data in data)
```

`BaseDataPreprocessor` will move the data to the specified device.

Before answering the second question, let's think about a few more questions

1. Where should we perform normalization? [transform](../advanced_tutorials/data_transform.md) or `Model`?

   It sounds reasonable to put it in transform to take advantage of Dataloader's multi-process acceleration, and in the model to move it to GPU to use GPU resources to accelerate normalization. However, while we are debating whether CPU normalization is faster than GPU normalization, the time of data moving from CPU to GPU is much longer than the former.

   In fact, for less computationally intensive operations like normalization, it takes much less time than data transferring, which has a higher priority for being optimized. If I could move the data to the specified device while it is still in `uint8` and before it is normalized (the size of normalized `float` data is 4 times larger than that of unit8), it would reduce the bandwidth and greatly improve the efficiency of data transferring. This "lagged" normalization behavior is one of the main reasons why we designed the `DataPreprocessor`. The data preprocessor moves the data first and then normalizes it.

2. How we implement the data augmentation like MixUp and Mosaic?

   Although it seems that MixUp and Mosaic are just special data transformations that should be implemented in transform. However, considering that these two transformations involve **fusing multiple images into one**, it would be very difficult to implement them in transform since the current paradigm of transform is to do various enhancements on **one** image. It would be hard to read additional images from dataset because the dataset is not accessible in the transform. However, if we implement Mosaic or Mixup based on the `batch_data` sampled from Dataloader, everything becomes easy. We can access multiple images at the same time, and we can easily perform the image fusion operation.

   ```python
   class MixUpDataPreprocessor(nn.Module):
       def __init__(self, num_class, alpha):
           self.alpha = alpha

       def forward(self, data, training=True):
           data = tuple(_data.cuda() for _data in data)
           # Only perform MixUp in training mode
           if not training:
               return data

           label = F.one_hot(label)  # label to OneHot
           batch_size = len(label)
           index = torch.randperm(batch_size)  # Get the index of fused image
           img, label = data
           lam = np.random.beta(self.alpha, self.alpha)  # Fusion factor

           # MixUp
           img = lam * img + (1 - lam) * img[index, :]
           label = lam * batch_scores + (1 - lam) * batch_scores[index, :]
           # Since the returned label is onehot encoded, the `forward` of the
           # model should also be adjusted.
           return tuple(img, label)
   ```

   Therefore, besides data transferring and normalization, another major function of `data_preprocessor` is BatchAugmentation. The modularity of the data preprocessor also helps us to achieve a free combination between algorithms and data augmentation.

3. What should we do if the data sampled from the DataLoader does not match the model input, should I modify the DataLoader or the model interface?

   The answer is: neither is appropriate. The ideal solution is to do the adaptation without breaking the existing interface between the model and the DataLoader. `DataPreprocessor` could also handle this, you can customize your `DataPreprocessor` to convert the incoming to the target type.

By now, You must understand the rationale of the data preprocessor and can confidently answer the two questions posed at the beginning of the tutorial! But you may still wonder what is the `optim_wrapper` passed to `train_step`, and how do the predictions returned by `test_step` and `val_step` relate to the evaluator. You will find more introduction in the [evaluation tutorial](./evaluation.md) and the [optimizer wrapper tutorial](./optim_wrapper.md).
