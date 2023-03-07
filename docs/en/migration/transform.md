# Migrate Data Transform to OpenMMLab 2.0

## Introduction

According to the data transform interface convention of TorchVision, all data transform classes need to
implement the `__call__` method. And in the convention of OpenMMLab 1.0, we require the input and output of
the `__call__` method should be a dictionary.

In OpenMMLab 2.0, to make the data transform classes more extensible, we use `transform` method instead of
`__call__` method to implement data transformation, and all data transform classes should inherit the
[mmcv.transforms.BaseTransform](mmcv.transforms.BaseTransform) class. And you can still use these data
transform classes by calling.

A tutorial to implement a data transform class can be found in the [Data Transform](../advanced_tutorials/data_element.md).

In addition, we move some common data transform classes from every repositories to MMCV, and in this document,
we will compare the functionalities, usages and implementations between the original data transform classes (in [MMClassification v0.23.2](https://github.com/open-mmlab/mmclassification/tree/v0.23.2), [MMDetection v2.25.1](https://github.com/open-mmlab/mmdetection/tree/v2.25.1)) and the new data transform classes (in [MMCV v2.0.0rc1](https://github.com/open-mmlab/mmcv/tree/2.x))

## Functionality Differences

<table class="colwidths-auto docutils align-default">
<thead>
  <tr>
    <th></th>
    <th>MMClassification (original)</th>
    <th>MMDetection (original)</th>
    <th>MMCV (new)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><code>LoadImageFromFile</code></td>
    <td>Join the 'img_prefix' and 'img_info.filename' field to find the path of images and loading.</td>
    <td>Join the 'img_prefix' and 'img_info.filename' field to find the path of images and loading. Support
    specifying the order of channels.</td>
    <td>Load images from 'img_path'. Support ignoring failed loading and specifying decode backend.</td>
  </tr>
  <tr>
    <td><code>LoadAnnotations</code></td>
    <td>Not available.</td>
    <td>Load bbox, label, mask (include polygon masks), semantic segmentation. Support converting bbox coordinate system.</td>
    <td>Load bbox, label, mask (not include polygon masks), semantic segmentation.</td>
  </tr>
  <tr>
    <td><code>Pad</code></td>
    <td>Pad all images in the "img_fields" field.</td>
    <td>Pad all images in the "img_fields" field. Support padding to integer multiple size.</td>
    <td>Pad the image in the "img" field. Support padding to integer multiple size.</td>
  </tr>
  <tr>
    <td><code>CenterCrop</code></td>
    <td>Crop all images in the "img_fields" field. Support cropping as EfficientNet style.</td>
    <td>Not available.</td>
    <td>Crop the image in the "img" field, the bbox in the "gt_bboxes" field, the semantic segmentation in the "gt_seg_map" field, the keypoints in the "gt_keypoints" field. Support padding the margin of the cropped image.</td>
  </tr>
  <tr>
    <td><code>Normalize</code></td>
    <td>Normalize the image.</td>
    <td>No differences.</td>
    <td>No differences, but we recommend to use <a href="../tutorials/model.html#datapreprocessor">data preprocessor</a> to normalize the image.</td>
  </tr>
  <tr>
    <td><code>Resize</code></td>
    <td>Resize all images in the "img_fields" field. Support resizing proportionally according to the specified edge.</td>
    <td>Use <code>Resize</code> with <code>ratio_range=None</code>, the <code>img_scale</code> have a single scale, and <code>multiscale_mode="value"</code>.</td>
    <td>Resize the image in the "img" field, the bbox in the "gt_bboxes" field, the semantic segmentation in the "gt_seg_map" field, the keypoints in the "gt_keypoints" field. Support specifying the ratio of new scale to original scale and support resizing proportionally.</td>
  </tr>
  <tr>
    <td><code>RandomResize</code></td>
    <td>Not available</td>
    <td>Use <code>Resize</code> with <code>ratio_range=None</code>, <code>img_scale</code> have two scales and <code>multiscale_mode="range"</code>, or <code>ratio_range</code> is not None.
    <pre>Resize(
    img_sacle=[(640, 480), (960, 720)],
    mode="range",
)</pre>
    </td>
    <td>Have the same resize function as <code>Resize</code>. Support sampling the scale from a scale range or scale ratio range.
    <pre>RandomResize(scale=[(640, 480), (960, 720)])</pre>
    </td>
  </tr>
  <tr>
    <td><code>RandomChoiceResize</code></td>
    <td>Not available</td>
    <td>Use <code>Resize</code> with <code>ratio_range=None</code>, <code>img_scale</code> have multiple scales, and <code>multiscale_mode="value"</code>.
    <pre>Resize(
    img_sacle=[(640, 480), (960, 720)],
    mode="value",
)</pre>
    </td>
    <td>Have the same resize function as <code>Resize</code>. Support randomly choosing the scale from multiple scales or multiple scale ratios.
    <pre>RandomChoiceResize(scales=[(640, 480), (960, 720)])</pre>
    </td>
  </tr>
  <tr>
    <td><code>RandomGrayscale</code></td>
    <td>Randomly grayscale all images in the "img_fields" field. Support keeping channels after grayscale.</td>
    <td>Not available</td>
    <td>Randomly grayscale the image in the "img" field. Support specifying the weight of each channel, and support keeping channels after grayscale.</td>
  </tr>
  <tr>
    <td><code>RandomFlip</code></td>
    <td>Randomly flip all images in the "img_fields" field. Support flipping horizontally and vertically.</td>
    <td>Randomly flip all values in the "img_fields", "bbox_fields", "mask_fields" and "seg_fields". Support flipping horizontally, vertically and diagonally, and support specifying the probability of every kind of flipping.</td>
    <td>Randomly flip the values in the "img", "gt_bboxes", "gt_seg_map", "gt_keypoints" field. Support flipping horizontally, vertically and diagonally, and support specifying the probability of every kind of flipping.</td>
  </tr>
  <tr>
    <td><code>MultiScaleFlipAug</code></td>
    <td>Not available</td>
    <td>Used for test-time-augmentation.</td>
    <td>Use <code><a href="https://mmcv.readthedocs.io/en/2.x/api/generated/mmcv.transforms.TestTimeAug.html">TestTimeAug</a></code></td>
  </tr>
  <tr>
    <td><code>ToTensor</code></td>
    <td>Convert the values in the specified fields to <code>torch.Tensor</code>.</td>
    <td>No differences</td>
    <td>No differences</td>
  </tr>
  <tr>
    <td><code>ImageToTensor</code></td>
    <td>Convert the values in the specified fields to <code>torch.Tensor</code> and transpose the channels to CHW.</td>
    <td>No differences.</td>
    <td>No differences.</td>
  </tr>
</tbody>
</table>

## Implementation Differences

Take `RandomFlip` as example, the new version [RandomFlip](mmcv.transforms.RandomFlip) in MMCV inherits `BaseTransfrom`, and move the
functionality implementation from `__call__` to `transform` method. In addition, the randomness related code
is placed in some extra methods and these methods need to be wrapped by `cache_randomness` decorator.

- MMDetection (original version)

```python
class RandomFlip:
    def __call__(self, results):
        """Randomly flip images."""
        ...
        # Randomly choose the flip direction
        cur_dir = np.random.choice(direction_list, p=flip_ratio_list)
        ...
        return results
```

- MMCV (new version)

```python
class RandomFlip(BaseTransfrom):
    def transform(self, results):
        """Randomly flip images"""
        ...
        cur_dir = self._random_direction()
        ...
        return results

    @cache_randomness
    def _random_direction(self):
        """Randomly choose the flip direction"""
        ...
        return np.random.choice(direction_list, p=flip_ratio_list)
```
