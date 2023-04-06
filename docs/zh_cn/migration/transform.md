# 数据变换类的迁移

## 简介

在 TorchVision 的数据变换类接口约定中，数据变换类需要实现 `__call__` 方法，而在 OpenMMLab 1.0 的接口约定中，进一步要求
`__call__` 方法的输出应当是一个字典，在各种数据变换中对这个字典进行增删查改。在 OpenMMLab 2.0 中，为了提升后续的可扩展性，我们将原先的 `__call__` 方法迁移为 `transform` 方法，并要求数据变换类应当继承
[mmcv.transforms.BaseTransform](mmcv.transforms.BaseTransform)。具体如何实现一个数据变换类，可以参见[文档](../advanced_tutorials/data_transform.md)。

由于在此次更新中，我们将部分共用的数据变换类统一迁移至 MMCV 中，因此本文将会对比这些数据变换在旧版本（[MMClassification v0.23.2](https://github.com/open-mmlab/mmclassification/tree/v0.23.2)、[MMDetection v2.25.1](https://github.com/open-mmlab/mmdetection/tree/v2.25.1)）和新版本（[MMCV v2.0.0rc0](https://github.com/open-mmlab/mmcv/tree/2.x)）中的功能、用法和实现上的差异。

## 功能差异

<table class="colwidths-auto docutils align-default">
<thead>
  <tr>
    <th></th>
    <th>MMClassification (旧)</th>
    <th>MMDetection (旧)</th>
    <th>MMCV (新)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><code>LoadImageFromFile</code></td>
    <td>从 'img_prefix' 和 'img_info.filename' 字段组合获得文件路径并读取</td>
    <td>从 'img_prefix' 和 'img_info.filename' 字段组合获得文件路径并读取，支持指定通道顺序</td>
    <td>从 'img_path' 获得文件路径并读取，支持指定加载失败不报错，支持指定解码后端</td>
  </tr>
  <tr>
    <td><code>LoadAnnotations</code></td>
    <td>无</td>
    <td>支持读取 bbox，label，mask（包括多边形样式），seg map，转换 bbox 坐标系</td>
    <td>支持读取 bbox，label，mask（不包括多边形样式），seg map</td>
  </tr>
  <tr>
    <td><code>Pad</code></td>
    <td>填充 "img_fields" 中所有字段，不支持指定填充至整数倍</td>
    <td>填充 "img_fields" 中所有字段，支持指定填充至整数倍</td>
    <td>填充 "img" 字段，支持指定填充至整数倍</td>
  </tr>
  <tr>
    <td><code>CenterCrop</code></td>
    <td>裁切 "img_fields" 中所有字段，支持以 EfficientNet 方式进行裁切</td>
    <td>无</td>
    <td>裁切 "img" 字段的图像，"gt_bboxes" 字段的 bbox，"gt_seg_map" 字段的分割图，"gt_keypoints" 字段的关键点，支持自动填充裁切边缘</td>
  </tr>
  <tr>
    <td><code>Normalize</code></td>
    <td>图像归一化</td>
    <td>无差异</td>
    <td>无差异，但 MMEngine 推荐在<a href="../tutorials/model.html#datapreprocessor">数据预处理器</a>中进行归一化</td>
  </tr>
  <tr>
    <td><code>Resize</code></td>
    <td>缩放 "img_fields" 中所有字段，允许指定根据某边长等比例缩放</td>
    <td>功能由 <code>Resize</code> 实现。需要 <code>ratio_range</code> 为 None，<code>img_scale</code> 仅指定一个尺寸，且 <code>multiscale_mode</code> 为 "value" 。</td>
    <td>缩放 "img" 字段的图像，"gt_bboxes" 字段的 bbox，"gt_seg_map" 字段的分割图，"gt_keypoints" 字段的关键点，支持指定缩放比例，支持等比例缩放图像至指定尺寸内</td>
  </tr>
  <tr>
    <td><code>RandomResize</code></td>
    <td>无</td>
    <td>功能由 <code>Resize</code> 实现。需要 <code>ratio_range</code> 为 None，<code>img_scale</code>指定两个尺寸，且 <code>multiscale_mode</code> 为 "range"，或 <code>ratio_range</code> 不为 None。
    <pre>Resize(
    img_sacle=[(640, 480), (960, 720)],
    mode="range",
)</pre>
    </td>
    <td>缩放功能同 <code>Resize</code>，支持从指定尺寸范围或指定比例范围随机采样缩放尺寸。
    <pre>RandomResize(scale=[(640, 480), (960, 720)])</pre>
    </td>
  </tr>
  <tr>
    <td><code>RandomChoiceResize</code></td>
    <td>无</td>
    <td>功能由 <code>Resize</code> 实现。需要 <code>ratio_range</code> 为 None，<code>img_scale</code> 指定多个尺寸，且 <code>multiscale_mode</code> 为 "value"。
    <pre>Resize(
    img_sacle=[(640, 480), (960, 720)],
    mode="value",
)</pre>
    </td>
    <td>缩放功能同 <code>Resize</code>，支持从若干指定尺寸中随机选择缩放尺寸。
    <pre>RandomChoiceResize(scales=[(640, 480), (960, 720)])</pre>
    </td>
  </tr>
  <tr>
    <td><code>RandomGrayscale</code></td>
    <td>灰度化 "img_fields" 中所有字段，灰度化后保持通道数。</td>
    <td>无</td>
    <td>灰度化 "img" 字段，支持指定灰度化权重，支持指定是否在灰度化后保持通道数（默认不保持）。</td>
  </tr>
  <tr>
    <td><code>RandomFlip</code></td>
    <td>翻转 "img_fields" 中所有字段，支持指定水平或垂直翻转。</td>
    <td>翻转 "img_fields", "bbox_fields", "mask_fields", "seg_fields" 中所有字段，支持指定水平、垂直或对角翻转，支持指定各类翻转概率。</td>
    <td>翻转 "img", "gt_bboxes", "gt_seg_map", "gt_keypoints" 字段，支持指定水平、垂直或对角翻转，支持指定各类翻转概率。</td>
  </tr>
  <tr>
    <td><code>MultiScaleFlipAug</code></td>
    <td>无</td>
    <td>用于测试时增强</td>
    <td>使用 <code><a href="https://mmcv.readthedocs.io/en/2.x/api/generated/mmcv.transforms.TestTimeAug.html">TestTimeAug</a></code></td>
  </tr>
  <tr>
    <td><code>ToTensor</code></td>
    <td>将指定字段转换为 <code>torch.Tensor</code></td>
    <td>无差异</td>
    <td>无差异</td>
  </tr>
  <tr>
    <td><code>ImageToTensor</code></td>
    <td>将指定字段转换为 <code>torch.Tensor</code>，并调整通道顺序至 CHW。</td>
    <td>无差异</td>
    <td>无差异</td>
  </tr>
</tbody>
</table>

## 实现差异

以 `RandomFlip` 为例，MMCV 的 [RandomFlip](https://github.com/open-mmlab/mmcv/blob/5947178e855c23eea6103b1d70e1f8027f7b2ca8/mmcv/transforms/processing.py#L985) 相比旧版 MMDetection 的 [RandomFlip](https://github.com/open-mmlab/mmdetection/blob/3b72b12fe9b14de906d1363982b9fba05e7d47c1/mmdet/datasets/pipelines/transforms.py#L333)，需要继承 `BaseTransfrom`，将功能实现放在 `transforms` 方法，并将生成随机结果的部分放在单独的方法中，用 `cache_randomness` 包装。有关随机方法的包装相关功能，参见[相关文档](TODO)。

- MMDetection (旧）

```python
class RandomFlip:
    def __call__(self, results):
        """调用时进行随机翻转"""
        ...
        # 随机选择翻转方向
        cur_dir = np.random.choice(direction_list, p=flip_ratio_list)
        ...
        return results
```

- MMCV

```python
class RandomFlip(BaseTransfrom):
    def transform(self, results):
        """调用时进行随机翻转"""
        ...
        cur_dir = self._random_direction()
        ...
        return results

    @cache_randomness
    def _random_direction(self):
        """随机选择翻转方向"""
        ...
        return np.random.choice(direction_list, p=flip_ratio_list)
```
