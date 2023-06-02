# 跨库调用模块

通过使用 MMEngine 的[注册器（Registry）](registry.md)和[配置文件（Config）](config.md)，用户可以实现跨软件包的模块构建。
例如，在 [MMDetection](https://github.com/open-mmlab/mmdetection) 中使用 [MMPretrain](https://github.com/open-mmlab/mmpretrain) 的 Backbone，或者在 [MMRotate](https://github.com/open-mmlab/mmrotate) 中使用 [MMDetection](https://github.com/open-mmlab/mmdetection) 的 Transform，或者在 [MMTracking](https://github.com/open-mmlab/mmtracking) 中使用 [MMDetection](https://github.com/open-mmlab/mmdetection) 的 Detector。
一般来说，同类模块都可以进行跨库调用，只需要在配置文件的模块类型前加上软件包名的前缀即可。下面举几个常见的例子：

## 跨库调用 Backbone:

以在 MMDetection 中调用 MMPretrain 的 ConvNeXt 为例，首先需要在配置中加入 `custom_imports` 字段将 MMPretrain 的 Backbone 添加进注册器，然后只需要在 Backbone 的配置中的 `type` 加上 MMPretrain 的软件包名 `mmpretrain` 作为前缀，即 `mmpretrain.ConvNeXt` 即可：

```python
# 使用 custom_imports 将 mmpretrain 的 models 添加进注册器
custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)

model = dict(
  type='MaskRCNN',
  data_preprocessor=dict(...),
  backbone=dict(
      type='mmpretrain.ConvNeXt',  # 添加 mmpretrain 前缀完成跨库调用
      arch='tiny',
      out_indices=[0, 1, 2, 3],
      drop_path_rate=0.4,
      layer_scale_init_value=1.0,
      gap_before_final_norm=False,
      init_cfg=dict(
          type='Pretrained',
          checkpoint=
          'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth',
          prefix='backbone.')),
  neck=dict(...),
  rpn_head=dict(...))
```

## 跨库调用 Transform:

与上文的跨库调用 Backbone 一样，使用 custom_imports 和添加前缀即可实现跨库调用：

```python
# 使用 custom_imports 将 mmdet 的 transforms 添加进注册器
custom_imports = dict(imports=['mmdet.datasets.transforms'], allow_failed_imports=False)

# 添加 mmdet 前缀完成跨库调用
train_pipeline=[
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 2014), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs')
]
```

## 跨库调用 Detector:

跨库调用算法是一个比较复杂的例子，一个算法会包含多个子模块，因此每个子模块也需要在`type`中增加前缀，以在 MMTracking 中调用 MMDetection 的 YOLOX 为例：

```python
# 使用 custom_imports 将 mmdet 的 models 添加进注册器
custom_imports = dict(imports=['mmdet.models'], allow_failed_imports=False)
model = dict(
    type='mmdet.YOLOX',
    backbone=dict(type='mmdet.CSPDarknet', deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        type='mmdet.YOLOXPAFPN',
        in_channels=[320, 640, 1280],
        out_channels=320,
        num_csp_blocks=4),
    bbox_head=dict(
        type='mmdet.YOLOXHead', num_classes=1, in_channels=320, feat_channels=320),
    train_cfg=dict(assigner=dict(type='mmdet.SimOTAAssigner', center_radius=2.5)))
```

为了避免给每个子模块手动增加前缀，配置文件中引入了 `_scope_` 关键字，当某一模块的配置中添加了 `_scope_` 关键字后，该模块配置文件下面的所有子模块配置都会从该关键字所对应的软件包内去构建：

```python
# 使用 custom_imports 将 mmdet 的 models 添加进注册器
custom_imports = dict(imports=['mmdet.models'], allow_failed_imports=False)
model = dict(
    _scope_='mmdet',  # 使用 _scope_ 关键字，避免给所有子模块添加前缀
    type='YOLOX',
    backbone=dict(type='CSPDarknet', deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[320, 640, 1280],
        out_channels=320,
        num_csp_blocks=4),
    bbox_head=dict(
        type='YOLOXHead', num_classes=1, in_channels=320, feat_channels=320),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)))
```

以上这两种写法互相等价。

若希望了解更多关于注册器和配置文件的内容，请参考[配置文件教程](config.md)和[注册器教程](registry.md)
