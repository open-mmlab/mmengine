# Copyright (c) OpenMMLab. All rights reserved.
_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py',
    './test_get_external_cfg_base.py'
]

custom_hooks = [dict(type='mmdet.DetVisualizationHook')]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(_delete_=True, type='test.ToyLoss')
        )
    )
)
