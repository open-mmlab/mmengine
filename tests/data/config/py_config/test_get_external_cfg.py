# Copyright (c) OpenMMLab. All rights reserved.
_base_ = [
    'mmdet::_base_/models/faster_rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]
