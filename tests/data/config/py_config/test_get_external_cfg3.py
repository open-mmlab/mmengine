_base_ = [
    'mmdet::_base_/models/faster_rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

backbone = _base_.model.backbone
backbone.style = 'pytorch'
_base_.model.roi_head.bbox_head.loss_cls = dict(type='test.ToyLoss')

student = dict(detector=_base_.model)