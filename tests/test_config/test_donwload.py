# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmengine import get_config, get_model


def test_get_config():
    # TODO compatible with downstream codebase.
    get_config('mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco')
    get_config('mmdet::_base_/models/cascade_mask_rcnn_r50_fpn')
    cfg = get_config(
        'mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco', pretrained=True)
    get_config(
        'mmpose::face/2d_kpt_sview_rgb_img/deeppose/wflw/res50_wflw_256x256'
    )  # noqa E301
    assert cfg.model_path == 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa E301


def test_get_model():
    # TODO compatible with downstream codebase.
    get_model(
        'mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco',
        build_func_name='build_detector')
    get_model(
        'mmpose::face/2d_kpt_sview_rgb_img/deeppose/wflw/res50_wflw_256x256',  # noqa E301
        build_func_name='build_posenet')
    with pytest.raises(RuntimeError):
        get_model(
            'mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco',
            build_func_name='invalid_func')
