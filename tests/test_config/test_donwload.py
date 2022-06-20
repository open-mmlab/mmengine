# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmengine import get_config, get_model
from mmengine import Config


def test_get_config():
    # Test load base config.
    base_cfg = get_config('mmdet::_base_/models/fast_rcnn_r50_fpn.py')
    test_base_cfg = Config.fromfile(
        '../data/config/py_config/base_faster_rcnn.py')
    assert test_base_cfg._cfg_dict == base_cfg._cfg_dict

    # Test load faster_rcnn config
    cfg = get_config('mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
    test_cfg = Config.fromfile(
        '../data/config/py_config/faster_rcnn_r50_fpn_1x_coco.py')
    assert cfg._cfg_dict == test_cfg._cfg_dict

    # Test pretrained
    cfg = get_config(
        'mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', pretrained=True)
    assert cfg.model_path == 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa E301

    # Test load mmpose
    get_config(
        ('mmpose::face/2d_kpt_sview_rgb_img/deeppose/wflw/res50_wflw_256x256'
         '.py'))


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
