# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest

from mmengine import Config, get_config, get_model
from mmengine.utils import get_installed_path, is_installed

data_path = osp.join(osp.dirname(osp.dirname(__file__)), 'data/')


# mmdet has a more typical config structure, while mmpose has a complex
# config structure
@pytest.mark.skipif(
    not (is_installed('mmdet') and is_installed('mmpose')),
    reason='mmdet and mmpose should be installed')
def test_get_config():
    # Test load base config.
    base_cfg = get_config('mmdet::_base_/models/faster_rcnn_r50_fpn.py')
    package_path = get_installed_path('mmdet')
    test_base_cfg = Config.fromfile(
        osp.join(package_path, '.mim',
                 'configs/_base_/models/faster_rcnn_r50_fpn.py'))
    assert test_base_cfg._cfg_dict == base_cfg._cfg_dict

    # Test load faster_rcnn config
    cfg = get_config('mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
    test_cfg = Config.fromfile(
        osp.join(package_path, '.mim',
                 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'))
    assert cfg._cfg_dict == test_cfg._cfg_dict

    # Test pretrained
    cfg = get_config(
        'mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', pretrained=True)
    assert cfg.model_path == 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa E301

    # Test load mmpose
    get_config(
        'mmpose::face/2d_kpt_sview_rgb_img/deeppose/wflw/res50_wflw_256x256'
        '.py')


@pytest.mark.skipif(
    not (is_installed('mmdet') and is_installed('mmpose')),
    reason='mmdet and mmpose should be installed')
def test_get_model():
    # TODO compatible with downstream codebase.
    get_model('mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
    get_model(
        'mmpose::face/2d_kpt_sview_rgb_img/deeppose/wflw/res50_wflw_256x256'
        '.py')
    with pytest.raises(RuntimeError):
        get_model('mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
