# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp

import torch.nn as nn

import mmengine.runner
from mmengine.utils import check_install_package, get_installed_path
from .collect_meta import (_get_cfg_meta, _get_external_cfg_base_path,
                           _parse_cfg_name, _parse_external_cfg_path)
from .config import Config


def get_config(cfg_name: str, pretrained: bool = False, suffix='.py', )\
        -> Config:
    """Get config from external package.

    Args:
        cfg_name (str): External relative config path with prefix
            'package::' and without suffix.
        pretrained (bool): Whether to save pretrained model path. If
            ``pretrained==True``, the url of pretrained model can be accessed
            by ``cfg.model_path``. Defaults to False.
        suffix (str): Suffix of ``cfg_name``. If cfg_name is a base
            cfg, the `suffix` will be used to get the absolute config path.
            Defaults to '.py'.

    Examples:
        >>> cfg = get_config('mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco',
        >>>                  pretrained=True)
        >>> # Equivalent to
        >>> Config.fromfile('/path/tofaster_rcnn_r50_fpn_1x_coco.py')
        >>> cfg.model_path
        https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

    Returns:
        Config: A `Config` parsed from external package.
    """ # noqa E301
    # Get package name and relative config path.
    package, cfg_name = _parse_external_cfg_path(cfg_name)
    # Check package is installed.
    check_install_package(package)
    package_path = get_installed_path(package)
    try:
        # Use `cfg_name` to search target config file.
        rel_cfg_file = _parse_cfg_name(cfg_name)
        cfg_meta = _get_cfg_meta(package_path, rel_cfg_file)
        cfg_path = osp.join(package_path, '.mim', cfg_meta['Config'])
        cfg = Config.fromfile(cfg_path)
        cfg.model_path = cfg_meta['Weights']
        if pretrained:
            assert 'Weights' in cfg_meta, ('Cannot find `Weights` in cfg_file'
                                           '.metafile.yml, please check the'
                                           'metafile')
    except ValueError:
        # Since the base config does not contain a metafile, the absolute
        # config is `osp.join(package_path, cfg_path_prefix, cfg_name)`
        cfg_path = _get_external_cfg_base_path(package_path, cfg_name + suffix)
        cfg = Config.fromfile(cfg_path)
    return cfg


def get_model(cfg_name: str,
              pretrained=False,
              build_func_name: str = 'build_model',
              suffix='.py',
              **kwargs) -> nn.Module:
    """Get built model from external package.

    Args:
        cfg_name (str): External relative config path with prefix
            'package::' and without suffix.
        pretrained (bool): Whether to load pretrained model. Defaults to False.
        build_func_name (str): Name of model build function. Defaults to
            'build_model'
        suffix (str): Suffix of ``cfg_name``. If cfg_name is a base
            cfg, the `suffix` will be used to get the absolute config path.
            Defaults to '.py'.

    Returns:
        nn.Module: Built model.
    """
    cfg = get_config(cfg_name, pretrained, suffix)
    package = cfg_name.split('::')[0]

    models_module = importlib.import_module(f'{package}.models')
    build_func = getattr(models_module, build_func_name, None)
    if build_func is None:
        raise RuntimeError(f'`{build_func_name}` is not defined in '
                           f'`{package}.models`')
    model = build_func(cfg.model, **kwargs)
    model = mmengine.runner.load_checkpoint(model, cfg.model_path)
    return model
