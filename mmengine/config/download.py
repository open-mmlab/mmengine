# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp

import torch.nn as nn

from mmengine.runner import load_checkpoint
from mmengine.utils import check_install_package, get_installed_path
from .collect_meta import (_get_cfg_meta, _get_external_cfg_base_path,
                           _parse_external_cfg_path, _parse_rel_cfg_path)
from .config import Config


def get_config(cfg_name: str, suffix='.py', pretrained: bool = False)\
        -> Config:
    """Get config from external package.

    Args:
        cfg_name (str): External relative config path with prefix
            'package::' and without suffix.
        suffix (str): Suffix of ``cfg_name``. If cfg_name is a base
            cfg, the `suffix` will be used to get the absolute config path.
            Defaults to '.py'.
        pretrained (bool): Whether to save pretrained model path. If
            ``pretrained==True``, the url of pretrained model can be accessed
            by ``cfg.model_path``. Defaults to False.

    Examples:
        >>> cfg = get_config('mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco',
        >>>                  pretrained=True)
        >>> # Equivalent to
        >>> Config.fromfile(/path/tofaster_rcnn_r50_fpn_1x_coco.py)
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
    # Since the base config does not contain a metafile, the absolute config
    # is `osp.join(package_path, cfg_path_prefix, cfg_name)`
    if '__base__' in cfg_name:
        cfg_path = _get_external_cfg_base_path(package_path, cfg_name)
        return Config.fromfile(cfg_path + suffix)
    # Use `rel_cfg_dir` to search `model-index.yml`, `rel_cfg_file` to search
    # specific metafile.yml.
    rel_cfg_dir, rel_cfg_file = _parse_rel_cfg_path(cfg_name)
    cfg_meta = _get_cfg_meta(package_path, rel_cfg_dir, rel_cfg_file)
    cfg_path = osp.join(package_path, cfg_meta['Config'])
    cfg = Config.fromfile(cfg_path)
    if pretrained:
        assert 'Weights' in cfg_meta, 'Cannot find `Weights` in ' \
                                      'cfg_file.metafile.yml, please check ' \
                                      'the metafile'
        cfg.model_path = cfg_meta['Weights']
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
        suffix (str): Suffix of ``cfg_name``. If cfg_name is a base
            cfg, the `suffix` will be used to get the absolute config path.
            Defaults to '.py'.
        build_func_name (str): Name of model build function. Defaults to
            'build_model'
        pretrained (bool): Whether to load pretrained model. Defaults to False.

    Returns:
        nn.Module: Built model.
    """
    cfg = get_config(cfg_name, suffix, pretrained)
    package = cfg_name.split('::')[0]

    try:
        models_module = importlib.import_module(f'{package}.models')
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError('Cannot find `models` moduls in package '
                                  f'`{package}`: {e}')
    build_func = getattr(models_module, build_func_name, None)
    if build_func is None:
        raise RuntimeError(f'`{build_func_name}` is not defined in '
                           f'`{package}.models`')
    model = build_func(cfg.model, **kwargs)
    if pretrained:
        assert hasattr(cfg, 'model_path'), 'Cannot find pretrained model. ' \
                                           f'Please ensure {cfg_name} ' \
                                           'is not a base config.'
        model = load_checkpoint(model, cfg.model_path)
    return model
