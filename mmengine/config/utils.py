# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp
import re
from typing import Tuple

import torch.nn as nn

import mmengine.runner
from mmengine.fileio import load
from mmengine.utils import (check_file_exist, check_install_package,
                            get_installed_path)
from .config import Config

PKG2PROJECT = {
    'mmcls': 'mmcls',
    'mmdet': 'mmdet',
    'mmdet3d': 'mmdet3d',
    'mmseg': 'mmsegmentation',
    'mmaction2': 'mmaction2',
    'mmtrack': 'mmtrack',
    'mmpose': 'mmpose',
    'mmedit': 'mmedit',
    'mmocr': 'mmocr',
    'mmgen': 'mmgen',
    'mmfewshot': 'mmfewshot',
    'mmrazor': 'mmrazor',
    'mmflow': 'mmflow',
    'mmhuman3d': 'mmhuman3d',
    'mmrotate': 'mmrotate',
    'mmselfsup': 'mmselfsup',
}


def _get_cfg_meta(package_path: str, cfg_file: str) -> dict:
    """Get target meta information from 'metafile.yml' of external package.

    Args:
        package_path (str): Path of external package.
        cfg_file (str): Name of experiment config.

    Returns:
        dict: Meta information of target experiment.
    """
    meta_index_path = osp.join(package_path, '.mim', 'model-index.yml')
    meta_index = load(meta_index_path)
    cfg_dict = dict()
    for meta_path in meta_index['Import']:
        meta_path = osp.join(package_path, '.mim', meta_path)
        cfg_meta = load(meta_path)
        for model_cfg in cfg_meta['Models']:
            cfg_name = model_cfg['Config'].split('/')[-1].strip('.py')
            cfg_dict[cfg_name] = model_cfg
    if cfg_file not in cfg_dict:
        raise ValueError(f'Expected configs: {cfg_dict.keys()}, but got '
                         f'{cfg_file}')
    return cfg_dict[cfg_file]


def _get_external_cfg_path(package_path: str, cfg_file: str) -> str:
    """Get relative config path from 'metafile.yml' of external package.

    Args:
        package_path (str): Path of external package.
        cfg_file (str): Name of experiment config.

    Returns:
        str: Absolute config path from external package.
    """
    cfg_file = cfg_file.split('.')[0]
    model_cfg = _get_cfg_meta(package_path, cfg_file)
    cfg_path = osp.join(package_path, model_cfg['Config'])
    check_file_exist(cfg_path)
    return cfg_path


def _get_external_cfg_base_path(package_path: str, cfg_name: str):
    """ Get base config path from external package.
    Args:
        package_path (str): Path of external package.
        cfg_name (str): External relative config path with 'package::'.

    Returns:
        str: Absolute config path from external package.
    """
    cfg_path = osp.join(package_path, '.mim', 'configs', cfg_name)
    check_file_exist(cfg_path)
    return cfg_path


def _parse_external_cfg_path(cfg_name: str) -> Tuple[str, str]:
    """Get package name and relative config path.

    Args:
        cfg_name (str): External relative config path with 'package::'.

    Returns:
        Tuple(str, str): Package name and relative config path.
    """
    if re.match(r'\w*::\w*/\w*', cfg_name) is None:
        raise ValueError('`_parse_external_cfg_path` is used for parse '
                         'external package, please specify the package name '
                         'and relative config path, just like '
                         '`mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco` ')
    package_cfg = cfg_name.split('::')
    if len(package_cfg) > 2:
        raise ValueError('`::` should only be used to separate package and '
                         'config name, but found multiple `::` in '
                         f'{cfg_name}')
    package, cfg_name = package_cfg
    assert package in PKG2PROJECT, 'mmengine does not support to load ' \
                                   f'{package} config.'
    package = PKG2PROJECT[package]
    return package, cfg_name


def _parse_cfg_name(cfg_name: str) -> str:
    """Get the econfig name.

    Args:
        cfg_name (str): External relative config path.

    Returns:
        str: The config name.
    """
    cfg_path_list = cfg_name.split('/')
    cfg_name = cfg_path_list[-1]
    return cfg_name


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
