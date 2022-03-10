# Copyright (c) OpenMMLab. All rights reserved.
import os.path
import os.path as osp
import re
from typing import Tuple

from mmengine.fileio import load
from mmengine.utils import check_file_exist


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


def _get_external_cfg_path(package_path: str, cfg_file: str):
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


def _parse_external_cfg_path(rel_cfg_path: str) -> Tuple[str, str]:
    """Get package name and relative config path.

    Args:
        rel_cfg_path (str): External relative config path with 'package::'.

    Returns:
        Tuple(str, str): Package name and relative config path.
    """
    if re.match(r'\w*::\w*/\w*', rel_cfg_path) is None:
        raise ValueError('`_parse_external_cfg_path` is used for parse '
                         'external package, please specify the package name '
                         'and relative config path, just like '
                         '`mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco` ')
    package_cfg = rel_cfg_path.split('::')
    if len(package_cfg) > 2:
        raise ValueError('`::` should only be used to separate package and '
                         'config name, but found multiple `::` in '
                         f'{rel_cfg_path}')
    package, rel_cfg_path = package_cfg
    assert package in PKG2PROJECT, 'mmengine does not support to load ' \
                                   f'{package} config.'
    package = PKG2PROJECT[package]
    return package, rel_cfg_path


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
