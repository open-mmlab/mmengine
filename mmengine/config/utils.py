# Copyright (c) OpenMMLab. All rights reserved.
import ast
import os.path as osp
import re
import warnings
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


def _get_cfg_metainfo(package_path: str, cfg_path: str) -> dict:
    """Get target meta information from all 'metafile.yml' defined in `mode-
    index.yml` of external package.

    Args:
        package_path (str): Path of external package.
        cfg_path (str): Name of experiment config.

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
            if 'Config' not in model_cfg:
                warnings.warn(f'There is not `Config` define in {model_cfg}')
                continue
            cfg_name = model_cfg['Config'].partition('/')[-1]
            # Some config could have multiple weights, we only pick the
            # first one.
            if cfg_name in cfg_dict:
                continue
            cfg_dict[cfg_name] = model_cfg
    if cfg_path not in cfg_dict:
        raise ValueError(f'Expected configs: {cfg_dict.keys()}, but got '
                         f'{cfg_path}')
    return cfg_dict[cfg_path]


def _get_external_cfg_path(package_path: str, cfg_file: str) -> str:
    """Get config path of external package.

    Args:
        package_path (str): Path of external package.
        cfg_file (str): Name of experiment config.

    Returns:
        str: Absolute config path from external package.
    """
    cfg_file = cfg_file.split('.')[0]
    model_cfg = _get_cfg_metainfo(package_path, cfg_file)
    cfg_path = osp.join(package_path, model_cfg['Config'])
    check_file_exist(cfg_path)
    return cfg_path


def _get_external_cfg_base_path(package_path: str, cfg_name: str) -> str:
    """Get base config path of external package.

    Args:
        package_path (str): Path of external package.
        cfg_name (str): External relative config path with 'package::'.

    Returns:
        str: Absolute config path from external package.
    """
    cfg_path = osp.join(package_path, '.mim', 'configs', cfg_name)
    check_file_exist(cfg_path)
    return cfg_path


def _get_package_and_cfg_path(cfg_path: str) -> Tuple[str, str]:
    """Get package name and relative config path.

    Args:
        cfg_path (str): External relative config path with 'package::'.

    Returns:
        Tuple[str, str]: Package name and config path.
    """
    if re.match(r'\w*::\w*/\w*', cfg_path) is None:
        raise ValueError(
            '`_get_package_and_cfg_path` is used for get external package, '
            'please specify the package name and relative config path, just '
            'like `mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py`')
    package_cfg = cfg_path.split('::')
    if len(package_cfg) > 2:
        raise ValueError('`::` should only be used to separate package and '
                         'config name, but found multiple `::` in '
                         f'{cfg_path}')
    package, cfg_path = package_cfg
    assert package in PKG2PROJECT, 'mmengine does not support to load ' \
                                   f'{package} config.'
    package = PKG2PROJECT[package]
    return package, cfg_path


class RemoveAssignFromAST(ast.NodeTransformer):
    """Remove Assign node if the target's name match the key.

    Args:
        key (str): The target name of the Assign node.
    """

    def __init__(self, key):
        self.key = key

    def visit_Assign(self, node):
        if (isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == self.key):
            return None
        else:
            return node
