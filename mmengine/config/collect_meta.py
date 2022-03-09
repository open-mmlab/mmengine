# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import re
from typing import Tuple

from mmengine.fileio import load
from mmengine.utils import check_file_exist


def _get_cfg_meta(package_path: str, cfg_dir: str, cfg_file: str) -> dict:
    """Get target meta information from 'metafile.yml' of external package.

    Args:
        package_path (str): Path of external package.
        cfg_dir (str): Name of experiment directory.
        cfg_file (str): Name of experiment config.

    Returns:
        dict: Meta information of target experiment.
    """
    meta_index_path = osp.join(package_path, '.mmengine', 'model-index.yml')
    meta_index = load(meta_index_path)
    for meta_path in meta_index['Import']:
        # `meta_path` endswith `metafile.yml`
        meta_path = osp.join(package_path, '.mmengine', meta_path)
        if cfg_dir in meta_path:
            check_file_exist(meta_path)
            cfg_meta = load(meta_path)
            break
    else:
        raise FileNotFoundError(f'{cfg_dir} is not recorded in '
                                f'{meta_index_path}')
    assert 'Models' in cfg_meta, f'Cannot find `Model` in {meta_path}, ' \
                                 'please check the format metafile.'

    for model_cfg in cfg_meta['Models']:
        if cfg_file in model_cfg['Config'].split('/')[-1]:
            return model_cfg
    else:
        raise FileNotFoundError(f'{cfg_file} is not recorded in {meta_path}')


def _get_external_cfg_path(package_path: str, cfg_dir: str, cfg_file: str):
    """Get relative config path from 'metafile.yml' of external package.

    Args:
        package_path (str): Path of external package.
        cfg_dir (str): Name of experiment directory.
        cfg_file (str): Name of experiment config.

    Returns:
        str: Absolute config path from external package.
    """
    model_cfg = _get_cfg_meta(package_path, cfg_dir, cfg_file)
    cfg_path = osp.join(package_path, model_cfg['Config'])
    check_file_exist(cfg_path)
    return cfg_path


def _get_external_cfg_base_path(package_path: str, rel_cfg_path: str):
    """ Get base config path from external package.
    Args:
        package_path (str): Path of external package.
        rel_cfg_path (str): External relative config path with 'package::'.

    Returns:
        str: Absolute config path from external package.
    """
    cfg_path = osp.join(package_path, '.mmengine', 'configs', rel_cfg_path)
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
    return package, rel_cfg_path


def _parse_rel_cfg_path(rel_cfg_path: str) -> Tuple[str, str]:
    """Get the experiment dir and experiment config name. This function is only
    used for getting non-base config from external packages.

    Args:
        rel_cfg_path (str): External relative config path with prefix
            'package::'.

    Returns:
        Tuple(str, str): Experiment dir and experiment config name.
    """
    rel_cfg_path_list = rel_cfg_path.split('/')
    assert len(rel_cfg_path_list) == 2, \
        '`rel_cfg_path` should only contain config file and config name.'
    rel_cfg_dir, rel_cfg_file = rel_cfg_path_list
    return rel_cfg_dir, rel_cfg_file
