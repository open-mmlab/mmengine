import os.path as osp
from typing import Tuple

from mmengine.utils import check_file_exist
from mmengine.fileio import load


def _get_cfg_meta(package_path: str, cfg_dir: str, cfg_file: str) -> dict:
    """Get target meta information from 'metafile.yml' of external package.

    Args:
        package_path (str): Path of external package.
        cfg_dir (str): Name of experiment directory.
        cfg_file (str): Name of experiment config.

    Returns:
        dict: Meta information of target experiment.
    """
    meta_index_path = osp.join(package_path, 'model-index.yml')
    meta_index = load(meta_index_path)
    for meta_path in meta_index['Import']:
        # `meta_path` endswith `metafile.yml`
        meta_path = osp.join(package_path, meta_path)
        if cfg_dir in meta_path and osp.isfile(meta_path):
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
    """ Get config path from 'metafile.yml' of external package.

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
    cfg_path = osp.join(package_path, 'configs', rel_cfg_path)
    check_file_exist(cfg_path)
    return cfg_path


def _parse_external_cfg_path(rel_cfg_path: str) -> Tuple[str, str]:
    """

    Args:
        rel_cfg_path (str): External relative config path with 'package::'.

    Returns:
        Tuple(str, str): Package name and relative config path.
    """
    if '::' not in rel_cfg_path:
        raise ValueError('`get_config` is used for loading config file cross '
                         'package, please specify the name of target package, '
                         'just like `mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco` ') # noqa E301
    package_cfg = rel_cfg_path.split('::')
    if len(package_cfg) > 2:
        raise ValueError('`::` should only be used to separate package and '
                         'config name, but found multiple `::` in '
                         f'{rel_cfg_path}')
    package, rel_cfg_path = package_cfg
    return package, rel_cfg_path


def _parse_rel_cfg_path(rel_cfg_path: str) -> Tuple[str, str]:
    """Get the experiment dir and experiment config name.

    Args:
        rel_cfg_path (str): External relative config path with prefix
            'package::' and without suffix.

    Returns:
        Tuple(str, str): Experiment dir and experiment config name.
    """
    rel_cfg_path_list = rel_cfg_path.split('/')
    assert len(rel_cfg_path_list) == 2, \
        '`rel_cfg_path` should only contain config file and config name.'
    rel_cfg_dir, rel_cfg_file = rel_cfg_path_list
    return rel_cfg_dir, rel_cfg_file
