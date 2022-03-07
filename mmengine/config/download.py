import os.path as osp
import importlib

import torch.nn as nn

from mmengine import (Config, get_installed_path, check_install_package)
from .collect_meta import (_parse_external_cfg_path, _parse_rel_cfg_path,
                           _get_cfg_meta, _get_external_cfg_base_path)


def get_config(rel_cfg_path: str, suffix='.py') -> Config:
    """Get config from external package.

    Args:
        rel_cfg_path (str): External relative config path with prefix
            'package::' and without suffix.
        suffix (str): Suffix of ``rel_cfg_path``. If rel_cfg_path is a base
            cfg, the `suffix` will be used to get the absolute config path.
            Defaults to '.py'.

    Returns:
        Config: A `Config` parsed from external package.
    """
    # Get package name and relative config path.
    package, rel_cfg_path = _parse_external_cfg_path(rel_cfg_path)
    # Check package is installed.
    check_install_package(package)
    package_path = get_installed_path(package)
    # Since the base config does not contain a metafile, the absolute config
    # is `osp.join(package_path, cfg_path_prefix, rel_cfg_path)`
    if '__base__' in rel_cfg_path:
        cfg_path = _get_external_cfg_base_path(package_path, rel_cfg_path)
        return Config.fromfile(cfg_path + suffix)
    # Use `rel_cfg_dir` to search `model-index.yml`, `rel_cfg_file` to search
    # specific metafile.yml.
    rel_cfg_dir, rel_cfg_file = _parse_rel_cfg_path(rel_cfg_path)
    cfg_meta = _get_cfg_meta(package_path, rel_cfg_dir, rel_cfg_file)
    cfg_path = osp.join(package_path, cfg_meta['Config'])
    cfg = Config.fromfile(cfg_path)
    assert 'Weights' in cfg_meta, 'Cannot find `Weights` in ' \
                                  'cfg_file.metafile.yml, please check the ' \
                                  'metafile'
    cfg.model_path = cfg_meta['Weights']
    return cfg


def get_model(rel_cfg_path: str, suffix='.py',
              build_func_name: str = 'build_model', **kwargs) -> nn.Module:
    """Get built model from external package.

    Args:
        rel_cfg_path (str): External relative config path with prefix
            'package::' and without suffix.
        suffix (str): Suffix of ``rel_cfg_path``. If rel_cfg_path is a base
            cfg, the `suffix` will be used to get the absolute config path.
            Defaults to '.py'.
        build_func_name: Name of model build function. Defaults to
            'build_model'

    Returns:
        nn.Module: Built model.
    """
    cfg = get_config(rel_cfg_path, suffix)
    package = rel_cfg_path.split('::')[0]

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
    return model
