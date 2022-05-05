# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import os.path as osp
from typing import Optional

from mmengine.fileio import dump
from . import root
from .registry import Registry


def traverse_registry_tree(registry: Registry, verbose: bool = True) -> list:
    """Traverse the whole registry tree from any given node, and collect
    information of all registered modules in this registry tree.

    Args:
        registry (Registry): a registry node in the registry tree.
        verbose (bool): Whether to print log. Default: True

    Returns:
        list: Statistic results of all modules in each node of the registry
        tree.
    """
    root_registry = registry.root
    modules_info = []

    def _dfs_registry(_registry):
        if isinstance(_registry, Registry):
            num_modules = len(_registry.module_dict)
            scope = _registry.scope
            registry_info = dict(num_modules=num_modules, scope=scope)
            for name, registered_class in _registry.module_dict.items():
                folder = '/'.join(registered_class.__module__.split('.')[:-1])
                if folder in registry_info:
                    registry_info[folder].append(name)
                else:
                    registry_info[folder] = [name]
            if verbose:
                print(f"Find {num_modules} modules in {scope}'s "
                      f"'{_registry.name}' registry ")
            modules_info.append(registry_info)
        else:
            return
        for _, child in _registry.children.items():
            _dfs_registry(child)

    _dfs_registry(root_registry)
    return modules_info


def count_registered_modules(save_path: Optional[str] = None,
                             verbose: bool = True) -> dict:
    """Scan all modules in MMEngine's root and child registries and dump to
    json.

    Args:
        save_path (str, optional): Path to save the json file.
        verbose (bool): Whether to print log. Default: True
    Returns:
        dict: Statistic results of all registered modules.
    """
    registries_info = {}
    # traverse all registries in MMEngine
    for item in dir(root):
        if not item.startswith('__'):
            registry = getattr(root, item)
            if isinstance(registry, Registry):
                registries_info[item] = traverse_registry_tree(
                    registry, verbose)
    scan_data = dict(
        scan_date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        registries=registries_info)
    if verbose:
        print('Finish registry analysis, got: ', scan_data)
    if save_path is not None:
        json_path = osp.join(save_path, 'modules_statistic_results.json')
        dump(scan_data, json_path, indent=2)
        print(f'Result has been saved to {json_path}')
    return scan_data
