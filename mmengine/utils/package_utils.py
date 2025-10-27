# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from typing import Any


def is_installed(package: str) -> bool:
    """Check package whether installed.

    Args:
        package (str): Name of package to be checked.
    """
    import importlib.util

    # First check if it's an importable module
    spec = importlib.util.find_spec(package)
    if spec is not None and spec.origin is not None:
        return True

    # If not found as module, check if it's a distribution package
    try:
        distribution(package)
        return True
    except PackageNotFoundError:
        return False


def get_installed_path(package: str) -> str:
    """Get installed path of package.

    Args:
        package (str): Name of package.

    Example:
        >>> get_installed_path('mmcls')
        >>> '.../lib/python3.7/site-packages/mmcls'
    """
    import importlib.util

    # if the package name is not the same as module name, module name should be
    # inferred. For example, mmcv-full is the package name, but mmcv is module
    # name. If we want to get the installed path of mmcv-full, we should concat
    # the pkg.location and module name
    # Try to get location from distribution package metadata
    location = None
    try:
        dist = distribution(package)
        locate_result: Any = dist.locate_file('')
        location = str(locate_result.parent)
    except PackageNotFoundError:
        pass

    # If distribution package not found, try to find via importlib
    if location is None:
        spec = importlib.util.find_spec(package)
        if spec is not None:
            if spec.origin is not None:
                return osp.dirname(spec.origin)
            else:
                # `get_installed_path` cannot get the installed path of
                # namespace packages
                raise RuntimeError(
                    f'{package} is a namespace package, which is invalid '
                    'for `get_install_path`')
        else:
            raise PackageNotFoundError(f'Package {package} is not installed')

    # Check if package directory exists in the location
    possible_path = osp.join(location, package)
    if osp.exists(possible_path):
        return possible_path
    else:
        return osp.join(location, package2module(package))


def package2module(package: str) -> str:
    """Infer module name from package.

    Args:
        package (str): Package to infer module name.
    """
    dist = distribution(package)

    # In importlib.metadata,
    # top-level modules are in dist.read_text('top_level.txt')
    top_level_text = dist.read_text('top_level.txt')
    if top_level_text is not None:
        lines = top_level_text.strip().split('\n')
        if lines:
            module_name = lines[0].strip()
            return module_name
    raise ValueError(f'can not infer the module name of {package}')


def call_command(cmd: list) -> None:
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        raise e  # type: ignore


def install_package(package: str):
    if not is_installed(package):
        call_command(['python', '-m', 'pip', 'install', package])
