# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import subprocess
from typing import Any

# Import distribution function with fallback for older Python versions
try:
    from importlib.metadata import PackageNotFoundError, distribution
except ImportError:
    from importlib_metadata import (  # type: ignore[import-untyped, no-redef, import-not-found]  # noqa: E501
        PackageNotFoundError, distribution)


def is_installed(package: str) -> bool:
    """Check package whether installed.

    Args:
        package (str): Name of package to be checked.
    """
    # Use importlib.metadata instead of deprecated pkg_resources
    # importlib.metadata is available in Python 3.8+
    # For Python 3.7, importlib_metadata backport can be used
    import importlib.util

    import pkg_resources  # type: ignore

    # refresh the pkg_resources
    # more datails at https://github.com/pypa/setuptools/issues/373
    importlib.reload(pkg_resources)
    try:
        distribution(package)
        return True
    except Exception:
        # If distribution not found, check if module can be imported
        spec = importlib.util.find_spec(package)
        if spec is None:
            return False
        elif spec.origin is not None:
            return True
        else:
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
    try:
        dist = distribution(package)
        # In importlib.metadata, we use dist.locate_file() or files
        if hasattr(dist, 'locate_file'):
            # Python 3.9+
            # locate_file returns PathLike, need to access parent
            locate_result: Any = dist.locate_file('')
            location = str(locate_result.parent)
        elif hasattr(dist, '_path'):
            # Python 3.8 - _path is a pathlib.Path object
            # We know _path exists because we checked with hasattr
            dist_any: Any = dist
            location = str(dist_any._path.parent)  # type: ignore[attr-defined]
        else:
            # Fallback: try to find via importlib
            spec = importlib.util.find_spec(package)
            if spec is not None and spec.origin is not None:
                return osp.dirname(spec.origin)
            raise RuntimeError(
                f'Cannot determine installation path for {package}')
    except PackageNotFoundError as e:
        # if the package is not installed, package path set in PYTHONPATH
        # can be detected by `find_spec`
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
            raise e

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
    if top_level_text:
        module_name = top_level_text.split('\n')[0]
        return module_name
    else:
        raise ValueError(f'can not infer the module name of {package}')


def call_command(cmd: list) -> None:
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        raise e  # type: ignore


def install_package(package: str):
    if not is_installed(package):
        call_command(['python', '-m', 'pip', 'install', package])
