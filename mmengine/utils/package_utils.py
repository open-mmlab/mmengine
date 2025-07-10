# Copyright (c) OpenMMLab. All rights reserved.
import subprocess


def is_installed(package: str) -> bool:
    """Check whether a package is installed.

    Args:
        package (str): Name of the package to check.

    Returns:
        bool: True if the package is installed, False otherwise.
    """
    import importlib.util
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        from importlib_metadata import PackageNotFoundError  # type: ignore
        from importlib_metadata import version  # type: ignore

    try:
        version(package)
        return True
    except PackageNotFoundError:
        spec = importlib.util.find_spec(package)
        return spec is not None and spec.origin is not None


def get_installed_path(package: str) -> str:
    """Get installed path of a package.

    Args:
        package (str): Name of the package.

    Returns:
        str: The path to the installed package.

    Example:
        >>> get_installed_path('mmcls')
        '.../lib/python3.10/site-packages/mmcls'
    """
    import importlib.util
    import os.path as osp

    try:
        # Use importlib.metadata for distribution info (Python 3.8+)
        try:
            from importlib.metadata import PackageNotFoundError, distribution
        except ImportError:
            from importlib_metadata import (  # type: ignore
                PackageNotFoundError, distribution)

        try:
            dist = distribution(package)
        except PackageNotFoundError:
            # If not installed as a distribution, try to find the module spec
            spec = importlib.util.find_spec(package)
            if spec and spec.origin:
                return osp.dirname(spec.origin)
            elif spec:
                raise RuntimeError(
                    f'{package} is a namespace package, which is invalid '
                    'for `get_installed_path`')
            else:
                raise ImportError(f'Package {package} is not installed.')

        # Try to infer the top-level module name from top_level.txt
        top_level = dist.read_text('top_level.txt')
        if top_level:
            module_name = top_level.split('\n')[0].strip()
            possible_path = osp.join(dist.locate_file(''), module_name)
            if osp.exists(possible_path):
                return possible_path

        # Fallback: try to find the module by spec
        spec = importlib.util.find_spec(package)
        if spec and spec.origin:
            return osp.dirname(spec.origin)
        else:
            raise PackageNotFoundError(
                f'Cannot determine installed path for {package}.')

    except Exception:
        raise PackageNotFoundError(
            f'Cannot determine installed path for {package}.')


def package2module(package: str):
    """Infer module name from package.

    Args:
        package (str): Package to infer module name.

    Returns:
        str: The top-level module name for the given package.

    Raises:
        ValueError: If the module name cannot be inferred.
    """
    try:
        from importlib.metadata import PackageNotFoundError, distribution
    except ImportError:
        from importlib_metadata import PackageNotFoundError  # type: ignore
        from importlib_metadata import distribution  # type: ignore

    try:
        dist = distribution(package)
    except PackageNotFoundError:
        raise ValueError(f'Package {package} is not installed.')

    if dist.read_text('top_level.txt'):
        module_name = dist.read_text('top_level.txt').split(  # type: ignore
            '\n')[0].strip()
        if module_name:
            return module_name
    raise ValueError(f'Cannot infer the module name of {package}')


def call_command(cmd: list) -> None:
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        raise e  # type: ignore


def install_package(package: str):
    if not is_installed(package):
        call_command(['python', '-m', 'pip', 'install', package])
