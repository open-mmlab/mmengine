# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp
import subprocess


def is_installed(package: str) -> bool:
    """Check package whether installed.

    Args:
        package (str): Name of package to be checked.
    """
    # When executing `import mmengine.runner`,
    # pkg_resources will be imported and it takes too much time.
    # Therefore, import it in function scope to save time.
    import pkg_resources
    from pkg_resources import get_distribution

    # refresh the pkg_resources
    # more datails at https://github.com/pypa/setuptools/issues/373
    importlib.reload(pkg_resources)
    try:
        get_distribution(package)
        return True
    except pkg_resources.DistributionNotFound:
        return False


def get_installed_path(package: str) -> str:
    """Get installed path of package.

    Args:
        package (str): Name of package.

    Example:
        >>> get_installed_path('mmcls')
        >>> '.../lib/python3.7/site-packages/mmcls'
    """
    from pkg_resources import get_distribution

    # if the package name is not the same as module name, module name should be
    # inferred. For example, mmcv-full is the package name, but mmcv is module
    # name. If we want to get the installed path of mmcv-full, we should concat
    # the pkg.location and module name
    pkg = get_distribution(package)
    possible_path = osp.join(pkg.location, package)
    if osp.exists(possible_path):
        return possible_path
    else:
        return osp.join(pkg.location, package2module(package))


def package2module(package: str):
    """Infer module name from package.

    Args:
        package (str): Package to infer module name.
    """
    from pkg_resources import get_distribution
    pkg = get_distribution(package)
    if pkg.has_metadata('top_level.txt'):
        module_name = pkg.get_metadata('top_level.txt').split('\n')[0]
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
