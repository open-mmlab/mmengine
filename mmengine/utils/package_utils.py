# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import subprocess
from importlib.metadata import PackageNotFoundError, distribution


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

    Returns:
        str: The installed path of the package.

    Example:
        >>> get_installed_path('mmcls')
        >>> '.../lib/python3.7/site-packages/mmcls'
    """
    import importlib.util

    # Resolve the location through the import machinery rather than the
    # distribution `location`. `find_spec` correctly handles regular,
    # editable (installs that expose the module via a `.pth`/import hook while
    # the files live outside site-packages), and `PYTHONPATH` installs, whereas
    # distribution metadata reports the site-packages directory even when the
    # module is not physically there.
    module_name = package2module(package)
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise PackageNotFoundError(f'Package {package} is not installed')

    if spec.origin is not None:
        return osp.dirname(spec.origin)
    if spec.submodule_search_locations:
        return spec.submodule_search_locations[0]
    # A namespace package has neither an origin nor a single concrete location.
    raise RuntimeError(
        f'{package} is a namespace package, which is invalid for '
        '`get_installed_path`')


def package2module(package: str) -> str:
    """Infer module name from package.

    Args:
        package (str): Package to infer module name.

    Returns:
        str: The inferred module name.
    """
    import importlib.util

    # The importable module name usually matches the package name. Probing the
    # import machinery first also covers editable installs, whose
    # `top_level.txt` may be absent even though the module is importable.
    if importlib.util.find_spec(package) is not None:
        return package

    # Distribution name differs from the module name (e.g. `mmcv-full` ->
    # `mmcv`); recover the top-level module from distribution metadata.
    dist = distribution(package)
    top_level_text = dist.read_text('top_level.txt')
    if top_level_text is not None:
        lines = [
            line.strip() for line in top_level_text.splitlines()
            if line.strip()
        ]
        if lines:
            return lines[0]
    raise ValueError(f'can not infer the module name of {package}')


def call_command(cmd: list) -> None:
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        raise e  # type: ignore


def install_package(package: str):
    if not is_installed(package):
        call_command(['python', '-m', 'pip', 'install', package])
