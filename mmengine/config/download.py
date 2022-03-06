from pkg_resources import get_distribution
import os.path as osp
from mmcv import load
from mmengine import Config


def get_installed_path(package: str) -> str:
    """Get installed path of package.

    Args:
        package (str): Name of package.

    Example:
        >>> get_installed_path('mmcls')
        >>> '.../lib/python3.7/site-packages/mmcls'
    """
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
    pkg = get_distribution(package)
    if pkg.has_metadata('top_level.txt'):
        module_name = pkg.get_metadata('top_level.txt').split('\n')[0]
        return module_name
    else:
        raise ValueError(f'can not infer the module name of {package}')


def _get_config_meta(package, config_dir):
    package_path = get_installed_path(package)
    config_path = osp.join(package_path, '.mim')
    config_meta = load(osp.join(config_path, config_dir, 'metafile.yml'))
    return config_meta
    # model_config_path = osp.join(config_path, config_meta['Models'])
    # config = Config.fromfile(model_config_path)

if __name__ == '__main__':
    _get_config_meta('mmdet', paa)