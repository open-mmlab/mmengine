from __future__ import annotations
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'packages' / 'mmengine-lite' / 'pyproject.toml'
DIST = ROOT / 'dist'


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(str(src), str(dst))


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def prepare_staging(staging: Path) -> None:
    staging = staging.resolve()
    root = ROOT.resolve()
    if staging == root or root in staging.parents:
        raise ValueError('Staging directory must be outside the repository.')
    if staging.exists():
        raise FileExistsError(f'Staging directory already exists: {staging}')
    staging.mkdir(parents=True)
    copy_tree(ROOT / 'mmengine', staging / 'mmengine')
    files = ('CITATION.cff', 'LICENSE', 'README.md', 'README_zh-CN.md')
    for name in files:
        copy_file(ROOT / name, staging / name)
    copy_file(TEMPLATE, staging / 'pyproject.toml')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdist', action='store_true')
    parser.add_argument('--wheel', action='store_true')
    parser.add_argument('--prepare', type=Path)
    args = parser.parse_args()

    if args.prepare is not None:
        prepare_staging(args.prepare)
        return

    targets = []
    if args.sdist:
        targets.append('--sdist')
    if args.wheel:
        targets.append('--wheel')
    if not targets:
        targets = ['--sdist', '--wheel']

    DIST.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mmengine-lite-') as tmp:
        staging = Path(tmp) / 'mmengine-lite'
        prepare_staging(staging)
        command = [
            'uv',
            'build',
            '--no-sources',
            '--out-dir',
            str(DIST),
            *targets,
            str(staging),
        ]
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
