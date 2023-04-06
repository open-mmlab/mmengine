# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import platform
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from shutil import SameFileError
from unittest.mock import MagicMock, patch

import pytest

import mmengine.fileio as fileio

sys.modules['petrel_client'] = MagicMock()
sys.modules['petrel_client.client'] = MagicMock()

test_data_dir = Path(__file__).parent.parent / 'data'
text_path = test_data_dir / 'filelist.txt'
img_path = test_data_dir / 'color.jpg'
img_url = 'https://raw.githubusercontent.com/mmengine/tests/data/img.png'


@contextmanager
def build_temporary_directory():
    """Build a temporary directory containing many files to test
    ``FileClient.list_dir_or_file``.

    . \n
    | -- dir1 \n
    | -- | -- text3.txt \n
    | -- dir2 \n
    | -- | -- dir3 \n
    | -- | -- | -- text4.txt \n
    | -- | -- img.jpg \n
    | -- text1.txt \n
    | -- text2.txt \n
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        text1 = Path(tmp_dir) / 'text1.txt'
        text1.open('w').write('text1')
        text2 = Path(tmp_dir) / 'text2.txt'
        text2.open('w').write('text2')
        dir1 = Path(tmp_dir) / 'dir1'
        dir1.mkdir()
        text3 = dir1 / 'text3.txt'
        text3.open('w').write('text3')
        dir2 = Path(tmp_dir) / 'dir2'
        dir2.mkdir()
        jpg1 = dir2 / 'img.jpg'
        jpg1.open('wb').write(b'img')
        dir3 = dir2 / 'dir3'
        dir3.mkdir()
        text4 = dir3 / 'text4.txt'
        text4.open('w').write('text4')
        yield tmp_dir


def test_parse_uri_prefix():
    # input path is None
    with pytest.raises(AssertionError):
        fileio.io._parse_uri_prefix(None)

    # input path is list
    with pytest.raises(AssertionError):
        fileio.io._parse_uri_prefix([])

    # input path is Path object
    assert fileio.io._parse_uri_prefix(uri=text_path) == ''

    # input path starts with https
    assert fileio.io._parse_uri_prefix(uri=img_url) == 'https'

    # input path starts with s3
    uri = 's3://your_bucket/img.png'
    assert fileio.io._parse_uri_prefix(uri) == 's3'

    # input path starts with clusterName:s3
    uri = 'clusterName:s3://your_bucket/img.png'
    assert fileio.io._parse_uri_prefix(uri) == 's3'


def test_get_file_backend():
    # other unit tests may have added instances so clear them here.
    fileio.io.backend_instances = {}

    # uri should not be None when "backend" does not exist in backend_args
    with pytest.raises(ValueError, match='uri should not be None'):
        fileio.get_file_backend(None, backend_args=None)

    # uri is not None
    backend = fileio.get_file_backend(uri=text_path)
    assert isinstance(backend, fileio.backends.LocalBackend)

    uri = 'petrel://your_bucket/img.png'
    backend = fileio.get_file_backend(uri=uri)
    assert isinstance(backend, fileio.backends.PetrelBackend)

    backend = fileio.get_file_backend(uri=img_url)
    assert isinstance(backend, fileio.backends.HTTPBackend)
    uri = 'http://raw.githubusercontent.com/mmengine/tests/data/img.png'
    backend = fileio.get_file_backend(uri=uri)
    assert isinstance(backend, fileio.backends.HTTPBackend)

    # backend_args is not None and it contains a backend name
    backend_args = {'backend': 'local'}
    backend = fileio.get_file_backend(uri=None, backend_args=backend_args)
    assert isinstance(backend, fileio.backends.LocalBackend)
    # backend_args should not be modified
    assert backend_args == {'backend': 'local'}

    backend_args = {'backend': 'petrel', 'enable_mc': True}
    backend = fileio.get_file_backend(uri=None, backend_args=backend_args)
    assert isinstance(backend, fileio.backends.PetrelBackend)
    assert backend_args == {'backend': 'petrel', 'enable_mc': True}

    # backend name has a higher priority
    backend_args = {'backend': 'http'}
    backend = fileio.get_file_backend(uri=text_path, backend_args=backend_args)
    assert isinstance(backend, fileio.backends.HTTPBackend)

    # test enable_singleton parameter
    assert len(fileio.io.backend_instances) == 0
    backend1 = fileio.get_file_backend(uri=text_path, enable_singleton=True)
    assert isinstance(backend1, fileio.backends.LocalBackend)
    assert len(fileio.io.backend_instances) == 1
    assert fileio.io.backend_instances[':{}'] is backend1

    backend2 = fileio.get_file_backend(uri=text_path, enable_singleton=True)
    assert isinstance(backend2, fileio.backends.LocalBackend)
    assert len(fileio.io.backend_instances) == 1
    assert backend2 is backend1

    backend3 = fileio.get_file_backend(uri=text_path, enable_singleton=False)
    assert isinstance(backend3, fileio.backends.LocalBackend)
    assert len(fileio.io.backend_instances) == 1
    assert backend3 is not backend2

    backend_args = {'path_mapping': {'src': 'dst'}, 'enable_mc': True}
    uri = 'petrel://your_bucket/img.png'
    backend4 = fileio.get_file_backend(
        uri=uri, backend_args=backend_args, enable_singleton=True)
    assert isinstance(backend4, fileio.backends.PetrelBackend)
    assert len(fileio.io.backend_instances) == 2
    unique_key = 'petrel:{"path_mapping": {"src": "dst"}, "enable_mc": true}'
    assert fileio.io.backend_instances[unique_key] is backend4
    assert backend4 is not backend2

    uri = 'petrel://your_bucket/img1.png'
    backend5 = fileio.get_file_backend(
        uri=uri, backend_args=backend_args, enable_singleton=True)
    assert isinstance(backend5, fileio.backends.PetrelBackend)
    assert len(fileio.io.backend_instances) == 2
    assert backend5 is backend4
    assert backend5 is not backend2

    backend_args = {'path_mapping': {'src1': 'dst1'}, 'enable_mc': True}
    backend6 = fileio.get_file_backend(
        uri=uri, backend_args=backend_args, enable_singleton=True)
    assert isinstance(backend6, fileio.backends.PetrelBackend)
    assert len(fileio.io.backend_instances) == 3
    unique_key = 'petrel:{"path_mapping": {"src1": "dst1"}, "enable_mc": true}'
    assert fileio.io.backend_instances[unique_key] is backend6
    assert backend6 is not backend4
    assert backend6 is not backend5

    backend7 = fileio.get_file_backend(
        uri=uri, backend_args=backend_args, enable_singleton=False)
    assert isinstance(backend7, fileio.backends.PetrelBackend)
    assert len(fileio.io.backend_instances) == 3
    assert backend7 is not backend6


def test_get():
    # test LocalBackend
    filepath = Path(img_path)
    img_bytes = fileio.get(filepath)
    assert filepath.open('rb').read() == img_bytes


def test_get_text():
    # test LocalBackend
    filepath = Path(text_path)
    text = fileio.get_text(filepath)
    assert filepath.open('r').read() == text


def test_put():
    # test LocalBackend
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / 'img.png'
        fileio.put(b'disk', filepath)
        assert fileio.get(filepath) == b'disk'

        # If the directory does not exist, put will create a
        # directory first
        filepath = Path(tmp_dir) / 'not_existed_dir' / 'test.jpg'
        fileio.put(b'disk', filepath)
        assert fileio.get(filepath) == b'disk'


def test_put_text():
    # test LocalBackend
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / 'text.txt'
        fileio.put_text('text', filepath)
        assert fileio.get_text(filepath) == 'text'

        # If the directory does not exist, put_text will create a
        # directory first
        filepath = Path(tmp_dir) / 'not_existed_dir' / 'test.txt'
        fileio.put_text('disk', filepath)
        assert fileio.get_text(filepath) == 'disk'


def test_exists():
    # test LocalBackend
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert fileio.exists(tmp_dir)
        filepath = Path(tmp_dir) / 'test.txt'
        assert not fileio.exists(filepath)
        fileio.put_text('disk', filepath)
        assert fileio.exists(filepath)


def test_isdir():
    # test LocalBackend
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert fileio.isdir(tmp_dir)
        filepath = Path(tmp_dir) / 'test.txt'
        fileio.put_text('disk', filepath)
        assert not fileio.isdir(filepath)


def test_isfile():
    # test LocalBackend
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert not fileio.isfile(tmp_dir)
        filepath = Path(tmp_dir) / 'test.txt'
        fileio.put_text('disk', filepath)
        assert fileio.isfile(filepath)


def test_join_path():
    # test LocalBackend
    filepath = fileio.join_path(test_data_dir, 'file')
    expected = osp.join(test_data_dir, 'file')
    assert filepath == expected

    filepath = fileio.join_path(test_data_dir, 'dir', 'file')
    expected = osp.join(test_data_dir, 'dir', 'file')
    assert filepath == expected


def test_get_local_path():
    # test LocalBackend
    with fileio.get_local_path(text_path) as filepath:
        assert str(text_path) == filepath


def test_copyfile():
    # test LocalBackend
    with tempfile.TemporaryDirectory() as tmp_dir:
        src = Path(tmp_dir) / 'test.txt'
        fileio.put_text('disk', src)
        dst = Path(tmp_dir) / 'test.txt.bak'
        assert fileio.copyfile(src, dst) == dst
        assert fileio.get_text(dst) == 'disk'

        # dst is a directory
        dst = Path(tmp_dir) / 'dir'
        dst.mkdir()
        assert fileio.copyfile(src, dst) == fileio.join_path(dst, 'test.txt')
        assert fileio.get_text(fileio.join_path(dst, 'test.txt')) == 'disk'

        # src and src should not be same file
        with pytest.raises(SameFileError):
            fileio.copyfile(src, src)


def test_copytree():
    # test LocalBackend
    with build_temporary_directory() as tmp_dir:
        # src and dst are Path objects
        src = Path(tmp_dir) / 'dir1'
        dst = Path(tmp_dir) / 'dir100'
        assert fileio.copytree(src, dst) == dst
        assert fileio.isdir(dst)
        assert fileio.isfile(dst / 'text3.txt')
        assert fileio.get_text(dst / 'text3.txt') == 'text3'

        # dst should not exist
        with pytest.raises(FileExistsError):
            fileio.copytree(src, Path(tmp_dir) / 'dir2')


def test_copyfile_from_local():
    # test LocalBackend
    with tempfile.TemporaryDirectory() as tmp_dir:
        src = Path(tmp_dir) / 'test.txt'
        fileio.put_text('disk', src)
        dst = Path(tmp_dir) / 'test.txt.bak'
        assert fileio.copyfile(src, dst) == dst
        assert fileio.get_text(dst) == 'disk'

        dst = Path(tmp_dir) / 'dir'
        dst.mkdir()
        assert fileio.copyfile(src, dst) == fileio.join_path(dst, 'test.txt')
        assert fileio.get_text(fileio.join_path(dst, 'test.txt')) == 'disk'

        # src and src should not be same file
        with pytest.raises(SameFileError):
            fileio.copyfile(src, src)


def test_copytree_from_local():
    # test LocalBackend
    with build_temporary_directory() as tmp_dir:
        # src and dst are Path objects
        src = Path(tmp_dir) / 'dir1'
        dst = Path(tmp_dir) / 'dir100'
        assert fileio.copytree(src, dst) == dst
        assert fileio.isdir(dst)
        assert fileio.isfile(dst / 'text3.txt')
        assert fileio.get_text(dst / 'text3.txt') == 'text3'

        # dst should not exist
        with pytest.raises(FileExistsError):
            fileio.copytree(src, Path(tmp_dir) / 'dir2')


def test_copyfile_to_local():
    # test LocalBackend
    with tempfile.TemporaryDirectory() as tmp_dir:
        src = Path(tmp_dir) / 'test.txt'
        fileio.put_text('disk', src)
        dst = Path(tmp_dir) / 'test.txt.bak'
        assert fileio.copyfile(src, dst) == dst
        assert fileio.get_text(dst) == 'disk'

        dst = Path(tmp_dir) / 'dir'
        dst.mkdir()
        assert fileio.copyfile(src, dst) == fileio.join_path(dst, 'test.txt')
        assert fileio.get_text(fileio.join_path(dst, 'test.txt')) == 'disk'

        # src and src should not be same file
        with pytest.raises(SameFileError):
            fileio.copyfile(src, src)


def test_copytree_to_local():
    # test LocalBackend
    with build_temporary_directory() as tmp_dir:
        # src and dst are Path objects
        src = Path(tmp_dir) / 'dir1'
        dst = Path(tmp_dir) / 'dir100'
        assert fileio.copytree(src, dst) == dst
        assert fileio.isdir(dst)
        assert fileio.isfile(dst / 'text3.txt')
        assert fileio.get_text(dst / 'text3.txt') == 'text3'

        # dst should not exist
        with pytest.raises(FileExistsError):
            fileio.copytree(src, Path(tmp_dir) / 'dir2')


def test_remove():
    # test LocalBackend
    with tempfile.TemporaryDirectory() as tmp_dir:
        # filepath is a Path object
        filepath = Path(tmp_dir) / 'test.txt'
        fileio.put_text('disk', filepath)
        assert fileio.exists(filepath)
        fileio.remove(filepath)
        assert not fileio.exists(filepath)

        # raise error if file does not exist
        with pytest.raises(FileNotFoundError):
            filepath = Path(tmp_dir) / 'test1.txt'
            fileio.remove(filepath)

        # can not remove directory
        filepath = Path(tmp_dir) / 'dir'
        filepath.mkdir()
        with pytest.raises(IsADirectoryError):
            fileio.remove(filepath)


def test_rmtree():
    # test LocalBackend
    with build_temporary_directory() as tmp_dir:
        # src and dst are Path objects
        dir_path = Path(tmp_dir) / 'dir1'
        assert fileio.exists(dir_path)
        fileio.rmtree(dir_path)
        assert not fileio.exists(dir_path)

        dir_path = Path(tmp_dir) / 'dir2'
        assert fileio.exists(dir_path)
        fileio.rmtree(dir_path)
        assert not fileio.exists(dir_path)


def test_copy_if_symlink_fails():
    # test LocalBackend
    with tempfile.TemporaryDirectory() as tmp_dir:
        # create a symlink for a file
        src = Path(tmp_dir) / 'test.txt'
        fileio.put_text('disk', src)
        dst = Path(tmp_dir) / 'test_link.txt'
        res = fileio.copy_if_symlink_fails(src, dst)
        if platform.system() == 'Linux':
            assert res
            assert osp.islink(dst)
        assert fileio.get_text(dst) == 'disk'

        # create a symlink for a directory
        src = Path(tmp_dir) / 'dir'
        src.mkdir()
        dst = Path(tmp_dir) / 'dir_link'
        res = fileio.copy_if_symlink_fails(src, dst)
        if platform.system() == 'Linux':
            assert res
            assert osp.islink(dst)
        assert fileio.exists(dst)

        def symlink(src, dst):
            raise Exception

        # copy files if symblink fails
        with patch.object(os, 'symlink', side_effect=symlink):
            src = Path(tmp_dir) / 'test.txt'
            dst = Path(tmp_dir) / 'test_link1.txt'
            res = fileio.copy_if_symlink_fails(src, dst)
            assert not res
            assert not osp.islink(dst)
            assert fileio.exists(dst)

        # copy directory if symblink fails
        with patch.object(os, 'symlink', side_effect=symlink):
            src = Path(tmp_dir) / 'dir'
            dst = Path(tmp_dir) / 'dir_link1'
            res = fileio.copy_if_symlink_fails(src, dst)
            assert not res
            assert not osp.islink(dst)
            assert fileio.exists(dst)


def test_list_dir_or_file():
    # test LocalBackend
    with build_temporary_directory() as tmp_dir:
        # list directories and files
        assert set(fileio.list_dir_or_file(tmp_dir)) == {
            'dir1', 'dir2', 'text1.txt', 'text2.txt'
        }

        # list directories and files recursively
        assert set(fileio.list_dir_or_file(tmp_dir, recursive=True)) == {
            'dir1',
            osp.join('dir1', 'text3.txt'), 'dir2',
            osp.join('dir2', 'dir3'),
            osp.join('dir2', 'dir3', 'text4.txt'),
            osp.join('dir2', 'img.jpg'), 'text1.txt', 'text2.txt'
        }

        # only list directories
        assert set(fileio.list_dir_or_file(
            tmp_dir, list_file=False)) == {'dir1', 'dir2'}

        with pytest.raises(
                TypeError,
                match='`suffix` should be None when `list_dir` is True'):
            list(
                fileio.list_dir_or_file(
                    tmp_dir, list_file=False, suffix='.txt'))

        # only list directories recursively
        assert set(
            fileio.list_dir_or_file(
                tmp_dir, list_file=False,
                recursive=True)) == {'dir1', 'dir2',
                                     osp.join('dir2', 'dir3')}

        # only list files
        assert set(fileio.list_dir_or_file(
            tmp_dir, list_dir=False)) == {'text1.txt', 'text2.txt'}

        # only list files recursively
        assert set(
            fileio.list_dir_or_file(tmp_dir, list_dir=False,
                                    recursive=True)) == {
                                        osp.join('dir1', 'text3.txt'),
                                        osp.join('dir2', 'dir3', 'text4.txt'),
                                        osp.join('dir2', 'img.jpg'),
                                        'text1.txt', 'text2.txt'
                                    }

        # only list files ending with suffix
        assert set(
            fileio.list_dir_or_file(
                tmp_dir, list_dir=False,
                suffix='.txt')) == {'text1.txt', 'text2.txt'}
        assert set(
            fileio.list_dir_or_file(
                tmp_dir, list_dir=False,
                suffix=('.txt', '.jpg'))) == {'text1.txt', 'text2.txt'}

        with pytest.raises(
                TypeError,
                match='`suffix` must be a string or tuple of strings'):
            list(
                fileio.list_dir_or_file(
                    tmp_dir, list_dir=False, suffix=['.txt', '.jpg']))

        # only list files ending with suffix recursively
        assert set(
            fileio.list_dir_or_file(
                tmp_dir, list_dir=False, suffix='.txt', recursive=True)) == {
                    osp.join('dir1', 'text3.txt'),
                    osp.join('dir2', 'dir3', 'text4.txt'), 'text1.txt',
                    'text2.txt'
                }

        # only list files ending with suffix
        assert set(
            fileio.list_dir_or_file(
                tmp_dir,
                list_dir=False,
                suffix=('.txt', '.jpg'),
                recursive=True)) == {
                    osp.join('dir1', 'text3.txt'),
                    osp.join('dir2', 'dir3', 'text4.txt'),
                    osp.join('dir2', 'img.jpg'), 'text1.txt', 'text2.txt'
                }
