# Copyright (c) OpenMMLab. All rights reserved.
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import mmengine.fileio as fileio

sys.modules['petrel_client'] = MagicMock()
sys.modules['petrel_client.client'] = MagicMock()

test_data_dir = Path(__file__).parent.parent / 'data'
text_path = test_data_dir / 'filelist.txt'
img_path = test_data_dir / 'color.jpg'
img_url = 'https://raw.githubusercontent.com/mmengine/tests/data/img.png'


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
    # uri should not be None when "backend" does not exist in backend_args
    with pytest.raises(ValueError, match='uri should not be None'):
        fileio.get_file_backend(None, backend_args=None)

    # uri is not None
    backend = fileio.get_file_backend(uri=text_path)
    assert isinstance(backend, fileio.HardDiskBackend)

    uri = 'petrel://your_bucket/img.png'
    backend = fileio.get_file_backend(uri=uri)
    assert isinstance(backend, fileio.PetrelBackend)

    backend = fileio.get_file_backend(uri=img_url)
    assert isinstance(backend, fileio.HTTPBackend)
    uri = 'http://raw.githubusercontent.com/mmengine/tests/data/img.png'
    backend = fileio.get_file_backend(uri=uri)
    assert isinstance(backend, fileio.HTTPBackend)

    # backend_args is not None and it contains a backend name
    backend_args = {'backend': 'disk'}
    backend = fileio.get_file_backend(uri=None, backend_args=backend_args)
    assert isinstance(backend, fileio.HardDiskBackend)

    backend_args = {'backend': 'petrel', 'enable_mc': True}
    backend = fileio.get_file_backend(uri=None, backend_args=backend_args)
    assert isinstance(backend, fileio.PetrelBackend)

    # backend name has a higher priority
    backend_args = {'backend': 'http'}
    backend = fileio.get_file_backend(uri=text_path, backend_args=backend_args)
    assert isinstance(backend, fileio.HTTPBackend)

    # test enable_singleton parameter
    assert len(fileio.io.backend_instances) == 0
    backend1 = fileio.get_file_backend(uri=text_path, enable_singleton=True)
    assert isinstance(backend1, fileio.HardDiskBackend)
    assert len(fileio.io.backend_instances) == 1
    assert fileio.io.backend_instances[':{}'] is backend1

    backend2 = fileio.get_file_backend(uri=text_path, enable_singleton=True)
    assert isinstance(backend2, fileio.HardDiskBackend)
    assert len(fileio.io.backend_instances) == 1
    assert backend2 is backend1

    backend3 = fileio.get_file_backend(uri=text_path, enable_singleton=False)
    assert isinstance(backend3, fileio.HardDiskBackend)
    assert len(fileio.io.backend_instances) == 1
    assert backend3 is not backend2

    backend_args = {'path_mapping': {'src': 'dst'}, 'enable_mc': True}
    uri = 'petrel://your_bucket/img.png'
    backend4 = fileio.get_file_backend(
        uri=uri, backend_args=backend_args, enable_singleton=True)
    assert isinstance(backend4, fileio.PetrelBackend)
    assert len(fileio.io.backend_instances) == 2
    unique_key = 'petrel:{"path_mapping": {"src": "dst"}, "enable_mc": true}'
    assert fileio.io.backend_instances[unique_key] is backend4
    assert backend4 is not backend2

    uri = 'petrel://your_bucket/img1.png'
    backend5 = fileio.get_file_backend(
        uri=uri, backend_args=backend_args, enable_singleton=True)
    assert isinstance(backend5, fileio.PetrelBackend)
    assert len(fileio.io.backend_instances) == 2
    assert backend5 is backend4
    assert backend5 is not backend2

    backend_args = {'path_mapping': {'src1': 'dst1'}, 'enable_mc': True}
    backend6 = fileio.get_file_backend(
        uri=uri, backend_args=backend_args, enable_singleton=True)
    assert isinstance(backend6, fileio.PetrelBackend)
    assert len(fileio.io.backend_instances) == 3
    unique_key = 'petrel:{"path_mapping": {"src1": "dst1"}, "enable_mc": true}'
    assert fileio.io.backend_instances[unique_key] is backend6
    assert backend6 is not backend4
    assert backend6 is not backend5

    backend7 = fileio.get_file_backend(
        uri=uri, backend_args=backend_args, enable_singleton=False)
    assert isinstance(backend7, fileio.PetrelBackend)
    assert len(fileio.io.backend_instances) == 3
    assert backend7 is not backend6


def test_get_bytes():
    pass


def test_get_text():
    pass


def test_put_bytes():
    pass


def test_put_text():
    pass


def test_exists():
    pass


def test_isdir():
    pass


def test_isfile():
    pass


def test_join_path():
    pass


def test_get_local_path():
    pass


def test_copyfile():
    pass


def test_copytree():
    pass


def test_copyfile_from_local():
    pass


def test_copytree_from_local():
    pass


def test_copyfile_to_local():
    pass


def test_copytree_to_local():
    pass


def test_rmfile():
    pass


def test_rmtree():
    pass


def test_copy_if_symlink_fails():
    pass


def test_list_dir_or_file():
    pass


def test_generate_presigned_url():
    pass
