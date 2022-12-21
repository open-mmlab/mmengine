# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from mmengine.fileio.backends import OSSBackend


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
        f_text1 = text1.open('w')
        f_text1.write('text1')
        f_text1.close()

        text2 = Path(tmp_dir) / 'text2.txt'
        f_text2 = text2.open('w')
        f_text2.write('text2')
        f_text2.close()

        dir1 = Path(tmp_dir) / 'dir1'
        dir1.mkdir()

        text3 = dir1 / 'text3.txt'
        f_text3 = text3.open('w')
        f_text3.write('text3')
        f_text3.close()

        dir2 = Path(tmp_dir) / 'dir2'
        dir2.mkdir()

        jpg1 = dir2 / 'img.jpg'
        f_jpg1 = jpg1.open('wb')
        f_jpg1.write(b'img')
        f_jpg1.close()

        dir3 = dir2 / 'dir3'
        dir3.mkdir()

        text4 = dir3 / 'text4.txt'
        f_text4 = text4.open('w')
        f_text4.write('text4')
        f_text4.close()
        yield tmp_dir


sys.modules['oss2'] = MagicMock()
sys.modules['oss2.Auth'] = MagicMock()
sys.modules['oss2.Bucket'] = MagicMock()
sys.modules['oss2.ObjectIterator'] = MagicMock()


class MockAuth:

    def __init__(self, access_key_id, access_key_secret):
        pass


class FileStream():

    def __init__(self, content):
        self.content = content

    def read(self):
        return self.content


class MockBucket:

    def __init__(self, auth, endpoint, bucket_name):
        pass

    def get_object(self, key):
        return FileStream(b'OSS')

    def put_object(self, key, data):
        return 'put'

    def object_exists(self, key):
        not_existed_path = 'img.jpg'
        exist_img_path = 'img.txt'
        if key == not_existed_path:
            return False
        elif key == exist_img_path:
            return True

    def delete_object(self, key):
        pass


class ObjectIteratorResult:

    def __init__(self, name, file_type) -> None:
        self.key = name
        self.is_dir = file_type

    def is_prefix(self):
        return self.is_dir


class MockObjectIterator:

    def __init__(self, bucket, prefix='', delimiter='') -> None:
        self.file_list = []
        with build_temporary_directory() as tmp_dir:
            for entry in os.scandir(os.path.join(tmp_dir, prefix)):
                if not entry.name.startswith('.') and entry.is_file():
                    self.file_list.append(
                        ObjectIteratorResult(entry.name, False))
                elif os.path.isdir(entry.path):
                    self.file_list.append(
                        ObjectIteratorResult(entry.name + '/', True))
        self.len = len(self.file_list)
        self.cur = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.cur += 1
        if self.cur < self.len:
            return self.file_list[self.cur]
        else:
            raise StopIteration()


@patch('oss2.ObjectIterator', MockObjectIterator)
@patch('oss2.Bucket', MockBucket)
@patch('oss2.Auth', MockAuth)
class TestOSSBackend(TestCase):  # type: ignore

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter('ignore', ResourceWarning)
        cls.test_data_dir = Path(__file__).parent.parent.parent / 'data'
        cls.local_img_path = cls.test_data_dir / 'color.jpg'
        cls.local_data_dir = cls.test_data_dir / 'imgs'
        cls.local_img_shape = (300, 400, 3)
        cls.oss_dir = 'oss://oss-cn-hangzhou.aliyuncs.com/mmengine/'

    def setUp(self):
        self.backend = OSSBackend(access_key_id='xxx', access_key_secret='xxx')

    def test_name(self):
        self.assertEqual(self.backend.name, 'OSSBackend')

    def test_map_path(self):
        path = os.path.join(f'{self.oss_dir}', 'img.txt')
        self.assertEqual(
            self.backend._map_path(path),
            'oss://oss-cn-hangzhou.aliyuncs.com/mmengine/img.txt')

    def test_format_path(self):
        path = os.path.join(f'{self.oss_dir}', 'img.jpg')
        self.assertEqual(
            self.backend._format_path(path),
            'oss://oss-cn-hangzhou.aliyuncs.com/mmengine/img.jpg')

    def test_exists(self):
        exist_img_path = os.path.join(f'{self.oss_dir}', 'img.txt')
        self.assertTrue(self.backend.exists(exist_img_path))

        # file and directory does not exist
        not_existed_path = os.path.join(f'{self.oss_dir}', 'img.jpg')
        self.assertFalse(self.backend.exists(not_existed_path))

    def test_get(self):
        img_path = f'{self.oss_dir}/dir2/img.jpg'
        self.assertEqual(self.backend.get(img_path), b'OSS')

    def test_get_text(self):
        text_path = f'{self.oss_dir}/text1.txt'
        self.assertEqual(self.backend.get_text(text_path), 'OSS')

    def test_put(self):
        img_path = f'{self.oss_dir}/img.jpg'
        self.backend.put(b'img', img_path)

    def test_put_text(self):
        text_path = f'{self.oss_dir}/text5.txt'
        self.backend.put_text('text5', text_path)

    def test_list_dir_or_file(self):
        base_dir = f'{self.oss_dir}'
        dir_path = os.path.join(f'{self.oss_dir}', 'dir2')
        self.assertEqual(
            set(self.backend.list_dir_or_file(base_dir)),
            {'dir2/', 'text1.txt', 'text2.txt', 'dir1/'})
        self.assertEqual(
            set(self.backend.list_dir_or_file(dir_path)), {'dir3/', 'img.jpg'})
        self.assertEqual(
            set(self.backend.list_dir_or_file(dir_path, list_dir=False)),
            {'img.jpg'})
        self.assertEqual(
            set(self.backend.list_dir_or_file(dir_path, list_file=False)),
            {'dir3/'})
        self.assertEqual(
            set(
                self.backend.list_dir_or_file(
                    dir_path, list_dir=False, suffix='.jpg')), {'img.jpg'})

    def test_rmtree(self):
        dir_path = os.path.join(f'{self.oss_dir}', 'dir2')
        self.backend.rmtree(dir_path)

    def test_copyfile_from_local(self):

        dst = os.path.join(f'{self.oss_dir}', 'dir2/')
        with patch.object(self.backend, 'put') as patch_copyfile_from_local:
            self.assertEqual(
                self.backend.copyfile_from_local(self.local_img_path, dst),
                os.path.join(dst, 'color.jpg'))
            patch_copyfile_from_local.assert_called_once()

        dst = os.path.join(f'{self.oss_dir}', 'color1.jpg')
        with patch.object(self.backend, 'put') as patch_copyfile_from_local:
            self.assertEqual(
                self.backend.copyfile_from_local(self.local_img_path, dst),
                dst, 'color1.jpg')
            patch_copyfile_from_local.assert_called_once()

    def test_copytree_from_local(self):
        files = []

        def put(file_stream, file_path):
            files.append([file_stream, file_path])

        src = self.local_data_dir
        dst = os.path.join(f'{self.oss_dir}', 'dir2')
        with patch.object(self.backend, 'put', put), patch.object(
                self.backend, 'exists', return_value=False):
            self.assertEqual(self.backend.copytree_from_local(src, dst), dst)
            self.assertEqual(len(files), 2)

            # dst should not exist
            with patch.object(self.backend, 'exists', return_value=True):
                with self.assertRaises(FileExistsError):
                    self.backend.copytree_from_local(src, dst)

    def test_copyfile_to_local(self):
        # dst is a file
        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
                self.backend, 'get', return_value=b'oss'):
            src = os.path.join(f'{self.oss_dir}', 'file.txt')
            dst = Path(tmp_dir) / 'file.txt'
            self.assertEqual(self.backend.copyfile_to_local(src, dst), dst)
            self.assertEqual(open(dst, 'rb').read(), b'oss')

        # dst is a folder
        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
                self.backend, 'get', return_value=b'oss'):
            src = os.path.join(f'{self.oss_dir}', 'file.txt')
            dst = tmp_dir
            self.assertEqual(
                self.backend.copyfile_to_local(src, dst),
                os.path.join(dst, 'file.txt'))
            self.assertEqual(
                open(os.path.join(dst, 'file.txt'), 'rb').read(), b'oss')

    def test_join_path(self):
        filepath = 'oss://endpoint/bucket/dir'
        self.assertEqual(
            self.backend.join_path(filepath, 'another/path'),
            'oss://endpoint/bucket/dir/another/path')
