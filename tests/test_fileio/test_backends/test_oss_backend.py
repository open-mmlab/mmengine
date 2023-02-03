# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from shutil import SameFileError
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


try:
    import oss2

    # raise ImportError("oss2") #offline test
except ImportError:

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
            exist_dir = 'data/'
            copyfile_path = 'img_copy.jpg'
            if key == not_existed_path:
                return False
            elif key == exist_img_path:
                return True
            elif key == exist_dir:
                return True
            elif key == copyfile_path:
                return True

        def delete_object(self, key):
            pass

        def put_symlink(self, target_key, symlink_key):
            return True

        def copy_object(self, source_bucket_name, source_key, target_key):

            class Result:
                status = 200

            return Result()

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
                            ObjectIteratorResult(prefix + entry.name, False))
                    elif os.path.isdir(entry.path):
                        self.file_list.append(
                            ObjectIteratorResult(prefix + entry.name + '/',
                                                 True))
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
            self.backend = OSSBackend(
                access_key_id='xxx', access_key_secret='xxx')

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

        def test_isdir(self):
            exist_directory = os.path.join(f'{self.oss_dir}', 'data')
            with self.assertRaises(NotImplementedError):
                self.backend.isdir(exist_directory)

        def test_isfile(self):
            exist_img_path = os.path.join(f'{self.oss_dir}', 'img.txt')
            self.assertTrue(self.backend.isfile(exist_img_path))

        def test_join_path(self):
            filepath = 'oss://endpoint/bucket/dir'
            self.assertEqual(
                self.backend.join_path(filepath, 'another/path'),
                'oss://endpoint/bucket/dir/another/path')

        def test_get_local_path(self):
            img_path = os.path.join(f'{self.oss_dir}', 'img.txt')
            with self.backend.get_loacl_path(img_path) as f:
                with open(f) as fobj:
                    self.assertEqual(fobj.read(), 'OSS')

        def test_copyfile(self):
            src_path = os.path.join(f'{self.oss_dir}', 'img.jpg')
            dst_path = os.path.join(f'{self.oss_dir}', 'img_copy.jpg')

            dst = self.backend.copyfile(src_path, dst_path)
            self.assertTrue(self.backend.exists(dst))

            with self.assertRaises(SameFileError):
                self.backend.copyfile(src_path, src_path)

        def test_party_copyfile(self):
            src_path = os.path.join(f'{self.oss_dir}', 'bigfile.zip')
            dst_path = os.path.join(f'{self.oss_dir}', 'bigfile_copy.zip')
            with patch.object(
                    self.backend, 'party_copyfile',
                    return_value=dst_path) as party_copyfil_obj:
                dst = self.backend.party_copyfile(src_path, dst_path)
                self.assertEqual(dst, dst_path)
                party_copyfil_obj.assert_called_once()

            with self.assertRaises(SameFileError):
                self.backend.copyfile(src_path, src_path)

        def test_get(self):
            img_path = os.path.join(f'{self.oss_dir}', 'dir2/oss.txt')
            self.assertEqual(self.backend.get(img_path), b'OSS')

        def test_get_text(self):
            text_path = os.path.join(f'{self.oss_dir}', 'text1.txt')
            self.assertEqual(self.backend.get_text(text_path), 'OSS')

        def test_put(self):
            img_path = f'{self.oss_dir}img.jpg'
            self.backend.put(b'img', img_path)

        def test_put_text(self):
            text_path = f'{self.oss_dir}text5.txt'
            self.backend.put_text('text5', text_path)

        def test_list_dir_or_file(self):
            base_dir = f'{self.oss_dir}'
            dir_path = f'{self.oss_dir}dir2/'
            self.assertEqual(
                set(self.backend.list_dir_or_file(base_dir)),
                {'dir2/', 'text1.txt', 'text2.txt', 'dir1/'})

            self.assertEqual(
                set(self.backend.list_dir_or_file(dir_path)),
                {'dir3/', 'img.jpg'})

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

            # test recursive
            self.assertEqual(
                set(self.backend.list_dir_or_file(dir_path, recursive=True)),
                {'img.jpg', 'dir3/', 'dir3/text4.txt'})

            with self.assertRaises(TypeError):
                self.backend.list_dir_or_file(
                    dir_path, list_file=False, suffix='.txt')

            with self.assertRaises(TypeError):
                dir_path = f'{self.oss_dir}dir2'
                self.backend.list_dir_or_file(dir_path, recursive=True)

        def test_copyfile_from_local(self):

            dst = os.path.join(f'{self.oss_dir}', 'dir2/')
            with patch.object(self.backend,
                              'put') as patch_copyfile_from_local:
                self.assertEqual(
                    self.backend.copyfile_from_local(self.local_img_path, dst),
                    os.path.join(dst, 'color.jpg'))
                patch_copyfile_from_local.assert_called_once()

            dst = os.path.join(f'{self.oss_dir}', 'color1.jpg')
            with patch.object(self.backend,
                              'put') as patch_copyfile_from_local:
                self.assertEqual(
                    self.backend.copyfile_from_local(self.local_img_path, dst),
                    dst, 'color1.jpg')
                patch_copyfile_from_local.assert_called_once()

        def test_copytree_from_local(self):
            files = []

            def put(file_stream, file_path):
                files.append([file_stream, file_path])

            src = self.local_data_dir
            dst = os.path.join(f'{self.oss_dir}', 'dir2/')
            with patch.object(self.backend, 'put', put), patch.object(
                    self.backend, 'exists', return_value=False):
                self.assertEqual(
                    self.backend.copytree_from_local(src, dst), dst)
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

        def test_copytree_to_local(self):
            src = f'{self.oss_dir}'
            dst = self.test_data_dir / 'mmengine'
            with patch.object(
                    self.backend, 'copytree_to_local', return_value=dst):
                self.assertEqual(self.backend.copytree_to_local(src, dst), dst)

        def test_remove(self):
            dir_path = os.path.join(f'{self.oss_dir}', 'img.txt')
            with patch.object(self.backend, 'remove') as remove_obj:
                self.backend.remove(dir_path)
                remove_obj.assert_called_once()

        def test_symlink(self):
            file_path = os.path.join(f'{self.oss_dir}', 'text5.txt')
            linked_file_path = os.path.join(f'{self.oss_dir}',
                                            'linked_text5.txt')
            with self.assertRaises(FileNotFoundError):
                self.backend.symlink(file_path, linked_file_path)

else:
    import pandas as pd

    class TestOSSBackend(TestCase):  # type: ignore

        @classmethod
        def setUpClass(cls):
            warnings.simplefilter('ignore', ResourceWarning)
            cls.test_data_dir = Path(__file__).parent.parent.parent / 'data'
            cls.local_img_path = cls.test_data_dir / 'color.jpg'
            cls.local_data_dir = cls.test_data_dir / 'imgs'

            cls.oss_dir = 'oss://oss-cn-hangzhou.aliyuncs.com/mmengine/'
            local_oss_ak_path = cls.test_data_dir / 'AccessKey.csv'
            access_key = pd.read_csv(local_oss_ak_path, sep=',')
            cls.access_key_id = access_key.loc[0, 'AccessKey ID']
            cls.access_key_secret = access_key.loc[0, 'AccessKey Secret']
            print(f'access_key_id: {cls.access_key_id},\
                access_key_secret: {cls.access_key_secret}')

        def setUp(self):
            self.backend = OSSBackend(
                access_key_id=self.access_key_id,
                access_key_secret=self.access_key_secret)

        @classmethod
        def tearDownClass(cls):
            print(f'{cls.__class__.__name__} test over')

        def tearDown(self):
            pass

        def test_name(self):
            self.assertEqual(self.backend.name, 'OSSBackend')

        def test_map_path(self):
            path = os.path.join(f'{self.oss_dir}', 'img.txt')
            self.assertEqual(
                self.backend._map_path(path),
                'oss://oss-cn-hangzhou.aliyuncs.com/mmengine/img.txt')

            backend = OSSBackend(
                access_key_id=self.access_key_id,
                access_key_secret=self.access_key_secret,
                path_mapping={'mmengine': 'oss_backend'})
            self.assertEqual(
                backend._map_path(path),
                'oss://oss-cn-hangzhou.aliyuncs.com/oss_backend/img.txt')

        def test_format_path(self):
            path = os.path.join(f'{self.oss_dir}', 'img.jpg')
            self.assertEqual(
                self.backend._format_path(path),
                'oss://oss-cn-hangzhou.aliyuncs.com/mmengine/img.jpg')

            self.assertEqual(
                self.backend._format_path(
                    'oss://oss-cn-hangzhou.aliyuncs.com/mmengine\\img.jpg'),
                'oss://oss-cn-hangzhou.aliyuncs.com/mmengine/img.jpg')

        def test_get(self):
            img_path = os.path.join(f'{self.oss_dir}', 'dir2/oss.txt')
            self.assertEqual(self.backend.get(img_path), b'OSS\n')

            img_path = os.path.join(f'{self.oss_dir}', 'dir2/oss1.txt')
            with self.assertRaises(oss2.exceptions.NoSuchKey):
                self.assertEqual(self.backend.get(img_path), b'OSS')

        def test_get_text(self):
            text_path = os.path.join(f'{self.oss_dir}', 'oss.txt')
            self.assertEqual(self.backend.get_text(text_path), 'OSS\n')

        def test_put(self):
            img_path = f'{self.oss_dir}img.jpg'
            self.backend.put(b'img', img_path)

        def test_put_text(self):
            text_path = f'{self.oss_dir}text5.txt'
            self.backend.put_text('text5', text_path)

        def test_exists(self):
            exist_img_path = os.path.join(f'{self.oss_dir}', 'oss.txt')
            self.assertTrue(self.backend.exists(exist_img_path))

            # file and directory does not exist
            not_existed_path = os.path.join(f'{self.oss_dir}', 'img1.jpg')
            self.assertFalse(self.backend.exists(not_existed_path))

        def test_isdir(self):
            exist_directory = os.path.join(f'{self.oss_dir}', 'data')
            with self.assertRaises(NotImplementedError):
                self.backend.isdir(exist_directory)

        def test_isfile(self):
            exist_img_path = os.path.join(f'{self.oss_dir}', 'img.jpg')
            self.assertTrue(self.backend.isfile(exist_img_path))

        def test_join_path(self):
            filepath = 'oss://endpoint/bucket/dir'
            self.assertEqual(
                self.backend.join_path(filepath, 'another/path'),
                'oss://endpoint/bucket/dir/another/path')

        def test_get_local_path(self):
            img_path = os.path.join(f'{self.oss_dir}', 'img.jpg')
            with self.backend.get_loacl_path(img_path) as f:
                with open(f) as fobj:
                    self.assertEqual(fobj.read(), 'img')

        def test_copyfile(self):
            src_path = os.path.join(f'{self.oss_dir}', 'img.jpg')
            dst_path = os.path.join(f'{self.oss_dir}', 'img_copy.jpg')
            dst = self.backend.copyfile(src_path, dst_path)
            self.assertTrue(self.backend.exists(dst))

            with self.assertRaises(SameFileError):
                self.backend.copyfile(src_path, src_path)

        def test_party_copyfile(self):
            src_path = os.path.join(f'{self.oss_dir}', 'bigfile.zip')
            dst_path = os.path.join(f'{self.oss_dir}', 'bigfile_copy.zip')
            dst = self.backend.party_copyfile(src_path, dst_path)
            self.assertTrue(self.backend.exists(dst))

            with self.assertRaises(SameFileError):
                self.backend.copyfile(src_path, src_path)

        def test_list_dir_or_file(self):
            dir_path = os.path.join(f'{self.oss_dir}', 'dir2/')
            self.assertEqual(
                set(self.backend.list_dir_or_file(dir_path)), {
                    'dir3/', 'oss.txt', 'oss.jpg', 'color.jpg', 'gray.jpg',
                    'test_img.jpg'
                })
            self.assertEqual(
                set(self.backend.list_dir_or_file(dir_path, list_dir=False)), {
                    'oss.txt', 'oss.jpg', 'color.jpg', 'gray.jpg',
                    'test_img.jpg'
                })
            self.assertEqual(
                set(self.backend.list_dir_or_file(dir_path, list_file=False)),
                {'dir3/'})
            self.assertEqual(
                set(
                    self.backend.list_dir_or_file(
                        dir_path, list_dir=False, suffix='.jpg')),
                {'oss.jpg', 'color.jpg', 'gray.jpg', 'test_img.jpg'})

            # test recursive
            dir_path = f'{self.oss_dir}data/dir2/'
            self.assertEqual(
                set(self.backend.list_dir_or_file(dir_path, recursive=True)),
                {'img.jpg', 'dir3/', 'dir3/text4.txt'})

            with self.assertRaises(TypeError):
                dir_path = f'{self.oss_dir}data/dir2/'
                # dir_path = f'{self.oss_dir}'

                self.backend.list_dir_or_file(
                    dir_path, list_file=False, suffix='.txt')

            with self.assertRaises(TypeError):
                dir_path = f'{self.oss_dir}data/dir2'
                # dir_path = f'{self.oss_dir}'

                self.backend.list_dir_or_file(dir_path, recursive=True)

        def test_remove(self):
            dir_path = os.path.join(f'{self.oss_dir}', 'remove.txt')
            self.backend.put(b'remove', dir_path)
            self.assertTrue(self.backend.exists(dir_path))
            self.backend.remove(dir_path)
            self.assertFalse(self.backend.exists(dir_path))

        def test_copyfile_from_local(self):

            # copy to dir
            dst = os.path.join(f'{self.oss_dir}', 'dir2/')

            self.assertEqual(
                self.backend.copyfile_from_local(self.local_img_path, dst),
                os.path.join(dst, 'color.jpg'))

            # copy to target file
            dst = os.path.join(f'{self.oss_dir}', 'color.jpg')
            with patch.object(self.backend,
                              'put') as patch_copyfile_from_local:
                self.assertEqual(
                    self.backend.copyfile_from_local(self.local_img_path, dst),
                    dst, 'color.jpg')
                patch_copyfile_from_local.assert_called_once()

        def test_copytree_from_local(self):

            src = self.local_data_dir / ''
            dst = os.path.join(f'{self.oss_dir}', 'imgs/')
            self.assertEqual(self.backend.copytree_from_local(src, dst), dst)

        def test_copyfile_to_local(self):
            # dst is a file

            src = os.path.join(f'{self.oss_dir}', 'oss.txt')
            dst = Path(self.local_data_dir) / 'oss.txt'
            self.assertEqual(self.backend.copyfile_to_local(src, dst), dst)
            self.assertEqual(open(dst, 'rb').read(), b'OSS\n')
            os.remove(dst)

            # dst is a folder
            src = os.path.join(f'{self.oss_dir}', 'oss.txt')
            dst = str(self.local_data_dir)
            self.assertEqual(
                self.backend.copyfile_to_local(src, dst),
                os.path.join(dst, 'oss.txt'))
            self.assertEqual(
                open(os.path.join(dst, 'oss.txt'), 'rb').read(), b'OSS\n')
            os.remove(os.path.join(dst, 'oss.txt'))

        def test_copytree_to_local(self):
            src = f'{self.oss_dir}'
            dst = self.test_data_dir / 'mmengine'
            self.assertEqual(self.backend.copytree_to_local(src, dst), dst)
            shutil.rmtree(dst)

        def test_symlink(self):
            file_path = os.path.join(f'{self.oss_dir}', 'text5.txt')
            linked_file_path = os.path.join(f'{self.oss_dir}',
                                            'linked_text5.txt')
            self.backend.symlink(file_path, linked_file_path)
            self.assertTrue(self.backend.exists(linked_file_path))
