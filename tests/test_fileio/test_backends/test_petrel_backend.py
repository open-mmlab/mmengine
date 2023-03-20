# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import sys
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from shutil import SameFileError
from unittest import TestCase
from unittest.mock import MagicMock, patch

from mmengine.fileio.backends import PetrelBackend
from mmengine.utils import has_method


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


try:
    # Other unit tests may mock these modules so we need to pop them first.
    sys.modules.pop('petrel_client', None)
    sys.modules.pop('petrel_client.client', None)

    # If petrel_client is imported successfully, we can test PetrelBackend
    # without mock.
    import petrel_client  # noqa: F401
except ImportError:
    sys.modules['petrel_client'] = MagicMock()
    sys.modules['petrel_client.client'] = MagicMock()

    class MockPetrelClient:

        def __init__(self,
                     enable_mc=True,
                     enable_multi_cluster=False,
                     conf_path=None):
            self.enable_mc = enable_mc
            self.enable_multi_cluster = enable_multi_cluster
            self.conf_path = conf_path

        def Get(self, filepath):
            with open(filepath, 'rb') as f:
                content = f.read()
            return content

        def put(self):
            pass

        def delete(self):
            pass

        def contains(self):
            pass

        def isdir(self):
            pass

        def list(self, dir_path):
            for entry in os.scandir(dir_path):
                if not entry.name.startswith('.') and entry.is_file():
                    yield entry.name
                elif osp.isdir(entry.path):
                    yield entry.name + '/'

    @contextmanager
    def delete_and_reset_method(obj, method):
        method_obj = deepcopy(getattr(type(obj), method))
        try:
            delattr(type(obj), method)
            yield
        finally:
            setattr(type(obj), method, method_obj)

    @patch('petrel_client.client.Client', MockPetrelClient)
    class TestPetrelBackend(TestCase):

        @classmethod
        def setUpClass(cls):
            cls.test_data_dir = Path(__file__).parent.parent.parent / 'data'
            cls.img_path = cls.test_data_dir / 'color.jpg'
            cls.img_shape = (300, 400, 3)
            cls.text_path = cls.test_data_dir / 'filelist.txt'
            cls.petrel_dir = 'petrel://user/data'
            cls.petrel_path = f'{cls.petrel_dir}/test.jpg'
            cls.expected_dir = 's3://user/data'
            cls.expected_path = f'{cls.expected_dir}/test.jpg'

        def test_name(self):
            backend = PetrelBackend()
            self.assertEqual(backend.name, 'PetrelBackend')

        def test_map_path(self):
            backend = PetrelBackend(path_mapping=None)
            self.assertEqual(
                backend._map_path(self.petrel_path), self.petrel_path)

            backend = PetrelBackend(
                path_mapping={'data/': 'petrel://user/data/'})
            self.assertEqual(
                backend._map_path('data/test.jpg'), self.petrel_path)

        def test_format_path(self):
            backend = PetrelBackend()
            formatted_filepath = backend._format_path(
                'petrel://user\\data\\test.jpg')
            self.assertEqual(formatted_filepath, self.petrel_path)

        def test_replace_prefix(self):
            backend = PetrelBackend()
            self.assertEqual(
                backend._replace_prefix(self.petrel_path), self.expected_path)

        def test_join_path(self):
            backend = PetrelBackend()
            self.assertEqual(
                backend.join_path(self.petrel_dir, 'file'),
                f'{self.petrel_dir}/file')
            self.assertEqual(
                backend.join_path(f'{self.petrel_dir}/', 'file'),
                f'{self.petrel_dir}/file')
            self.assertEqual(
                backend.join_path(f'{self.petrel_dir}/', '/file'),
                f'{self.petrel_dir}/file')
            self.assertEqual(
                backend.join_path(self.petrel_dir, 'dir', 'file'),
                f'{self.petrel_dir}/dir/file')

        def test_get(self):
            backend = PetrelBackend()
            with patch.object(
                    backend._client, 'Get',
                    return_value=b'petrel') as patched_get:
                self.assertEqual(backend.get(self.petrel_path), b'petrel')
                patched_get.assert_called_once_with(self.expected_path)

        def test_get_text(self):
            backend = PetrelBackend()
            with patch.object(
                    backend._client, 'Get',
                    return_value=b'petrel') as patched_get:
                self.assertEqual(backend.get_text(self.petrel_path), 'petrel')
                patched_get.assert_called_once_with(self.expected_path)

        def test_put(self):
            backend = PetrelBackend()
            with patch.object(backend._client, 'put') as patched_put:
                backend.put(b'petrel', self.petrel_path)
                patched_put.assert_called_once_with(self.expected_path,
                                                    b'petrel')

        def test_put_text(self):
            backend = PetrelBackend()
            with patch.object(backend._client, 'put') as patched_put:
                backend.put_text('petrel', self.petrel_path)
                patched_put.assert_called_once_with(self.expected_path,
                                                    b'petrel')

        def test_exists(self):
            backend = PetrelBackend()
            self.assertTrue(has_method(backend._client, 'contains'))
            self.assertTrue(has_method(backend._client, 'isdir'))
            # raise Exception if `_client.contains` and '_client.isdir' are not
            # implemented
            with delete_and_reset_method(backend._client, 'contains'), \
                 delete_and_reset_method(backend._client, 'isdir'):
                self.assertFalse(has_method(backend._client, 'contains'))
                self.assertFalse(has_method(backend._client, 'isdir'))
                with self.assertRaises(NotImplementedError):
                    backend.exists(self.petrel_path)

            with patch.object(
                    backend._client, 'contains',
                    return_value=True) as patched_contains:
                self.assertTrue(backend.exists(self.petrel_path))
                patched_contains.assert_called_once_with(self.expected_path)

        def test_isdir(self):
            backend = PetrelBackend()
            self.assertTrue(has_method(backend._client, 'isdir'))
            # raise Exception if `_client.isdir` is not implemented
            with delete_and_reset_method(backend._client, 'isdir'):
                self.assertFalse(has_method(backend._client, 'isdir'))
                with self.assertRaises(NotImplementedError):
                    backend.isdir(self.petrel_path)

            with patch.object(
                    backend._client, 'isdir',
                    return_value=True) as patched_contains:
                self.assertTrue(backend.isdir(self.petrel_path))
                patched_contains.assert_called_once_with(self.expected_path)

        def test_isfile(self):
            backend = PetrelBackend()
            self.assertTrue(has_method(backend._client, 'contains'))
            # raise Exception if `_client.contains` is not implemented
            with delete_and_reset_method(backend._client, 'contains'):
                self.assertFalse(has_method(backend._client, 'contains'))
                with self.assertRaises(NotImplementedError):
                    backend.isfile(self.petrel_path)

            with patch.object(
                    backend._client, 'contains',
                    return_value=True) as patched_contains:
                self.assertTrue(backend.isfile(self.petrel_path))
                patched_contains.assert_called_once_with(self.expected_path)

        def test_get_local_path(self):
            backend = PetrelBackend()
            with patch.object(backend._client, 'Get',
                              return_value=b'petrel') as patched_get, \
                patch.object(backend._client, 'contains',
                             return_value=True) as patch_contains:
                with backend.get_local_path(self.petrel_path) as path:
                    self.assertTrue(osp.isfile(path))
                    self.assertEqual(Path(path).open('rb').read(), b'petrel')
                # exist the with block and path will be released
                self.assertFalse(osp.isfile(path))
                patched_get.assert_called_once_with(self.expected_path)
                patch_contains.assert_called_once_with(self.expected_path)

        def test_copyfile(self):
            backend = PetrelBackend()
            with patch.object(backend._client, 'Get',
                              return_value=b'petrel') as patched_get, \
                patch.object(backend._client, 'put') as patched_put, \
                patch.object(backend._client, 'isdir', return_value=False) as \
                    patched_isdir:
                src = self.petrel_path
                dst = f'{self.petrel_dir}/test.bak.jpg'
                expected_dst = f'{self.expected_dir}/test.bak.jpg'
                self.assertEqual(backend.copyfile(src, dst), dst)
                patched_get.assert_called_once_with(self.expected_path)
                patched_put.assert_called_once_with(expected_dst, b'petrel')
                patched_isdir.assert_called_once_with(expected_dst)

            with patch.object(backend._client, 'Get',
                              return_value=b'petrel') as patched_get, \
                patch.object(backend._client, 'put') as patched_put, \
                patch.object(backend._client, 'isdir', return_value=True) as \
                    patched_isdir:
                # dst is a directory
                dst = f'{self.petrel_dir}/dir'
                expected_dst = f'{self.expected_dir}/dir/test.jpg'
                self.assertEqual(backend.copyfile(src, dst), f'{dst}/test.jpg')
                patched_get.assert_called_once_with(self.expected_path)
                patched_put.assert_called_once_with(expected_dst, b'petrel')
                patched_isdir.assert_called_once_with(
                    f'{self.expected_dir}/dir')

            with patch.object(backend._client, 'Get',
                              return_value=b'petrel') as patched_get, \
                patch.object(backend._client, 'isdir', return_value=False) as \
                    patched_isdir:
                # src and src should not be same file
                with self.assertRaises(SameFileError):
                    backend.copyfile(src, src)

        def test_copytree(self):
            backend = PetrelBackend()
            put_inputs = []
            get_inputs = []

            def put(obj, filepath):
                put_inputs.append((obj, filepath))

            def get(filepath):
                get_inputs.append(filepath)

            with build_temporary_directory() as tmp_dir, \
                 patch.object(backend, 'put', side_effect=put),\
                 patch.object(backend, 'get', side_effect=get),\
                 patch.object(backend, 'exists', return_value=False):
                tmp_dir = tmp_dir.replace('\\', '/')
                dst = f'{tmp_dir}/dir'
                self.assertEqual(backend.copytree(tmp_dir, dst), dst)

                self.assertEqual(len(put_inputs), 5)
                self.assertEqual(len(get_inputs), 5)

                # dst should not exist
                with patch.object(backend, 'exists', return_value=True):
                    with self.assertRaises(FileExistsError):
                        backend.copytree(dst, tmp_dir)

        def test_copyfile_from_local(self):
            backend = PetrelBackend()
            with patch.object(backend._client, 'put') as patched_put, \
                 patch.object(backend._client, 'isdir', return_value=False) \
                 as patched_isdir:
                src = self.img_path
                dst = f'{self.petrel_dir}/color.bak.jpg'
                expected_dst = f'{self.expected_dir}/color.bak.jpg'
                self.assertEqual(backend.copyfile_from_local(src, dst), dst)
                patched_put.assert_called_once_with(expected_dst,
                                                    src.open('rb').read())
                patched_isdir.assert_called_once_with(expected_dst)

            with patch.object(backend._client, 'put') as patched_put, \
                patch.object(backend._client, 'isdir', return_value=True) as \
                    patched_isdir:
                # dst is a directory
                src = self.img_path
                dst = f'{self.petrel_dir}/dir'
                expected_dst = f'{self.expected_dir}/dir/color.jpg'
                self.assertEqual(
                    backend.copyfile_from_local(src, dst), f'{dst}/color.jpg')
                patched_put.assert_called_once_with(expected_dst,
                                                    src.open('rb').read())
                patched_isdir.assert_called_once_with(
                    f'{self.expected_dir}/dir')

        def test_copytree_from_local(self):
            backend = PetrelBackend()
            inputs = []

            def copyfile_from_local(src, dst):
                inputs.append((src, dst))

            with build_temporary_directory() as tmp_dir, \
                 patch.object(backend, 'copyfile_from_local',
                              side_effect=copyfile_from_local),\
                 patch.object(backend, 'exists', return_value=False):
                backend.copytree_from_local(tmp_dir, self.petrel_dir)

                self.assertEqual(len(inputs), 5)

                # dst should not exist
                with patch.object(backend, 'exists', return_value=True):
                    with self.assertRaises(FileExistsError):
                        backend.copytree_from_local(tmp_dir, self.petrel_dir)

        def test_copyfile_to_local(self):
            backend = PetrelBackend()
            with patch.object(backend._client, 'Get',
                              return_value=b'petrel') as patched_get, \
                 tempfile.TemporaryDirectory() as tmp_dir:
                src = self.petrel_path
                dst = Path(tmp_dir) / 'test.bak.jpg'
                self.assertEqual(backend.copyfile_to_local(src, dst), dst)
                patched_get.assert_called_once_with(self.expected_path)
                self.assertEqual(dst.open('rb').read(), b'petrel')

            with patch.object(backend._client, 'Get',
                              return_value=b'petrel') as patched_get, \
                 tempfile.TemporaryDirectory() as tmp_dir:
                # dst is a directory
                src = self.petrel_path
                dst = Path(tmp_dir) / 'dir'
                dst.mkdir()
                self.assertEqual(
                    backend.copyfile_to_local(src, dst), dst / 'test.jpg')
                patched_get.assert_called_once_with(self.expected_path)
                self.assertEqual((dst / 'test.jpg').open('rb').read(),
                                 b'petrel')

        def test_copytree_to_local(self):
            backend = PetrelBackend()
            inputs = []

            def get(filepath):
                inputs.append(filepath)
                return b'petrel'

            with build_temporary_directory() as tmp_dir, \
                 patch.object(backend, 'get', side_effect=get):
                dst = f'{tmp_dir}/dir'
                backend.copytree_to_local(tmp_dir, dst)

                self.assertEqual(len(inputs), 5)

        def test_remove(self):
            backend = PetrelBackend()
            self.assertTrue(has_method(backend._client, 'delete'))
            # raise Exception if `delete` is not implemented
            with delete_and_reset_method(backend._client, 'delete'):
                self.assertFalse(has_method(backend._client, 'delete'))
                with self.assertRaises(NotImplementedError):
                    backend.remove(self.petrel_path)

            with patch.object(backend._client, 'delete') as patched_delete, \
                 patch.object(backend._client, 'isdir', return_value=False) \
                 as patched_isdir, \
                 patch.object(backend._client, 'contains', return_value=True) \
                 as patched_contains:
                backend.remove(self.petrel_path)
                patched_delete.assert_called_once_with(self.expected_path)
                patched_isdir.assert_called_once_with(self.expected_path)
                patched_contains.assert_called_once_with(self.expected_path)

        def test_rmtree(self):
            backend = PetrelBackend()
            inputs = []

            def remove(filepath):
                inputs.append(filepath)

            with build_temporary_directory() as tmp_dir,\
                 patch.object(backend, 'remove', side_effect=remove):
                backend.rmtree(tmp_dir)

                self.assertEqual(len(inputs), 5)

        def test_copy_if_symlink_fails(self):
            backend = PetrelBackend()
            copyfile_inputs = []
            copytree_inputs = []

            def copyfile(src, dst):
                copyfile_inputs.append((src, dst))

            def copytree(src, dst):
                copytree_inputs.append((src, dst))

            with patch.object(backend, 'copyfile', side_effect=copyfile), \
                 patch.object(backend, 'isfile', return_value=True):
                backend.copy_if_symlink_fails(self.petrel_path, 'path')

                self.assertEqual(len(copyfile_inputs), 1)

            with patch.object(backend, 'copytree', side_effect=copytree), \
                 patch.object(backend, 'isfile', return_value=False):
                backend.copy_if_symlink_fails(self.petrel_dir, 'path')

                self.assertEqual(len(copytree_inputs), 1)

        def test_list_dir_or_file(self):
            backend = PetrelBackend()

            # raise Exception if `_client.list` is not implemented
            self.assertTrue(has_method(backend._client, 'list'))
            with delete_and_reset_method(backend._client, 'list'):
                self.assertFalse(has_method(backend._client, 'list'))
                with self.assertRaises(NotImplementedError):
                    list(backend.list_dir_or_file(self.petrel_dir))

            with build_temporary_directory() as tmp_dir:
                # list directories and files
                self.assertEqual(
                    set(backend.list_dir_or_file(tmp_dir)),
                    {'dir1', 'dir2', 'text1.txt', 'text2.txt'})

                # list directories and files recursively
                self.assertEqual(
                    set(backend.list_dir_or_file(tmp_dir, recursive=True)), {
                        'dir1', '/'.join(('dir1', 'text3.txt')), 'dir2',
                        '/'.join(('dir2', 'dir3')), '/'.join(
                            ('dir2', 'dir3', 'text4.txt')), '/'.join(
                                ('dir2', 'img.jpg')), 'text1.txt', 'text2.txt'
                    })

                # only list directories
                self.assertEqual(
                    set(backend.list_dir_or_file(tmp_dir, list_file=False)),
                    {'dir1', 'dir2'})
                with self.assertRaisesRegex(
                        TypeError,
                        '`list_dir` should be False when `suffix` is not None'
                ):
                    backend.list_dir_or_file(
                        tmp_dir, list_file=False, suffix='.txt')

                # only list directories recursively
                self.assertEqual(
                    set(
                        backend.list_dir_or_file(
                            tmp_dir, list_file=False, recursive=True)),
                    {'dir1', 'dir2', '/'.join(('dir2', 'dir3'))})

                # only list files
                self.assertEqual(
                    set(backend.list_dir_or_file(tmp_dir, list_dir=False)),
                    {'text1.txt', 'text2.txt'})

                # only list files recursively
                self.assertEqual(
                    set(
                        backend.list_dir_or_file(
                            tmp_dir, list_dir=False, recursive=True)),
                    {
                        '/'.join(('dir1', 'text3.txt')), '/'.join(
                            ('dir2', 'dir3', 'text4.txt')), '/'.join(
                                ('dir2', 'img.jpg')), 'text1.txt', 'text2.txt'
                    })

                # only list files ending with suffix
                self.assertEqual(
                    set(
                        backend.list_dir_or_file(
                            tmp_dir, list_dir=False, suffix='.txt')),
                    {'text1.txt', 'text2.txt'})
                self.assertEqual(
                    set(
                        backend.list_dir_or_file(
                            tmp_dir, list_dir=False, suffix=('.txt', '.jpg'))),
                    {'text1.txt', 'text2.txt'})
                with self.assertRaisesRegex(
                        TypeError,
                        '`suffix` must be a string or tuple of strings'):
                    backend.list_dir_or_file(
                        tmp_dir, list_dir=False, suffix=['.txt', '.jpg'])

                # only list files ending with suffix recursively
                self.assertEqual(
                    set(
                        backend.list_dir_or_file(
                            tmp_dir,
                            list_dir=False,
                            suffix='.txt',
                            recursive=True)), {
                                '/'.join(('dir1', 'text3.txt')), '/'.join(
                                    ('dir2', 'dir3', 'text4.txt')),
                                'text1.txt', 'text2.txt'
                            })

                # only list files ending with suffix
                self.assertEqual(
                    set(
                        backend.list_dir_or_file(
                            tmp_dir,
                            list_dir=False,
                            suffix=('.txt', '.jpg'),
                            recursive=True)),
                    {
                        '/'.join(('dir1', 'text3.txt')), '/'.join(
                            ('dir2', 'dir3', 'text4.txt')), '/'.join(
                                ('dir2', 'img.jpg')), 'text1.txt', 'text2.txt'
                    })

        def test_generate_presigned_url(self):
            pass

else:

    class TestPetrelBackend(TestCase):  # type: ignore

        @classmethod
        def setUpClass(cls):
            cls.test_data_dir = Path(__file__).parent.parent.parent / 'data'
            cls.local_img_path = cls.test_data_dir / 'color.jpg'
            cls.local_img_shape = (300, 400, 3)
            cls.petrel_dir = 'petrel://mmengine-test/data'

        def setUp(self):
            backend = PetrelBackend()
            backend.rmtree(self.petrel_dir)
            with build_temporary_directory() as tmp_dir:
                backend.copytree_from_local(tmp_dir, self.petrel_dir)

            text1_path = f'{self.petrel_dir}/text1.txt'
            text2_path = f'{self.petrel_dir}/text2.txt'
            text3_path = f'{self.petrel_dir}/dir1/text3.txt'
            text4_path = f'{self.petrel_dir}/dir2/dir3/text4.txt'
            img_path = f'{self.petrel_dir}/dir2/img.jpg'
            self.assertTrue(backend.isfile(text1_path))
            self.assertTrue(backend.isfile(text2_path))
            self.assertTrue(backend.isfile(text3_path))
            self.assertTrue(backend.isfile(text4_path))
            self.assertTrue(backend.isfile(img_path))

        def test_get(self):
            backend = PetrelBackend()
            img_path = f'{self.petrel_dir}/dir2/img.jpg'
            self.assertEqual(backend.get(img_path), b'img')

        def test_get_text(self):
            backend = PetrelBackend()
            text_path = f'{self.petrel_dir}/text1.txt'
            self.assertEqual(backend.get_text(text_path), 'text1')

        def test_put(self):
            backend = PetrelBackend()
            img_path = f'{self.petrel_dir}/img.jpg'
            backend.put(b'img', img_path)

        def test_put_text(self):
            backend = PetrelBackend()
            text_path = f'{self.petrel_dir}/text5.txt'
            backend.put_text('text5', text_path)

        def test_exists(self):
            backend = PetrelBackend()

            # file and directory exist
            dir_path = f'{self.petrel_dir}/dir2'
            self.assertTrue(backend.exists(dir_path))
            img_path = f'{self.petrel_dir}/dir2/img.jpg'
            self.assertTrue(backend.exists(img_path))

            # file and directory does not exist
            not_existed_dir = f'{self.petrel_dir}/not_existed_dir'
            self.assertFalse(backend.exists(not_existed_dir))
            not_existed_path = f'{self.petrel_dir}/img.jpg'
            self.assertFalse(backend.exists(not_existed_path))

        def test_isdir(self):
            backend = PetrelBackend()
            dir_path = f'{self.petrel_dir}/dir2'
            self.assertTrue(backend.isdir(dir_path))
            img_path = f'{self.petrel_dir}/dir2/img.jpg'
            self.assertFalse(backend.isdir(img_path))

        def test_isfile(self):
            backend = PetrelBackend()
            dir_path = f'{self.petrel_dir}/dir2'
            self.assertFalse(backend.isfile(dir_path))
            img_path = f'{self.petrel_dir}/dir2/img.jpg'
            self.assertTrue(backend.isfile(img_path))

        def test_get_local_path(self):
            backend = PetrelBackend()
            img_path = f'{self.petrel_dir}/dir2/img.jpg'
            with backend.get_local_path(img_path) as path:
                self.assertTrue(osp.isfile(path))
                self.assertEqual(Path(path).open('rb').read(), b'img')
            # exist the with block and path will be released
            self.assertFalse(osp.isfile(path))

        def test_copyfile(self):
            backend = PetrelBackend()

            # dst is a file
            src = f'{self.petrel_dir}/dir2/img.jpg'
            dst = f'{self.petrel_dir}/img.jpg'
            self.assertEqual(backend.copyfile(src, dst), dst)
            self.assertTrue(backend.isfile(dst))

            # dst is a directory
            dst = f'{self.petrel_dir}/dir1'
            expected_dst = f'{self.petrel_dir}/dir1/img.jpg'
            self.assertEqual(backend.copyfile(src, dst), expected_dst)
            self.assertTrue(backend.isfile(expected_dst))

            # src and src should not be same file
            with self.assertRaises(SameFileError):
                backend.copyfile(src, src)

        def test_copytree(self):
            backend = PetrelBackend()
            src = f'{self.petrel_dir}/dir2'
            dst = f'{self.petrel_dir}/dir3'
            self.assertFalse(backend.exists(dst))
            self.assertEqual(backend.copytree(src, dst), dst)
            self.assertEqual(
                list(backend.list_dir_or_file(src)),
                list(backend.list_dir_or_file(dst)))

            # dst should not exist
            with self.assertRaises(FileExistsError):
                backend.copytree(src, dst)

        def test_copyfile_from_local(self):
            backend = PetrelBackend()

            # dst is a file
            src = self.local_img_path
            dst = f'{self.petrel_dir}/color.jpg'
            self.assertFalse(backend.exists(dst))
            self.assertEqual(backend.copyfile_from_local(src, dst), dst)
            self.assertTrue(backend.isfile(dst))

            # dst is a directory
            src = self.local_img_path
            dst = f'{self.petrel_dir}/dir1'
            expected_dst = f'{self.petrel_dir}/dir1/color.jpg'
            self.assertFalse(backend.exists(expected_dst))
            self.assertEqual(
                backend.copyfile_from_local(src, dst), expected_dst)
            self.assertTrue(backend.isfile(expected_dst))

        def test_copytree_from_local(self):
            backend = PetrelBackend()
            backend.rmtree(self.petrel_dir)
            with build_temporary_directory() as tmp_dir:
                backend.copytree_from_local(tmp_dir, self.petrel_dir)
                files = backend.list_dir_or_file(
                    self.petrel_dir, recursive=True)
                self.assertEqual(len(list(files)), 8)

        def test_copyfile_to_local(self):
            backend = PetrelBackend()
            with tempfile.TemporaryDirectory() as tmp_dir:
                # dst is a file
                src = f'{self.petrel_dir}/dir2/img.jpg'
                dst = Path(tmp_dir) / 'img.jpg'
                self.assertEqual(backend.copyfile_to_local(src, dst), dst)
                self.assertEqual(dst.open('rb').read(), b'img')

                # dst is a directory
                dst = Path(tmp_dir) / 'dir'
                dst.mkdir()
                self.assertEqual(
                    backend.copyfile_to_local(src, dst), dst / 'img.jpg')
                self.assertEqual((dst / 'img.jpg').open('rb').read(), b'img')

        def test_copytree_to_local(self):
            backend = PetrelBackend()
            with tempfile.TemporaryDirectory() as tmp_dir:
                backend.copytree_to_local(self.petrel_dir, tmp_dir)
                self.assertTrue(osp.exists(Path(tmp_dir) / 'text1.txt'))
                self.assertTrue(osp.exists(Path(tmp_dir) / 'dir2' / 'img.jpg'))

        def test_remove(self):
            backend = PetrelBackend()
            img_path = f'{self.petrel_dir}/dir2/img.jpg'
            self.assertTrue(backend.isfile(img_path))
            backend.remove(img_path)
            self.assertFalse(backend.exists(img_path))

        def test_rmtree(self):
            backend = PetrelBackend()
            dir_path = f'{self.petrel_dir}/dir2'
            self.assertTrue(backend.isdir(dir_path))
            backend.rmtree(dir_path)
            self.assertFalse(backend.exists(dir_path))

        def test_copy_if_symlink_fails(self):
            backend = PetrelBackend()

            # dst is a file
            src = f'{self.petrel_dir}/dir2/img.jpg'
            dst = f'{self.petrel_dir}/img.jpg'
            self.assertFalse(backend.exists(dst))
            self.assertFalse(backend.copy_if_symlink_fails(src, dst))
            self.assertTrue(backend.isfile(dst))

            # dst is a directory
            src = f'{self.petrel_dir}/dir2'
            dst = f'{self.petrel_dir}/dir'
            self.assertFalse(backend.exists(dst))
            self.assertFalse(backend.copy_if_symlink_fails(src, dst))
            self.assertTrue(backend.isdir(dst))

        def test_list_dir_or_file(self):
            backend = PetrelBackend()

            # list directories and files
            self.assertEqual(
                set(backend.list_dir_or_file(self.petrel_dir)),
                {'dir1', 'dir2', 'text1.txt', 'text2.txt'})

            # list directories and files recursively
            self.assertEqual(
                set(backend.list_dir_or_file(self.petrel_dir, recursive=True)),
                {
                    'dir1', '/'.join(('dir1', 'text3.txt')), 'dir2', '/'.join(
                        ('dir2', 'dir3')), '/'.join(
                            ('dir2', 'dir3', 'text4.txt')), '/'.join(
                                ('dir2', 'img.jpg')), 'text1.txt', 'text2.txt'
                })

            # only list directories
            self.assertEqual(
                set(
                    backend.list_dir_or_file(self.petrel_dir,
                                             list_file=False)),
                {'dir1', 'dir2'})
            with self.assertRaisesRegex(
                    TypeError,
                    '`list_dir` should be False when `suffix` is not None'):
                backend.list_dir_or_file(
                    self.petrel_dir, list_file=False, suffix='.txt')

            # only list directories recursively
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        self.petrel_dir, list_file=False, recursive=True)),
                {'dir1', 'dir2', '/'.join(('dir2', 'dir3'))})

            # only list files
            self.assertEqual(
                set(backend.list_dir_or_file(self.petrel_dir, list_dir=False)),
                {'text1.txt', 'text2.txt'})

            # only list files recursively
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        self.petrel_dir, list_dir=False, recursive=True)),
                {
                    '/'.join(('dir1', 'text3.txt')), '/'.join(
                        ('dir2', 'dir3', 'text4.txt')), '/'.join(
                            ('dir2', 'img.jpg')), 'text1.txt', 'text2.txt'
                })

            # only list files ending with suffix
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        self.petrel_dir, list_dir=False, suffix='.txt')),
                {'text1.txt', 'text2.txt'})
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        self.petrel_dir,
                        list_dir=False,
                        suffix=('.txt', '.jpg'))), {'text1.txt', 'text2.txt'})
            with self.assertRaisesRegex(
                    TypeError,
                    '`suffix` must be a string or tuple of strings'):
                backend.list_dir_or_file(
                    self.petrel_dir, list_dir=False, suffix=['.txt', '.jpg'])

            # only list files ending with suffix recursively
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        self.petrel_dir,
                        list_dir=False,
                        suffix='.txt',
                        recursive=True)), {
                            '/'.join(('dir1', 'text3.txt')), '/'.join(
                                ('dir2', 'dir3', 'text4.txt')), 'text1.txt',
                            'text2.txt'
                        })

            # only list files ending with suffix
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        self.petrel_dir,
                        list_dir=False,
                        suffix=('.txt', '.jpg'),
                        recursive=True)),
                {
                    '/'.join(('dir1', 'text3.txt')), '/'.join(
                        ('dir2', 'dir3', 'text4.txt')), '/'.join(
                            ('dir2', 'img.jpg')), 'text1.txt', 'text2.txt'
                })
