# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import platform
import tempfile
from contextlib import contextmanager
from pathlib import Path
from shutil import SameFileError
from unittest import TestCase
from unittest.mock import patch

import cv2
import numpy as np
from parameterized import parameterized

from mmengine.fileio.backends import LocalBackend


def imfrombytes(content):
    img_np = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    return img


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


class TestLocalBackend(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = Path(__file__).parent.parent.parent / 'data'
        cls.img_path = cls.test_data_dir / 'color.jpg'
        cls.img_shape = (300, 400, 3)
        cls.text_path = cls.test_data_dir / 'filelist.txt'

    def test_name(self):
        backend = LocalBackend()
        self.assertEqual(backend.name, 'LocalBackend')

    @parameterized.expand([[Path], [str]])
    def test_get(self, path_type):
        backend = LocalBackend()
        img_bytes = backend.get(path_type(self.img_path))
        self.assertEqual(self.img_path.open('rb').read(), img_bytes)
        img = imfrombytes(img_bytes)
        self.assertEqual(img.shape, self.img_shape)

    @parameterized.expand([[Path], [str]])
    def test_get_text(self, path_type):
        backend = LocalBackend()
        text = backend.get_text(path_type(self.text_path))
        self.assertEqual(self.text_path.open('r').read(), text)

    @parameterized.expand([[Path], [str]])
    def test_put(self, path_type):
        backend = LocalBackend()

        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / 'test.jpg'
            backend.put(b'disk', path_type(filepath))
            self.assertEqual(backend.get(filepath), b'disk')

            # If the directory does not exist, put will create a
            # directory first
            filepath = Path(tmp_dir) / 'not_existed_dir' / 'test.jpg'
            backend.put(b'disk', path_type(filepath))
            self.assertEqual(backend.get(filepath), b'disk')

    @parameterized.expand([[Path], [str]])
    def test_put_text(self, path_type):
        backend = LocalBackend()

        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / 'test.txt'
            backend.put_text('disk', path_type(filepath))
            self.assertEqual(backend.get_text(filepath), 'disk')

            # If the directory does not exist, put_text will create a
            # directory first
            filepath = Path(tmp_dir) / 'not_existed_dir' / 'test.txt'
            backend.put_text('disk', path_type(filepath))
            self.assertEqual(backend.get_text(filepath), 'disk')

    @parameterized.expand([[Path], [str]])
    def test_exists(self, path_type):
        backend = LocalBackend()
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertTrue(backend.exists(path_type(tmp_dir)))
            filepath = Path(tmp_dir) / 'test.txt'
            self.assertFalse(backend.exists(path_type(filepath)))
            backend.put_text('disk', filepath)
            self.assertTrue(backend.exists(path_type(filepath)))
            backend.remove(filepath)

    @parameterized.expand([[Path], [str]])
    def test_isdir(self, path_type):
        backend = LocalBackend()
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertTrue(backend.isdir(path_type(tmp_dir)))
            filepath = Path(tmp_dir) / 'test.txt'
            backend.put_text('disk', filepath)
            self.assertFalse(backend.isdir(path_type(filepath)))

    @parameterized.expand([[Path], [str]])
    def test_isfile(self, path_type):
        backend = LocalBackend()
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertFalse(backend.isfile(path_type(tmp_dir)))
            filepath = Path(tmp_dir) / 'test.txt'
            backend.put_text('disk', filepath)
            self.assertTrue(backend.isfile(path_type(filepath)))

    @parameterized.expand([[Path], [str]])
    def test_join_path(self, path_type):
        backend = LocalBackend()
        filepath = backend.join_path(
            path_type(self.test_data_dir), path_type('file'))
        expected = osp.join(path_type(self.test_data_dir), path_type('file'))
        self.assertEqual(filepath, expected)

        filepath = backend.join_path(
            path_type(self.test_data_dir), path_type('dir'), path_type('file'))
        expected = osp.join(
            path_type(self.test_data_dir), path_type('dir'), path_type('file'))
        self.assertEqual(filepath, expected)

    @parameterized.expand([[Path], [str]])
    def test_get_local_path(self, path_type):
        backend = LocalBackend()
        with backend.get_local_path(path_type(self.text_path)) as filepath:
            self.assertEqual(path_type(self.text_path), path_type(filepath))

    @parameterized.expand([[Path], [str]])
    def test_copyfile(self, path_type):
        backend = LocalBackend()
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / 'test.txt'
            backend.put_text('disk', src)
            dst = Path(tmp_dir) / 'test.txt.bak'
            self.assertEqual(
                backend.copyfile(path_type(src), path_type(dst)),
                path_type(dst))
            self.assertEqual(backend.get_text(dst), 'disk')

            # dst is a directory
            dst = Path(tmp_dir) / 'dir'
            dst.mkdir()
            self.assertEqual(
                backend.copyfile(path_type(src), path_type(dst)),
                backend.join_path(path_type(dst), 'test.txt'))
            self.assertEqual(
                backend.get_text(backend.join_path(dst, 'test.txt')), 'disk')

            # src and src should not be same file
            with self.assertRaises(SameFileError):
                backend.copyfile(path_type(src), path_type(src))

    @parameterized.expand([[Path], [str]])
    def test_copytree(self, path_type):
        backend = LocalBackend()
        with build_temporary_directory() as tmp_dir:
            # src and dst are Path objects
            src = Path(tmp_dir) / 'dir1'
            dst = Path(tmp_dir) / 'dir100'
            self.assertEqual(
                backend.copytree(path_type(src), path_type(dst)),
                path_type(dst))
            self.assertTrue(backend.isdir(dst))
            self.assertTrue(backend.isfile(dst / 'text3.txt'))
            self.assertEqual(backend.get_text(dst / 'text3.txt'), 'text3')

            # dst should not exist
            with self.assertRaises(FileExistsError):
                backend.copytree(
                    path_type(src), path_type(Path(tmp_dir) / 'dir2'))

    @parameterized.expand([[Path], [str]])
    def test_copyfile_from_local(self, path_type):
        backend = LocalBackend()
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / 'test.txt'
            backend.put_text('disk', src)
            dst = Path(tmp_dir) / 'test.txt.bak'
            self.assertEqual(
                backend.copyfile(path_type(src), path_type(dst)),
                path_type(dst))
            self.assertEqual(backend.get_text(dst), 'disk')

            dst = Path(tmp_dir) / 'dir'
            dst.mkdir()
            self.assertEqual(
                backend.copyfile(path_type(src), path_type(dst)),
                backend.join_path(path_type(dst), 'test.txt'))
            self.assertEqual(
                backend.get_text(backend.join_path(dst, 'test.txt')), 'disk')

            # src and src should not be same file
            with self.assertRaises(SameFileError):
                backend.copyfile(path_type(src), path_type(src))

    @parameterized.expand([[Path], [str]])
    def test_copytree_from_local(self, path_type):
        backend = LocalBackend()
        with build_temporary_directory() as tmp_dir:
            # src and dst are Path objects
            src = Path(tmp_dir) / 'dir1'
            dst = Path(tmp_dir) / 'dir100'
            self.assertEqual(
                backend.copytree(path_type(src), path_type(dst)),
                path_type(dst))
            self.assertTrue(backend.isdir(dst))
            self.assertTrue(backend.isfile(dst / 'text3.txt'))
            self.assertEqual(backend.get_text(dst / 'text3.txt'), 'text3')

            # dst should not exist
            with self.assertRaises(FileExistsError):
                backend.copytree(
                    path_type(src), path_type(Path(tmp_dir) / 'dir2'))

    @parameterized.expand([[Path], [str]])
    def test_copyfile_to_local(self, path_type):
        backend = LocalBackend()
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / 'test.txt'
            backend.put_text('disk', src)
            dst = Path(tmp_dir) / 'test.txt.bak'
            self.assertEqual(
                backend.copyfile(path_type(src), path_type(dst)),
                path_type(dst))
            self.assertEqual(backend.get_text(dst), 'disk')

            dst = Path(tmp_dir) / 'dir'
            dst.mkdir()
            self.assertEqual(
                backend.copyfile(path_type(src), path_type(dst)),
                backend.join_path(path_type(dst), 'test.txt'))
            self.assertEqual(
                backend.get_text(backend.join_path(dst, 'test.txt')), 'disk')

            # src and src should not be same file
            with self.assertRaises(SameFileError):
                backend.copyfile(path_type(src), path_type(src))

    @parameterized.expand([[Path], [str]])
    def test_copytree_to_local(self, path_type):
        backend = LocalBackend()
        with build_temporary_directory() as tmp_dir:
            # src and dst are Path objects
            src = Path(tmp_dir) / 'dir1'
            dst = Path(tmp_dir) / 'dir100'
            self.assertEqual(
                backend.copytree(path_type(src), path_type(dst)),
                path_type(dst))
            self.assertTrue(backend.isdir(dst))
            self.assertTrue(backend.isfile(dst / 'text3.txt'))
            self.assertEqual(backend.get_text(dst / 'text3.txt'), 'text3')

            # dst should not exist
            with self.assertRaises(FileExistsError):
                backend.copytree(
                    path_type(src), path_type(Path(tmp_dir) / 'dir2'))

    @parameterized.expand([[Path], [str]])
    def test_remove(self, path_type):
        backend = LocalBackend()
        with tempfile.TemporaryDirectory() as tmp_dir:
            # filepath is a Path object
            filepath = Path(tmp_dir) / 'test.txt'
            backend.put_text('disk', filepath)
            self.assertTrue(backend.exists(filepath))
            backend.remove(path_type(filepath))
            self.assertFalse(backend.exists(filepath))

            # raise error if file does not exist
            with self.assertRaises(FileNotFoundError):
                filepath = Path(tmp_dir) / 'test1.txt'
                backend.remove(path_type(filepath))

            # can not remove directory
            filepath = Path(tmp_dir) / 'dir'
            filepath.mkdir()
            with self.assertRaises(IsADirectoryError):
                backend.remove(path_type(filepath))

    @parameterized.expand([[Path], [str]])
    def test_rmtree(self, path_type):
        backend = LocalBackend()
        with build_temporary_directory() as tmp_dir:
            # src and dst are Path objects
            dir_path = Path(tmp_dir) / 'dir1'
            self.assertTrue(backend.exists(dir_path))
            backend.rmtree(path_type(dir_path))
            self.assertFalse(backend.exists(dir_path))

            dir_path = Path(tmp_dir) / 'dir2'
            self.assertTrue(backend.exists(dir_path))
            backend.rmtree(path_type(dir_path))
            self.assertFalse(backend.exists(dir_path))

    @parameterized.expand([[Path], [str]])
    def test_copy_if_symlink_fails(self, path_type):
        backend = LocalBackend()
        with tempfile.TemporaryDirectory() as tmp_dir:
            # create a symlink for a file
            src = Path(tmp_dir) / 'test.txt'
            backend.put_text('disk', src)
            dst = Path(tmp_dir) / 'test_link.txt'
            res = backend.copy_if_symlink_fails(path_type(src), path_type(dst))
            if platform.system() == 'Linux':
                self.assertTrue(res)
                self.assertTrue(osp.islink(dst))
            self.assertEqual(backend.get_text(dst), 'disk')

            # create a symlink for a directory
            src = Path(tmp_dir) / 'dir'
            src.mkdir()
            dst = Path(tmp_dir) / 'dir_link'
            res = backend.copy_if_symlink_fails(path_type(src), path_type(dst))
            if platform.system() == 'Linux':
                self.assertTrue(res)
                self.assertTrue(osp.islink(dst))
            self.assertTrue(backend.exists(dst))

            def symlink(src, dst):
                raise Exception

            # copy files if symblink fails
            with patch.object(os, 'symlink', side_effect=symlink):
                src = Path(tmp_dir) / 'test.txt'
                dst = Path(tmp_dir) / 'test_link1.txt'
                res = backend.copy_if_symlink_fails(
                    path_type(src), path_type(dst))
                self.assertFalse(res)
                self.assertFalse(osp.islink(dst))
                self.assertTrue(backend.exists(dst))

            # copy directory if symblink fails
            with patch.object(os, 'symlink', side_effect=symlink):
                src = Path(tmp_dir) / 'dir'
                dst = Path(tmp_dir) / 'dir_link1'
                res = backend.copy_if_symlink_fails(
                    path_type(src), path_type(dst))
                self.assertFalse(res)
                self.assertFalse(osp.islink(dst))
                self.assertTrue(backend.exists(dst))

    @parameterized.expand([[Path], [str]])
    def test_list_dir_or_file(self, path_type):
        backend = LocalBackend()
        with build_temporary_directory() as tmp_dir:
            # list directories and files
            self.assertEqual(
                set(backend.list_dir_or_file(path_type(tmp_dir))),
                {'dir1', 'dir2', 'text1.txt', 'text2.txt'})

            # list directories and files recursively
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        path_type(tmp_dir), recursive=True)),
                {
                    'dir1',
                    osp.join('dir1', 'text3.txt'), 'dir2',
                    osp.join('dir2', 'dir3'),
                    osp.join('dir2', 'dir3', 'text4.txt'),
                    osp.join('dir2', 'img.jpg'), 'text1.txt', 'text2.txt'
                })

            # only list directories
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        path_type(tmp_dir), list_file=False)),
                {'dir1', 'dir2'})

            with self.assertRaisesRegex(
                    TypeError,
                    '`suffix` should be None when `list_dir` is True'):
                backend.list_dir_or_file(
                    path_type(tmp_dir), list_file=False, suffix='.txt')

            # only list directories recursively
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        path_type(tmp_dir), list_file=False, recursive=True)),
                {'dir1', 'dir2', osp.join('dir2', 'dir3')})

            # only list files
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        path_type(tmp_dir), list_dir=False)),
                {'text1.txt', 'text2.txt'})

            # only list files recursively
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        path_type(tmp_dir), list_dir=False, recursive=True)),
                {
                    osp.join('dir1', 'text3.txt'),
                    osp.join('dir2', 'dir3', 'text4.txt'),
                    osp.join('dir2', 'img.jpg'), 'text1.txt', 'text2.txt'
                })

            # only list files ending with suffix
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        path_type(tmp_dir), list_dir=False, suffix='.txt')),
                {'text1.txt', 'text2.txt'})
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        path_type(tmp_dir),
                        list_dir=False,
                        suffix=('.txt', '.jpg'))), {'text1.txt', 'text2.txt'})

            with self.assertRaisesRegex(
                    TypeError,
                    '`suffix` must be a string or tuple of strings'):
                backend.list_dir_or_file(
                    path_type(tmp_dir),
                    list_dir=False,
                    suffix=['.txt', '.jpg'])

            # only list files ending with suffix recursively
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        path_type(tmp_dir),
                        list_dir=False,
                        suffix='.txt',
                        recursive=True)), {
                            osp.join('dir1', 'text3.txt'),
                            osp.join('dir2', 'dir3', 'text4.txt'), 'text1.txt',
                            'text2.txt'
                        })

            # only list files ending with suffix
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        path_type(tmp_dir),
                        list_dir=False,
                        suffix=('.txt', '.jpg'),
                        recursive=True)),
                {
                    osp.join('dir1', 'text3.txt'),
                    osp.join('dir2', 'dir3', 'text4.txt'),
                    osp.join('dir2', 'img.jpg'), 'text1.txt', 'text2.txt'
                })
