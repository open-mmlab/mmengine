# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from unittest import TestCase

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


class TestOSSBackend(TestCase):  # type: ignore

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter('ignore', ResourceWarning)
        cls.test_data_dir = Path(__file__).parent.parent.parent / 'data'
        cls.local_img_path = cls.test_data_dir / 'color.jpg'
        cls.local_img_shape = (300, 400, 3)
        cls.oss_dir = 'oss://oss-cn-hangzhou.aliyuncs.com/mmengine/data'

    def setUp(self):
        self.backend = OSSBackend(access_key_id='xxx', access_key_secret='xxx')
        self.backend.rmtree(self.oss_dir)
        with build_temporary_directory() as tmp_dir:
            self.backend.copytree_from_local(tmp_dir, self.oss_dir)

        text1_path = f'{self.oss_dir}/text1.txt'
        text2_path = f'{self.oss_dir}/text2.txt'
        text3_path = f'{self.oss_dir}/dir1/text3.txt'
        text4_path = f'{self.oss_dir}/dir2/dir3/text4.txt'
        img_path = f'{self.oss_dir}/dir2/img.jpg'
        self.assertTrue(self.backend.exists(text1_path))
        self.assertTrue(self.backend.exists(text2_path))
        self.assertTrue(self.backend.exists(text3_path))
        self.assertTrue(self.backend.exists(text4_path))
        self.assertTrue(self.backend.exists(img_path))

    def test_get(self):
        img_path = f'{self.oss_dir}/dir2/img.jpg'
        self.assertEqual(self.backend.get(img_path), b'img')

    def test_get_text(self):
        text_path = f'{self.oss_dir}/text1.txt'
        self.assertEqual(self.backend.get_text(text_path), 'text1')

    def test_put(self):
        img_path = f'{self.oss_dir}/img.jpg'
        self.backend.put(b'img', img_path)

    def test_put_text(self):
        text_path = f'{self.oss_dir}/text5.txt'
        self.backend.put_text('text5', text_path)

    def test_exists(self):
        # file  exist
        img_path = f'{self.oss_dir}/dir2/img.jpg'
        self.assertTrue(self.backend.exists(img_path))

        # file and directory does not exist
        not_existed_path = f'{self.oss_dir}/img.jpg'
        self.assertFalse(self.backend.exists(not_existed_path))

    def test_list_dir_or_file(self):
        dir_path = f'{self.oss_dir}/dir2/'
        print(list(self.backend.list_dir_or_file(dir_path)))
        print(list(self.backend.list_dir_or_file(dir_path, list_dir=False)))
        print(list(self.backend.list_dir_or_file(dir_path, list_file=False)))
        print(
            list(
                self.backend.list_dir_or_file(
                    dir_path, list_dir=False, suffix='.jpg')))
