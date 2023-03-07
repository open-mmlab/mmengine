# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import unittest
from typing import Sequence
from unittest import TestCase, TestLoader
from unittest.mock import MagicMock, patch

from mmengine.fileio import AWSBackend


class MyTestLoader(TestLoader):

    def getTestCaseNames(self, testCaseClass) -> Sequence[str]:
        testNames = super().getTestCaseNames(
            testCaseClass)  # get all test function objects
        testCaseMethods = list(
            testCaseClass.__dict__.keys())  # get all function names
        testNames = list(testNames)
        testNames.sort(key=testCaseMethods.index
                       )  # invoke function using the order of function names
        return testNames


try:
    import boto3
    from botocore.exceptions import ClientError

    # raise ImportError('aws')

except ImportError:
    sys.modules['boto3'] = MagicMock()
    sys.modules['boto3.client'] = MagicMock()

    class MockAWSClient(MagicMock):

        def __init__(self, *args, **kwargs) -> None:
            pass

        def head_bucket(self, Bucket):
            return True

        def head_object(self, Bucket, Key):
            return True

        def download_fileobj(self, bucket, obj_name, value, *args, **kwargs):
            with open(bucket, 'rb') as f:
                content = f.read()
                value.write(content)

        def upload_fileobj(self, buff, bucket, obj_name, *args, **kwargs):
            pass

        def delete_object(self, *args, **kwargs):
            pass

        def get_object(Bucket, Key):
            return b's3'

        def get_paginator(self, type='list_objects_v2'):

            class Paginator:

                @staticmethod
                def paginate(Bucket, Prefix, *args, **kwargs):
                    dir_path = Prefix
                    paths = []
                    for root, dirs, files in os.walk(dir_path):
                        for name in files:
                            paths.append(os.path.join(root, name))
                    response = {
                        'ResponseMetadata': {
                            'HTTPStatusCode': 200
                        },
                        'Contents': [{
                            'Key': path.replace('\\', '/')
                        } for path in paths]
                    }
                    return [response]

            return Paginator

        @staticmethod
        def _parse_path(obj, filepath):
            return str(filepath), str(filepath)

    @patch('boto3.client', MockAWSClient)
    class TestMockAWSBackend(TestCase):

        @classmethod
        def setUpClass(cls):
            cls.aws = 'aws://mmengine/'
            cls.backend = AWSBackend()

        def setUp(self) -> None:
            pass

        def test_get(self):
            file_path = self.aws + 'test.txt'
            with patch.object(self.backend._client, 'get_object') as fobj:
                self.backend.get(file_path)
                fobj.assert_called_once()

        def test_get_text(self):
            file_path = self.aws + 's3.txt'
            with patch.object(self.backend, 'get') as fobj:
                self.backend.get_text(file_path)
                fobj.assert_called_once()

        def test_put(self):
            file_path = self.aws + 'test_put.txt'
            with patch.object(self.backend._client, 'upload_fileobj') as fobj:
                self.backend.put(b'file_path', file_path)
                fobj.assert_called_once()

        def test_put_text(self):
            file_path = self.aws + 'test_put.txt'
            with patch.object(self.backend, 'put') as fobj:
                self.backend.put_text('file_path', file_path)
                fobj.assert_called_once()

        def test_remove(self):
            file_path = self.aws + 'test_put.txt'
            with patch.object(self.backend._client, 'delete_object') as fobj:
                self.backend.remove(file_path)
                fobj.assert_called_once()

        def test_exists(self):
            exit_file_path = self.aws + 's3.txt'
            with patch.object(self.backend, '_check_object') as fobj:
                self.backend.exists(exit_file_path)
                fobj.assert_called_once()

        def test_isdir(self):
            file_path = self.aws + 'data/'
            with patch.object(self.backend, '_check_object') as fobj:
                self.backend.exists(file_path)
                fobj.assert_called_once()

        def test_isfile(self):
            # create a file
            file_path = self.aws + 'data/' + 'tmp.txt'
            with patch.object(self.backend, '_check_object') as fobj:
                self.backend.exists(file_path)
                fobj.assert_called_once()

        def test_get_local_path(self):
            file_path = self.aws + 'data/' + 'tmp.txt'
            with patch.object(self.backend, 'get', return_value=b'ss') as fobj:
                with self.backend.get_local_path(file_path):
                    fobj.assert_called_once()

        def test_list_dir_or_file(self):
            pass

        def tearDown(self) -> None:
            pass

        @classmethod
        def tearDownClass(self) -> None:
            pass

else:

    class TestAWSBackend(TestCase):
        aws = 'aws://mmengine/'

        @classmethod
        def setUpClass(cls):
            cls.s3 = boto3.resource('s3')
            # Upload a new file for test
            cls.s3.Bucket('mmengine').put_object(Key='s3.txt', Body=b's3')
            cls.s3.Bucket('mmengine').put_object(Key='dir/a.txt', Body=b's3')
            cls.s3.Bucket('mmengine').put_object(Key='dir/b.txt', Body=b's3')
            cls.s3.Bucket('mmengine').put_object(Key='dir/b.jpg', Body=b's3')

            cls.backend = AWSBackend()

        def setUp(self) -> None:
            pass

        def test_list_all_buckets(self):
            buckets = []
            for bucket in self.s3.buckets.all():
                buckets.append(bucket.name)
            self.assertEqual(buckets, ['goog1', 'mmengine', 'yehaochen'])

        @unittest.skip('test')
        def test_create_bucket(self):
            # Create bucket
            flag = False
            try:
                s3_client = boto3.client('s3')
                s3_client.create_bucket(Bucket='goog1')
                flag = True
            except ClientError as e:
                print(e)
                flag = False
            self.assertEqual(flag, True)

        def test_delete_bucket(self):
            pass

        def test_upload_file(self):
            pass

        def test_create_file(self):
            pass

        def test_list_list_dir_or_file(self):
            pass

        def test_get(self):
            file_path = self.aws + 's3.txt'
            self.assertEqual(self.backend.get(file_path), b's3')

        def test_get_text(self):
            file_path = self.aws + 's3.txt'
            self.assertEqual(self.backend.get_text(file_path), 's3')

        def test_put(self):
            file_path = self.aws + 'test_put.txt'
            self.backend.put(b'test put', file_path)
            self.assertEqual(self.backend.get_text(file_path), 'test put')
            self.backend.remove(file_path)

        def test_put_text(self):
            file_path = self.aws + 'test_put.txt'
            self.backend.put_text('test put', file_path)
            self.assertEqual(self.backend.get_text(file_path), 'test put')
            self.backend.remove(file_path)

        def test_remove(self):
            file_path = self.aws + 'test_put.txt'
            self.backend.put_text('test put', file_path)
            self.assertEqual(self.backend.get_text(file_path), 'test put')
            self.backend.remove(file_path)
            self.assertFalse(self.backend.exists(file_path))

        def test_exists(self):
            exit_file_path = self.aws + 's3.txt'
            not_exit_file_path = self.aws + 'ss3.txt'
            self.assertTrue(self.backend.exists(exit_file_path))
            self.assertFalse(self.backend.exists(not_exit_file_path))

        def test_isdir(self):
            # create a file
            file_path = self.aws + 'data/' + 'tmp.txt'
            self.backend.put_text('tmp', file_path)
            self.assertEqual(self.backend.get_text(file_path), 'tmp')
            self.assertTrue(self.backend.isdir(self.aws + 'data/'))
            self.backend.remove(file_path)

        def test_isfile(self):
            # create a file
            file_path = self.aws + 'data/' + 'tmp.txt'
            self.backend.put_text('tmp', file_path)
            self.assertEqual(self.backend.get_text(file_path), 'tmp')
            self.assertTrue(self.backend.isfile(file_path))
            self.backend.remove(file_path)

        def test_get_local_path(self):
            file_path = self.aws + 's3.txt'
            with self.backend.get_local_path(file_path) as f:
                self.assertEqual(open(f).read(), 's3')

        def test_list_dir_or_file(self):
            dir_path = self.aws + 'dir'
            self.assertEqual(
                set(self.backend.list_dir_or_file(dir_path)),
                {'a.txt', 'b.txt', 'b.jpg'})

        def tearDown(self) -> None:
            pass

        @classmethod
        def tearDownClass(cls) -> None:
            pass


# if __name__ == '__main__':
#     unittest.main(testLoader=MyTestLoader())
