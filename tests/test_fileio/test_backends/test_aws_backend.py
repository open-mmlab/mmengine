# Copyright (c) OpenMMLab. All rights reserved.
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

    class MockClient(MagicMock):
        pass

    @patch('boto3.client', MagicMock)
    class TestMockAWSBackend(TestCase):

        @classmethod
        def setUpClass(cls):
            print('11111111111import error')

        def setUp(self) -> None:
            pass

        def test_cal(self):
            self.assertEqual(1, 1)

        def test_cal1(self):
            self.assertEqual(1, 1)

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

            cls.backend = AWSBackend()

        def setUp(self) -> None:
            pass

        def test_list_all_buckets(self):
            buckets = []
            for bucket in self.s3.buckets.all():
                buckets.append(bucket.name)
            self.assertEqual(buckets, ['goog1', 'mmengine', 'yehaochen'])

        # @unittest.skip('test')
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

        def tearDown(self) -> None:
            pass

        @classmethod
        def tearDownClass(cls) -> None:
            pass


# if __name__ == '__main__':
#     unittest.main(testLoader=MyTestLoader())
