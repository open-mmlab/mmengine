# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn.functional as F

from mmengine import InstanceData
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor
from mmengine.testing import assert_allclose


class TestBaseDataPreprocessor(TestCase):

    def test_init(self):
        base_data_preprocessor = BaseDataPreprocessor()
        self.assertEqual(base_data_preprocessor._device, 'cpu')

    def test_forward(self):
        base_data_preprocessor = BaseDataPreprocessor()
        input1 = torch.randn(1, 3, 5)
        input2 = torch.randn(1, 3, 5)
        label1 = torch.randn(1)
        label2 = torch.randn(1)

        data = [
            dict(inputs=input1, data_sample=label1),
            dict(inputs=input2, data_sample=label2)
        ]

        batch_inputs, batch_labels = base_data_preprocessor(data)
        self.assertEqual(batch_inputs.shape, (2, 1, 3, 5))

        assert_allclose(input1, batch_inputs[0])
        assert_allclose(input2, batch_inputs[1])
        assert_allclose(label1, batch_labels[0])
        assert_allclose(label2, batch_labels[1])

        if torch.cuda.is_available():
            base_data_preprocessor = base_data_preprocessor.cuda()
            batch_inputs, batch_labels = base_data_preprocessor(data)
            self.assertEqual(batch_inputs.device.type, 'cuda')

            base_data_preprocessor = base_data_preprocessor.cpu()
            batch_inputs, batch_labels = base_data_preprocessor(data)
            self.assertEqual(batch_inputs.device.type, 'cpu')

            base_data_preprocessor = base_data_preprocessor.to('cuda:0')
            batch_inputs, batch_labels = base_data_preprocessor(data)
            self.assertEqual(batch_inputs.device.type, 'cuda')


class TestImageDataPreprocessor(TestBaseDataPreprocessor):

    def test_init(self):
        # initiate model without `preprocess_cfg`
        data_processor = ImgDataPreprocessor()
        self.assertFalse(data_processor.channel_conversion)
        assert_allclose(data_processor.mean,
                        torch.tensor([127.5, 127.5, 127.5]).view(-1, 1, 1))
        assert_allclose(data_processor.std,
                        torch.tensor([127.5, 127.5, 127.5]).view(-1, 1, 1))
        self.assertEqual(data_processor.pad_size_divisor, 1)
        assert_allclose(data_processor.pad_value, torch.tensor(0))
        # initiate model with preprocess_cfg` and feat keys
        data_processor = ImgDataPreprocessor(
            bgr_to_rgb=True,
            mean=[0, 0, 0],
            std=[255, 255, 255],
            pad_size_divisor=16,
            pad_value=10)
        self.assertTrue(data_processor.channel_conversion, True)
        assert_allclose(data_processor.mean,
                        torch.tensor([0, 0, 0]).view(-1, 1, 1))
        assert_allclose(data_processor.std,
                        torch.tensor([255, 255, 255]).view(-1, 1, 1))
        assert_allclose(data_processor.pad_value, torch.tensor(10))
        self.assertEqual(data_processor.pad_size_divisor, 16)

        with self.assertRaisesRegex(AssertionError, 'The length of mean'):
            ImgDataPreprocessor(mean=(1, 2))

        with self.assertRaisesRegex(AssertionError, 'The length of std'):
            ImgDataPreprocessor(std=(1, 2))

        with self.assertRaisesRegex(AssertionError, '`bgr2rgb` and `rgb2bgr`'):
            ImgDataPreprocessor(bgr_to_rgb=True, rgb_to_bgr=True)

    def test_forward(self):
        # Test `pad_value`, `to_rgb`, `pad_size_divisor`.
        data_preprocessor = ImgDataPreprocessor(
            mean=[127.5],
            std=[1, 2, 3],
            pad_size_divisor=16,
            pad_value=10,
            rgb_to_bgr=True,
        )
        inputs1 = torch.randn(3, 10, 10)
        inputs2 = torch.randn(3, 15, 15)
        data_sample1 = InstanceData(bboxes=torch.randn(5, 4))
        data_sample2 = InstanceData(bboxes=torch.randn(5, 4))
        data = [
            dict(inputs=inputs1.clone(), data_sample=data_sample1.clone()),
            dict(inputs=inputs2.clone(), data_sample=data_sample2.clone())
        ]

        std = torch.tensor([1, 2, 3]).view(-1, 1, 1)
        inputs1 = (inputs1[[2, 1, 0], ...] - 127.5) / std
        inputs2 = (inputs2[[2, 1, 0], ...] - 127.5) / std
        inputs1 = F.pad(inputs1, (0, 6, 0, 6), value=10)
        inputs2 = F.pad(inputs2, (0, 1, 0, 1), value=10)

        target_inputs = [inputs1, inputs2]
        inputs, data_samples = data_preprocessor(data, True)

        target_data_samples = [data_sample1, data_sample2]
        for input_, data_sample, target_input, target_data_sample in zip(
                inputs, data_samples, target_inputs, target_data_samples):
            assert_allclose(input_, target_input)
            assert_allclose(data_sample.bboxes, target_data_sample.bboxes)

        # Test empty `data_sample`
        data = [dict(inputs=inputs1.clone()), dict(inputs=inputs2.clone())]
        data_preprocessor(data, True)
