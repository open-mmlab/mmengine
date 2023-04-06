# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn.functional as F

from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor
from mmengine.structures import InstanceData
from mmengine.testing import assert_allclose


class TestBaseDataPreprocessor(TestCase):

    def test_init(self):
        base_data_preprocessor = BaseDataPreprocessor()
        self.assertEqual(base_data_preprocessor._device.type, 'cpu')
        self.assertEqual(base_data_preprocessor._non_blocking, False)

        base_data_preprocessor = BaseDataPreprocessor(True)
        self.assertEqual(base_data_preprocessor._device.type, 'cpu')
        self.assertEqual(base_data_preprocessor._non_blocking, True)

    def test_forward(self):
        # Test cpu forward with list of data samples.
        base_data_preprocessor = BaseDataPreprocessor()
        input1 = torch.randn(1, 3, 5)
        input2 = torch.randn(1, 3, 5)
        label1 = torch.randn(1)
        label2 = torch.randn(1)

        # Test with dict of batch inputs and batch data samples
        data = dict(inputs=[input1, input2], data_sample=[label1, label2])
        output = base_data_preprocessor(data)
        batch_inputs, batch_labels = output['inputs'], output['data_sample']
        self.assertTrue(torch.is_floating_point(batch_inputs[0]))
        self.assertEqual(batch_inputs[0].shape, (1, 3, 5))

        assert_allclose(input1, batch_inputs[0])
        assert_allclose(input2, batch_inputs[1])
        assert_allclose(label1, batch_labels[0])
        assert_allclose(label2, batch_labels[1])

        # Test with tuple of batch inputs and batch data samples
        data = (torch.stack([input1, input2]), (label1, label2))
        batch_inputs, batch_labels = base_data_preprocessor(data)
        self.assertTrue(torch.is_floating_point(batch_inputs))
        self.assertEqual(batch_inputs[0].shape, (1, 3, 5))
        self.assertEqual(batch_inputs[1].shape, (1, 3, 5))
        self.assertTrue(torch.is_floating_point(batch_inputs[0]))

        # Test cuda forward
        if torch.cuda.is_available():
            # Test with list of data samples.
            data = dict(inputs=[input1, input2], data_sample=[label1, label2])
            base_data_preprocessor = base_data_preprocessor.cuda()
            output = base_data_preprocessor(data)
            batch_inputs, batch_labels = output['inputs'], output[
                'data_sample']
            self.assertTrue(torch.is_floating_point(batch_inputs[0]))
            self.assertEqual(batch_inputs[0].device.type, 'cuda')

            # Fallback to test with cpu.
            base_data_preprocessor = base_data_preprocessor.cpu()
            output = base_data_preprocessor(data)
            batch_inputs, batch_labels = output['inputs'], output[
                'data_sample']
            self.assertTrue(torch.is_floating_point(batch_inputs[0]))
            self.assertEqual(batch_inputs[0].device.type, 'cpu')

            # Test `base_data_preprocessor` can be moved to cuda again.
            base_data_preprocessor = base_data_preprocessor.to('cuda:0')
            output = base_data_preprocessor(data)
            batch_inputs, batch_labels = output['inputs'], output[
                'data_sample']
            self.assertTrue(torch.is_floating_point(batch_inputs[0]))
            self.assertEqual(batch_inputs[0].device.type, 'cuda')

            # device of `base_data_preprocessor` is cuda, output should be
            # cuda tensor.
            self.assertEqual(batch_inputs[0].device.type, 'cuda')
            self.assertEqual(batch_labels[0].device.type, 'cuda')

        # Test forward with string value
        data = dict(string='abc')
        base_data_preprocessor(data)


class TestImgDataPreprocessor(TestBaseDataPreprocessor):

    def test_init(self):
        # Initiate processor without arguments
        data_processor = ImgDataPreprocessor()
        self.assertFalse(data_processor._channel_conversion)
        self.assertFalse(hasattr(data_processor, 'mean'))
        self.assertFalse(hasattr(data_processor, 'std'))
        self.assertEqual(data_processor.pad_size_divisor, 1)
        assert_allclose(data_processor.pad_value, torch.tensor(0))

        # Initiate model with bgr2rgb, mean, std .etc..
        data_processor = ImgDataPreprocessor(
            bgr_to_rgb=True,
            mean=[0, 0, 0],
            std=[255, 255, 255],
            pad_size_divisor=16,
            pad_value=10)
        self.assertTrue(data_processor._enable_normalize)
        self.assertTrue(data_processor._channel_conversion, True)
        assert_allclose(data_processor.mean,
                        torch.tensor([0, 0, 0]).view(-1, 1, 1))
        assert_allclose(data_processor.std,
                        torch.tensor([255, 255, 255]).view(-1, 1, 1))
        assert_allclose(data_processor.pad_value, torch.tensor(10))
        self.assertEqual(data_processor.pad_size_divisor, 16)

        with self.assertRaisesRegex(AssertionError, '`mean` should have'):
            ImgDataPreprocessor(mean=(1, 2), std=(1, 2, 3))

        with self.assertRaisesRegex(AssertionError, '`std` should have'):
            ImgDataPreprocessor(mean=(1, 2, 3), std=(1, 2))

        with self.assertRaisesRegex(AssertionError, '`bgr2rgb` and `rgb2bgr`'):
            ImgDataPreprocessor(bgr_to_rgb=True, rgb_to_bgr=True)

        with self.assertRaisesRegex(AssertionError, 'mean and std should be'):
            ImgDataPreprocessor(
                bgr_to_rgb=True,
                mean=None,
                std=[255, 255, 255],
                pad_size_divisor=16,
                pad_value=10)

        data_processor = ImgDataPreprocessor(
            bgr_to_rgb=True, pad_size_divisor=16, pad_value=10)
        self.assertFalse(data_processor._enable_normalize)

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

        data = dict(
            inputs=[inputs1.clone(), inputs2.clone()],
            data_sample=[data_sample1.clone(),
                         data_sample2.clone()])

        std = torch.tensor([1, 2, 3]).view(-1, 1, 1)
        target_inputs1 = (inputs1.clone()[[2, 1, 0], ...] - 127.5) / std
        target_inputs2 = (inputs2.clone()[[2, 1, 0], ...] - 127.5) / std

        target_inputs1 = F.pad(target_inputs1, (0, 6, 0, 6), value=10)
        target_inputs2 = F.pad(target_inputs2, (0, 1, 0, 1), value=10)

        target_inputs = [target_inputs1, target_inputs2]
        output = data_preprocessor(data, True)
        inputs, data_samples = output['inputs'], output['data_sample']
        self.assertTrue(torch.is_floating_point(inputs))

        target_data_samples = [data_sample1, data_sample2]
        for input_, data_sample, target_input, target_data_sample in zip(
                inputs, data_samples, target_inputs, target_data_samples):
            assert_allclose(input_, target_input)
            assert_allclose(data_sample.bboxes, target_data_sample.bboxes)

        # Test image without normalization.
        data_preprocessor = ImgDataPreprocessor(
            pad_size_divisor=16,
            pad_value=10,
            rgb_to_bgr=True,
        )
        target_inputs1 = (inputs1.clone()[[2, 1, 0], ...])
        target_inputs2 = (inputs2.clone()[[2, 1, 0], ...])
        target_inputs1 = F.pad(target_inputs1, (0, 6, 0, 6), value=10)
        target_inputs2 = F.pad(target_inputs2, (0, 1, 0, 1), value=10)

        target_inputs = [target_inputs1, target_inputs2]
        output = data_preprocessor(data, True)
        inputs, data_samples = output['inputs'], output['data_sample']
        self.assertTrue(torch.is_floating_point(inputs))

        target_data_samples = [data_sample1, data_sample2]
        for input_, data_sample, target_input, target_data_sample in zip(
                inputs, data_samples, target_inputs, target_data_samples):
            assert_allclose(input_, target_input)
            assert_allclose(data_sample.bboxes, target_data_sample.bboxes)

        # Test gray image with 3 dim mean will raise error
        data_preprocessor = ImgDataPreprocessor(
            mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5))
        data = dict(
            inputs=[torch.ones(10, 10), torch.ones(10, 10)], data_sample=None)
        with self.assertRaisesRegex(AssertionError,
                                    'If the mean has 3 values'):
            data_preprocessor(data)

        data = dict(
            inputs=[torch.ones(10, 10), torch.ones(10, 10)], data_sample=None)
        with self.assertRaisesRegex(AssertionError,
                                    'If the mean has 3 values'):
            data_preprocessor(data)

        # Test stacked batch inputs and batch data samples
        data_preprocessor = ImgDataPreprocessor(
            mean=(127.5, 127.5, 127.5),
            std=(127.5, 127.5, 127.5),
            rgb_to_bgr=True,
            pad_size_divisor=16)
        _batch_inputs = torch.randn(2, 3, 10, 10)
        _batch_labels = [torch.randn(1), torch.randn(1)]
        data = dict(inputs=_batch_inputs, data_sample=_batch_labels)
        output = data_preprocessor(data)
        inputs, data_samples = output['inputs'], output['data_sample']
        target_batch_inputs = _batch_inputs[:, [2, 1, 0], ...]
        target_batch_inputs = (target_batch_inputs - 127.5) / 127.5
        target_batch_inputs = F.pad(target_batch_inputs, (0, 6, 0, 6), value=0)
        self.assertEqual(inputs.shape, torch.Size([2, 3, 16, 16]))
        self.assertTrue(torch.is_floating_point(inputs))
        assert_allclose(target_batch_inputs, inputs)

        # Test batch inputs without convert channel order and pad
        data_preprocessor = ImgDataPreprocessor(
            mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5))
        _batch_inputs = torch.randn(2, 3, 10, 10)
        _batch_labels = [torch.randn(1), torch.randn(1)]
        data = dict(inputs=_batch_inputs, data_sample=_batch_labels)
        output = data_preprocessor(data)
        inputs, data_samples = output['inputs'], output['data_sample']
        target_batch_inputs = (_batch_inputs - 127.5) / 127.5
        self.assertEqual(inputs.shape, torch.Size([2, 3, 10, 10]))
        self.assertTrue(torch.is_floating_point(inputs))
        assert_allclose(target_batch_inputs, inputs)

        # Test empty `data_sample`
        data = dict(
            inputs=[inputs1.clone(), inputs2.clone()], data_sample=None)
        output = data_preprocessor(data, True)
        inputs, data_samples = output['inputs'], output['data_sample']
        self.assertIsNone(data_samples)
        self.assertTrue(torch.is_floating_point(inputs))
