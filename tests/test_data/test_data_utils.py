# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch

from mmengine.dataset import default_collate, pseudo_collate
from mmengine.structures import BaseDataElement
from mmengine.utils import is_list_of


class TestDataUtils(TestCase):

    def test_pseudo_collate(self):
        # Test with list of dict tensor inputs.
        input1 = torch.randn(1, 3, 5)
        input2 = torch.randn(1, 3, 5)
        label1 = torch.randn(1)
        label2 = torch.randn(1)

        data_batch = [
            dict(inputs=input1, data_sample=label1),
            dict(inputs=input2, data_sample=label2)
        ]
        data_batch = pseudo_collate(data_batch)
        self.assertTrue(torch.allclose(input1, data_batch['inputs'][0]))
        self.assertTrue(torch.allclose(input2, data_batch['inputs'][1]))
        self.assertTrue(torch.allclose(label1, data_batch['data_sample'][0]))
        self.assertTrue(torch.allclose(label2, data_batch['data_sample'][1]))

        # Test with list of dict, and each element contains `data_sample`
        # inputs
        data_sample1 = BaseDataElement(label=torch.tensor(1))
        data_sample2 = BaseDataElement(label=torch.tensor(1))
        data = [
            dict(inputs=input1, data_sample=data_sample1),
            dict(inputs=input2, data_sample=data_sample2),
        ]
        data_batch = pseudo_collate(data)
        batch_inputs, batch_data_sample = (data_batch['inputs'],
                                           data_batch['data_sample'])
        # check batch_inputs
        self.assertTrue(is_list_of(batch_inputs, torch.Tensor))
        self.assertIs(input1, batch_inputs[0])
        self.assertIs(input2, batch_inputs[1])

        # check data_sample
        self.assertIs(batch_data_sample[0], data_sample1)
        self.assertIs(batch_data_sample[1], data_sample2)

        # Test with list of tuple, each tuple is a nested dict instance
        data_batch = [(dict(
            inputs=input1,
            data_sample=data_sample1,
            value=1,
            name='1',
            nested=dict(data_sample=data_sample1)),
                       dict(
                           inputs=input2,
                           data_sample=data_sample2,
                           value=2,
                           name='2',
                           nested=dict(data_sample=data_sample2))),
                      (dict(
                          inputs=input1,
                          data_sample=data_sample1,
                          value=1,
                          name='1',
                          nested=dict(data_sample=data_sample1)),
                       dict(
                           inputs=input2,
                           data_sample=data_sample2,
                           value=2,
                           name='2',
                           nested=dict(data_sample=data_sample2)))]
        data_batch = pseudo_collate(data_batch)
        batch_inputs_0 = data_batch[0]['inputs']
        batch_inputs_1 = data_batch[1]['inputs']
        batch_data_sample_0 = data_batch[0]['data_sample']
        batch_data_sample_1 = data_batch[1]['data_sample']
        batch_value_0 = data_batch[0]['value']
        batch_value_1 = data_batch[1]['value']
        batch_name_0 = data_batch[0]['name']
        batch_name_1 = data_batch[1]['name']
        batch_nested_0 = data_batch[0]['nested']
        batch_nested_1 = data_batch[1]['nested']

        self.assertTrue(is_list_of(batch_inputs_0, torch.Tensor))
        self.assertTrue(is_list_of(batch_inputs_1, torch.Tensor))
        self.assertIs(batch_inputs_0[0], input1)
        self.assertIs(batch_inputs_0[1], input1)
        self.assertIs(batch_inputs_1[0], input2)
        self.assertIs(batch_inputs_1[1], input2)

        self.assertIs(batch_data_sample_0[0], data_sample1)
        self.assertIs(batch_data_sample_0[1], data_sample1)
        self.assertIs(batch_data_sample_1[0], data_sample2)
        self.assertIs(batch_data_sample_1[1], data_sample2)

        self.assertEqual(batch_value_0, [1, 1])
        self.assertEqual(batch_value_1, [2, 2])

        self.assertEqual(batch_name_0, ['1', '1'])
        self.assertEqual(batch_name_1, ['2', '2'])

        self.assertIs(batch_nested_0['data_sample'][0], data_sample1)
        self.assertIs(batch_nested_0['data_sample'][1], data_sample1)
        self.assertIs(batch_nested_1['data_sample'][0], data_sample2)
        self.assertIs(batch_nested_1['data_sample'][1], data_sample2)

    def test_default_collate(self):
        # `default_collate` has comment logic with `pseudo_collate`, therefore
        # only test it cam stack batch tensor, convert int or float to tensor.
        input1 = torch.randn(1, 3, 5)
        input2 = torch.randn(1, 3, 5)
        data_batch = [(
            dict(inputs=input1, value=1, array=np.array(1)),
            dict(inputs=input2, value=2, array=np.array(2)),
        ),
                      (
                          dict(inputs=input1, value=1, array=np.array(1)),
                          dict(inputs=input2, value=2, array=np.array(2)),
                      )]
        data_batch = default_collate(data_batch)
        batch_inputs_0 = data_batch[0]['inputs']
        batch_inputs_1 = data_batch[1]['inputs']
        batch_value_0 = data_batch[0]['value']
        batch_value_1 = data_batch[1]['value']
        batch_array_0 = data_batch[0]['array']
        batch_array_1 = data_batch[1]['array']

        self.assertEqual(tuple(batch_inputs_0.shape), (2, 1, 3, 5))
        self.assertEqual(tuple(batch_inputs_1.shape), (2, 1, 3, 5))

        self.assertTrue(
            torch.allclose(batch_inputs_0, torch.stack([input1, input1])))
        self.assertTrue(
            torch.allclose(batch_inputs_1, torch.stack([input2, input2])))

        target1 = torch.stack([torch.tensor(1), torch.tensor(1)])
        target2 = torch.stack([torch.tensor(2), torch.tensor(2)])

        self.assertTrue(
            torch.allclose(batch_value_0.to(target1.dtype), target1))
        self.assertTrue(
            torch.allclose(batch_value_1.to(target2.dtype), target2))

        self.assertTrue(
            torch.allclose(batch_array_0.to(target1.dtype), target1))
        self.assertTrue(
            torch.allclose(batch_array_1.to(target2.dtype), target2))
