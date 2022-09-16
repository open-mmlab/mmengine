# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from torch.utils.data import DataLoader

from mmengine.model import BaseModel, BaseTestTimeAugModel


class ToyTestTimeAugModel(BaseTestTimeAugModel):

    def merge_results(self, data_samples_list):
        result = list(map(lambda x: sum(x), zip(*data_samples_list)))
        return result


class ToyModel(BaseModel):

    def forward(self, inputs, data_samples, mode='tensor'):
        return data_samples


class TestBaseTestTimeAugModel(TestCase):

    def setUp(self) -> None:
        dict_dataset = [dict(inputs=[1, 2], data_samples=[3, 4])] * 10
        tuple_dataset = [([1, 2], [3, 4])] * 10
        self.model = ToyModel()
        self.dict_dataloader = DataLoader(dict_dataset, batch_size=2)
        self.tuple_dataloader = DataLoader(tuple_dataset, batch_size=2)

    def test_init(self):
        tta_model = ToyTestTimeAugModel(self.model)
        self.assertIs(tta_model.module, self.model)

    def test_test_step(self):
        tta_model = ToyTestTimeAugModel(self.model)

        # Test dict dataset

        for data in self.dict_dataloader:
            # Test step will call forward.
            result = tta_model.test_step(data)
            self.assertEqual(result, [7, 7])

        for data in self.tuple_dataloader:
            result = tta_model.test_step(data)
            self.assertEqual(result, [7, 7])

    def test_train_step(self):
        tta_model = ToyTestTimeAugModel(self.model)
        with self.assertRaisesRegex(NotImplementedError,
                                    'train_step should not be called'):
            tta_model.train_step(None, None)
