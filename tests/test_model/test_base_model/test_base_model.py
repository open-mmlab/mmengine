from unittest import TestCase
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from mmengine.model import BaseModel
from mmengine.testing import assert_allclose
from mmengine import InstanceData


class TestBaseModel(TestCase):
    def test_init(self):
        # initiate model without `preprocess_cfg`
        model = BaseModel()
        self.assertFalse(model.to_rgb, False)
        self.assertFalse(model._fp16_enabled, False)
        assert_allclose(model.pixel_mean,
                        torch.tensor([127.5, 127.5, 127.5]).view(-1, 1, 1))
        assert_allclose(model.pixel_std,
                        torch.tensor([127.5, 127.5, 127.5]).view(-1, 1, 1))
        self.assertEqual(model.size_divisor, 1)
        self.assertDictEqual(model._feats_dict, dict())
        # initiate model with preprocess_cfg` and feat keys
        preprocess_cfg = dict(to_rgb=True, pixel_mean=[0, 0, 0], pixel_std=[
            255, 255, 255], size_divisor=16)
        model = BaseModel(preprocess_cfg, feat_keys=('backbone', ))
        self.assertTrue(model.to_rgb, True)
        self.assertFalse(model._fp16_enabled, False)
        assert_allclose(model.pixel_mean,
                        torch.tensor([0, 0, 0]).view(-1, 1, 1))
        assert_allclose(model.pixel_std,
                        torch.tensor([255, 255, 255]).view(-1, 1, 1))
        self.assertEqual(model.size_divisor, 16)
        self.assertDictEqual(model._feats_dict, dict(backbone=None))

    def test_preprocess(self):
        # Test with torch.Tensor data
        preprocess_cfg = dict(to_rgb=True)
        model = BaseModel(preprocess_cfg)
        data = torch.randn(3, 10, 10)
        target_inputs = data.clone()
        outputs = model.preprocess(data)[0][:, [2, 1, 0], ...]
        target_inputs = ((target_inputs - 127.5) / 127.5)
        target_inputs = target_inputs.unsqueeze(0)
        assert_allclose(outputs, target_inputs)

        # Test with list of dict data
        inputs1 = torch.randn(3, 10, 10)
        inputs2 = torch.randn(3, 10, 10)
        data_sample1 = InstanceData(bboxes=torch.randn(5, 4))
        data_sample2 = InstanceData(bboxes=torch.randn(5, 4))
        data = [dict(inputs=inputs1.clone(), data_sample=data_sample1.clone()),
                dict(inputs=inputs2.clone(), data_sample=data_sample2.clone())]

        inputs, data_samples = model.preprocess(data)
        target_inputs = [((inputs1 - 127.5) / 127.5)[[2, 1, 0], ...],
                         ((inputs2 - 127.5) / 127.5)[[2, 1, 0], ...]]
        target_data_samples = [data_sample1, data_sample2]

        for input_, data_sample, target_input, target_data_sample in zip(
                inputs, data_samples, target_inputs, target_data_samples):
            assert_allclose(input_, target_input)
            assert_allclose(data_sample.bboxes, target_data_sample.bboxes)

    def test_parse_losses(self):
        model = BaseModel()
        loss_cls = torch.tensor(1, dtype=torch.float32)
        loss_list = [torch.tensor(2, dtype=torch.float32),
                     torch.tensor(3, dtype=torch.float32)]
        losses = dict(loss_cls=loss_cls, loss_list=loss_list)
        target_parsed_losses = torch.tensor(6, dtype=torch.float32)
        targe_log_vars = dict(
            loss=torch.tensor(6, dtype=torch.float32),
            loss_cls=torch.tensor(1, dtype=torch.float32),
            loss_list=torch.tensor(5, dtype=torch.float32))
        parse_losses, log_vars = model._parse_losses(losses)
        assert_allclose(parse_losses, target_parsed_losses)
        for key in log_vars:
            self.assertIn(key, targe_log_vars)
            assert_allclose(log_vars[key], targe_log_vars[key])

    def test_train_step(self):
        class ToyModel(BaseModel):
            def forward(self, *args, **kwargs):
                return dict(loss_cls=torch.tensor(1, dtype=torch.float32))
        optimizer_wrapper = MagicMock()
        data = MagicMock()
        toy_model = ToyModel()
        toy_model.preprocess = MagicMock(return_value=(None, None))

        log_vars = toy_model.train_step(data, optimizer_wrapper)
        optimizer_wrapper.optimizer_step.assert_called_with(
            torch.tensor(1, dtype=torch.float32))
        assert_allclose(
            log_vars['loss_cls'], torch.tensor(1, dtype=torch.float32))
        assert_allclose(
            log_vars['loss'], torch.tensor(1, dtype=torch.float32))

    def test_val_step(self):
        model = BaseModel()
        model.forward = MagicMock()
        model.preprocess = MagicMock(return_value=(None, None))
        data = MagicMock()
        model.val_step(data, return_loss=True)
        model.forward.assert_called_with(None, None, return_loss=True)

    def test_test_step(self):
        model = BaseModel()
        model.forward = MagicMock()
        model.preprocess = MagicMock(return_value=(None, None))
        data = MagicMock()
        model.test_step(data)
        model.forward.assert_called_with(None, None, return_loss=False)

    def test_register_get_feature_hook(self):
        class ToyModel(BaseModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.backbone = nn.ReLU()
                self.register_get_feature_hook()

            def forward(self, inputs, data_samples=None, return_loss=False):
                return self.backbone(inputs)

        toy_model = ToyModel(feat_keys=('backbone', ))
        toy_model(torch.tensor(1))
        assert_allclose(toy_model.feats_dict['backbone'], torch.tensor(1))
