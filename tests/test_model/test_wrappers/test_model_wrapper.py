from unittest import TestCase
from unittest.mock import patch, MagicMock
import pytest

from mmengine.model import ModelWrapper, BaseModel



class TestModelWrapper(TestCase):
    @pytest.mark.skipif(pytest.mark.skipif, reason='requires CUDA support')
    def test_init(self):
        model_wrapper = ModelWrapper(BaseModel())
        self.assertEqual(model_wrapper.device.type, "cuda")

    @pytest.mark.skipif(pytest.mark.skipif, reason='requires CUDA support')
    def test_forward(self):
        # Test `train_step`
        base_model = MagicMock()
        model_wrapper = ModelWrapper(base_model)
        data = MagicMock()
        model_wrapper(data, 'train')
        base_model.train_step.assert_called_with(data, 'train')
        base_model.val_step.assert_not_called()
        base_model.test_step.assert_not_called()
        # Test `val_step`
        model_wrapper(data, 'val')
        base_model.assert_called_with(data, 'val', return_loss=False)
        base_model.test_step.assert_not_called()
        # Test `test_step`
        model_wrapper(data, 'test')
        base_model.test_step.assert_called_with(data)


