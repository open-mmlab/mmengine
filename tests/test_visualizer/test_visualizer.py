# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Any, List, Optional, Union
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn as nn

from mmengine import VISBACKENDS
from mmengine.visualization import Visualizer


@VISBACKENDS.register_module()
class MockVisBackend:

    def __init__(self, save_dir: Optional[str] = None):
        self._save_dir = save_dir
        self._close = False

    @property
    def experiment(self) -> Any:
        return self

    def add_config(self, params_dict: dict, **kwargs) -> None:
        self._add_config = True

    def add_graph(self, model: torch.nn.Module,
                  input_tensor: Union[torch.Tensor,
                                      List[torch.Tensor]], **kwargs) -> None:

        self._add_graph = True

    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        self._add_image = True

    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        self._add_scalar = True

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        self._add_scalars = True

    def close(self) -> None:
        """close an opened object."""
        self._close = True


class TestVisualizer(TestCase):

    def setUp(self):
        """Setup the demo image in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.image = np.random.randint(
            0, 256, size=(10, 10, 3)).astype('uint8')
        self.vis_backend_cfg = [
            dict(type='MockVisBackend', name='mock1', save_dir='tmp'),
            dict(type='MockVisBackend', name='mock2', save_dir='tmp')
        ]

    def test_init(self):
        visualizer = Visualizer(image=self.image)
        visualizer.get_image()

        visualizer = Visualizer(
            vis_backends=copy.deepcopy(self.vis_backend_cfg))
        assert isinstance(visualizer.get_backend('mock1'), MockVisBackend)
        assert len(visualizer._vis_backends) == 2

        # test global
        visualizer = Visualizer.get_instance(
            'visualizer', vis_backends=copy.deepcopy(self.vis_backend_cfg))
        assert len(visualizer._vis_backends) == 2
        visualizer_any = Visualizer.get_instance('visualizer')
        assert visualizer_any == visualizer

    def test_set_image(self):
        visualizer = Visualizer()
        visualizer.set_image(self.image)
        with pytest.raises(AssertionError):
            visualizer.set_image(None)

    def test_get_image(self):
        visualizer = Visualizer(image=self.image)
        visualizer.get_image()

    def test_draw_bboxes(self):
        visualizer = Visualizer(image=self.image)

        # only support 4 or nx4 tensor and numpy
        visualizer.draw_bboxes(torch.tensor([1, 1, 2, 2]))
        # valid bbox
        visualizer.draw_bboxes(torch.tensor([1, 1, 1, 2]))
        bboxes = torch.tensor([[1, 1, 2, 2], [1, 2, 2, 2.5]])
        visualizer.draw_bboxes(
            bboxes, alpha=0.5, edge_colors=(255, 0, 0), line_styles='-')
        bboxes = bboxes.numpy()
        visualizer.draw_bboxes(bboxes)

        # test invalid bbox
        with pytest.raises(AssertionError):
            # x1 > x2
            visualizer.draw_bboxes(torch.tensor([5, 1, 2, 2]))

        # test out of bounds
        with pytest.warns(
                UserWarning,
                match='Warning: The bbox is out of bounds,'
                ' the drawn bbox may not be in the image'):
            visualizer.draw_bboxes(torch.tensor([1, 1, 20, 2]))

        # test incorrect bbox format
        with pytest.raises(TypeError):
            visualizer.draw_bboxes([1, 1, 2, 2])

    def test_close(self):
        visualizer = Visualizer(
            image=self.image, vis_backends=copy.deepcopy(self.vis_backend_cfg))
        fig_num = visualizer.fig_save_num
        assert fig_num in plt.get_fignums()
        for name in ['mock1', 'mock2']:
            assert visualizer.get_backend(name)._close is False
        visualizer.close()
        assert fig_num not in plt.get_fignums()
        for name in ['mock1', 'mock2']:
            assert visualizer.get_backend(name)._close is True

    def test_draw_texts(self):
        visualizer = Visualizer(image=self.image)

        # only support tensor and numpy
        visualizer.draw_texts(
            'text1', positions=torch.tensor([5, 5]), colors=(0, 255, 0))
        visualizer.draw_texts(['text1', 'text2'],
                              positions=torch.tensor([[5, 5], [3, 3]]),
                              colors=[(255, 0, 0), (255, 0, 0)])
        visualizer.draw_texts('text1', positions=np.array([5, 5]))
        visualizer.draw_texts(['text1', 'text2'],
                              positions=np.array([[5, 5], [3, 3]]))
        visualizer.draw_texts(
            'text1',
            positions=torch.tensor([5, 5]),
            bboxes=dict(facecolor='r', alpha=0.6))
        # test out of bounds
        with pytest.warns(
                UserWarning,
                match='Warning: The text is out of bounds,'
                ' the drawn text may not be in the image'):
            visualizer.draw_texts('text1', positions=torch.tensor([15, 5]))

        # test incorrect format
        with pytest.raises(TypeError):
            visualizer.draw_texts('text', positions=[5, 5])

        # test length mismatch
        with pytest.raises(AssertionError):
            visualizer.draw_texts(['text1', 'text2'],
                                  positions=torch.tensor([5, 5]))
        with pytest.raises(AssertionError):
            visualizer.draw_texts(
                'text1', positions=torch.tensor([[5, 5], [3, 3]]))
        with pytest.raises(AssertionError):
            visualizer.draw_texts(['text1', 'test2'],
                                  positions=torch.tensor([[5, 5], [3, 3]]),
                                  colors=['r'])
        with pytest.raises(AssertionError):
            visualizer.draw_texts(['text1', 'test2'],
                                  positions=torch.tensor([[5, 5], [3, 3]]),
                                  vertical_alignments=['top'])
        with pytest.raises(AssertionError):
            visualizer.draw_texts(['text1', 'test2'],
                                  positions=torch.tensor([[5, 5], [3, 3]]),
                                  horizontal_alignments=['left'])
        with pytest.raises(AssertionError):
            visualizer.draw_texts(['text1', 'test2'],
                                  positions=torch.tensor([[5, 5], [3, 3]]),
                                  font_sizes=[1])

        # test type valid
        with pytest.raises(TypeError):
            visualizer.draw_texts(['text1', 'test2'],
                                  positions=torch.tensor([[5, 5], [3, 3]]),
                                  font_sizes='b')

    def test_draw_lines(self):
        visualizer = Visualizer(image=self.image)

        # only support tensor and numpy
        visualizer.draw_lines(
            x_datas=torch.tensor([1, 5]), y_datas=torch.tensor([2, 6]))
        visualizer.draw_lines(
            x_datas=np.array([[1, 5], [2, 4]]),
            y_datas=np.array([[2, 6], [4, 7]]))
        visualizer.draw_lines(
            x_datas=np.array([[1, 5], [2, 4]]),
            y_datas=np.array([[2, 6], [4, 7]]),
            colors='r',
            line_styles=['-', '-.'],
            line_widths=[1, 2])
        # test out of bounds
        with pytest.warns(
                UserWarning,
                match='Warning: The line is out of bounds,'
                ' the drawn line may not be in the image'):
            visualizer.draw_lines(
                x_datas=torch.tensor([12, 5]), y_datas=torch.tensor([2, 6]))

        # test incorrect format
        with pytest.raises(TypeError):
            visualizer.draw_lines(x_datas=[5, 5], y_datas=torch.tensor([2, 6]))
        with pytest.raises(TypeError):
            visualizer.draw_lines(y_datas=[5, 5], x_datas=torch.tensor([2, 6]))

        # test length mismatch
        with pytest.raises(AssertionError):
            visualizer.draw_lines(
                x_datas=torch.tensor([1, 5]),
                y_datas=torch.tensor([[2, 6], [4, 7]]))

    def test_draw_circles(self):
        visualizer = Visualizer(image=self.image)

        # only support tensor and numpy
        visualizer.draw_circles(torch.tensor([1, 5]), torch.tensor([1]))
        visualizer.draw_circles(np.array([1, 5]), np.array([1]))
        visualizer.draw_circles(
            torch.tensor([[1, 5], [2, 6]]), radius=torch.tensor([1, 2]))

        # test face_colors
        visualizer.draw_circles(
            torch.tensor([[1, 5], [2, 6]]),
            radius=torch.tensor([1, 2]),
            face_colors=(255, 0, 0),
            edge_colors=(255, 0, 0))

        # test config
        visualizer.draw_circles(
            torch.tensor([[1, 5], [2, 6]]),
            radius=torch.tensor([1, 2]),
            edge_colors=['g', 'r'],
            line_styles=['-', '-.'],
            line_widths=[1, 2])

        # test out of bounds
        with pytest.warns(
                UserWarning,
                match='Warning: The circle is out of bounds,'
                ' the drawn circle may not be in the image'):
            visualizer.draw_circles(
                torch.tensor([12, 5]), radius=torch.tensor([1]))
            visualizer.draw_circles(
                torch.tensor([1, 5]), radius=torch.tensor([10]))

        # test incorrect format
        with pytest.raises(TypeError):
            visualizer.draw_circles([1, 5], radius=torch.tensor([1]))
        with pytest.raises(TypeError):
            visualizer.draw_circles(np.array([1, 5]), radius=10)

        # test length mismatch
        with pytest.raises(AssertionError):
            visualizer.draw_circles(
                torch.tensor([[1, 5]]), radius=torch.tensor([1, 2]))

    def test_draw_polygons(self):
        visualizer = Visualizer(image=self.image)
        # shape Nx2 or list[Nx2]
        visualizer.draw_polygons(torch.tensor([[1, 1], [2, 2], [3, 4]]))
        visualizer.draw_polygons(np.array([[1, 1], [2, 2], [3, 4]]))
        visualizer.draw_polygons([
            np.array([[1, 1], [2, 2], [3, 4]]),
            torch.tensor([[1, 1], [2, 2], [3, 4]])
        ])
        visualizer.draw_polygons(
            polygons=[
                np.array([[1, 1], [2, 2], [3, 4]]),
                torch.tensor([[1, 1], [2, 2], [3, 4]])
            ],
            face_colors=(255, 0, 0),
            edge_colors=(255, 0, 0))
        visualizer.draw_polygons(
            polygons=[
                np.array([[1, 1], [2, 2], [3, 4]]),
                torch.tensor([[1, 1], [2, 2], [3, 4]])
            ],
            edge_colors=['r', 'g'],
            line_styles='-',
            line_widths=[2, 1])

        # test out of bounds
        with pytest.warns(
                UserWarning,
                match='Warning: The polygon is out of bounds,'
                ' the drawn polygon may not be in the image'):
            visualizer.draw_polygons(torch.tensor([[1, 1], [2, 2], [16, 4]]))

    def test_draw_binary_masks(self):
        binary_mask = np.random.randint(0, 2, size=(10, 10)).astype(np.bool)
        visualizer = Visualizer(image=self.image)
        visualizer.draw_binary_masks(binary_mask)
        visualizer.draw_binary_masks(torch.from_numpy(binary_mask))
        # multi binary
        binary_mask = np.random.randint(0, 2, size=(2, 10, 10)).astype(np.bool)
        visualizer = Visualizer(image=self.image)
        visualizer.draw_binary_masks(binary_mask, colors=['r', (0, 255, 0)])
        # test the error that the size of mask and image are different.
        with pytest.raises(AssertionError):
            binary_mask = np.random.randint(0, 2, size=(8, 10)).astype(np.bool)
            visualizer.draw_binary_masks(binary_mask)

        # test non binary mask error
        binary_mask = np.random.randint(0, 2, size=(10, 10, 3)).astype(np.bool)
        with pytest.raises(AssertionError):
            visualizer.draw_binary_masks(binary_mask)

        # test color dim
        with pytest.raises(AssertionError):
            visualizer.draw_binary_masks(
                binary_mask, colors=np.array([1, 22, 4, 45]))
        binary_mask = np.random.randint(0, 2, size=(10, 10))
        with pytest.raises(AssertionError):
            visualizer.draw_binary_masks(binary_mask)

    def test_draw_featmap(self):
        visualizer = Visualizer()
        image = np.random.randint(0, 256, size=(3, 3, 3), dtype='uint8')
        # test tensor format
        with pytest.raises(AssertionError, match='Input dimension must be 3'):
            visualizer.draw_featmap(torch.randn(1, 1, 3, 3))

        # test mode parameter
        # mode only supports 'mean' and 'max'
        with pytest.raises(AssertionError):
            visualizer.draw_featmap(torch.randn(2, 3, 3), mode='xx')
        # test tensor_chw and img have difference height and width
        with pytest.raises(AssertionError):
            visualizer.draw_featmap(torch.randn(2, 3, 3), mode='xx')

        # test topk parameter
        with pytest.raises(
                AssertionError,
                match='The input tensor channel dimension must be 1 or 3 '
                'when topk is less than 1, but the channel '
                'dimension you input is 6, you can use the '
                'mode parameter or set topk greater than 0 to solve '
                'the error'):
            visualizer.draw_featmap(torch.randn(6, 3, 3), mode=None, topk=0)

        visualizer.draw_featmap(torch.randn(6, 3, 3), mode='mean')
        visualizer.draw_featmap(torch.randn(1, 3, 3), mode='mean')
        visualizer.draw_featmap(torch.randn(6, 3, 3), mode='max')
        visualizer.draw_featmap(torch.randn(6, 3, 3), mode='max', topk=10)
        visualizer.draw_featmap(torch.randn(1, 3, 3), mode=None, topk=-1)
        visualizer.draw_featmap(
            torch.randn(3, 3, 3), image=image, mode=None, topk=-1)
        visualizer.draw_featmap(torch.randn(6, 3, 3), mode=None, topk=4)
        visualizer.draw_featmap(
            torch.randn(6, 3, 3), image=image, mode=None, topk=8)

        # test gray
        visualizer.draw_featmap(
            torch.randn(6, 3, 3),
            image=np.random.randint(0, 256, size=(3, 3), dtype='uint8'),
            mode=None,
            topk=8)

        # test arrangement
        with pytest.raises(AssertionError):
            visualizer.draw_featmap(
                torch.randn(10, 3, 3),
                image=image,
                mode=None,
                topk=8,
                arrangement=(2, 2))

    def test_chain_call(self):
        visualizer = Visualizer(image=self.image)
        binary_mask = np.random.randint(0, 2, size=(10, 10)).astype(np.bool)
        visualizer.draw_bboxes(torch.tensor([1, 1, 2, 2])). \
            draw_texts('test', torch.tensor([5, 5])). \
            draw_lines(x_datas=torch.tensor([1, 5]),
                       y_datas=torch.tensor([2, 6])). \
            draw_circles(torch.tensor([1, 5]), radius=torch.tensor([2])). \
            draw_polygons(torch.tensor([[1, 1], [2, 2], [3, 4]])). \
            draw_binary_masks(binary_mask)

    def test_get_backend(self):
        visualizer = Visualizer(
            image=self.image, vis_backends=copy.deepcopy(self.vis_backend_cfg))
        for name in ['mock1', 'mock2']:
            assert isinstance(visualizer.get_backend(name), MockVisBackend)

    def test_add_config(self):
        visualizer = Visualizer(
            vis_backends=copy.deepcopy(self.vis_backend_cfg))

        params_dict = dict(lr=0.1, wd=0.2, mode='linear')
        visualizer.add_config(params_dict)
        for name in ['mock1', 'mock2']:
            assert visualizer.get_backend(name)._add_config is True

    def test_add_graph(self):
        visualizer = Visualizer(
            vis_backends=copy.deepcopy(self.vis_backend_cfg))

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 2, 1)

            def forward(self, x, y=None):
                return self.conv(x)

        visualizer.add_graph(Model(), np.zeros([1, 1, 3, 3]))
        for name in ['mock1', 'mock2']:
            assert visualizer.get_backend(name)._add_graph is True

    def test_add_image(self):
        image = np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)
        visualizer = Visualizer(
            vis_backends=copy.deepcopy(self.vis_backend_cfg))

        visualizer.add_image('img', image)
        for name in ['mock1', 'mock2']:
            assert visualizer.get_backend(name)._add_image is True

    def test_add_scalar(self):
        visualizer = Visualizer(
            vis_backends=copy.deepcopy(self.vis_backend_cfg))
        visualizer.add_scalar('map', 0.9, step=0)
        for name in ['mock1', 'mock2']:
            assert visualizer.get_backend(name)._add_scalar is True

    def test_add_scalars(self):
        visualizer = Visualizer(
            vis_backends=copy.deepcopy(self.vis_backend_cfg))
        input_dict = {'map': 0.7, 'acc': 0.9}
        visualizer.add_scalars(input_dict)
        for name in ['mock1', 'mock2']:
            assert visualizer.get_backend(name)._add_scalars is True


if __name__ == '__main__':
    t = TestVisualizer()
    t.setUp()
    t.test_init()
