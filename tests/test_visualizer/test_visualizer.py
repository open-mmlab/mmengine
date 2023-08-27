# Copyright (c) OpenMMLab. All rights reserved.
import copy
import time
from typing import Any
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from mmengine import VISBACKENDS, Config
from mmengine.visualization import Visualizer


@VISBACKENDS.register_module()
class MockVisBackend:

    def __init__(self, save_dir: str = 'none'):
        self._save_dir = save_dir
        self._close = False

    @property
    def experiment(self) -> Any:
        return self

    def add_config(self, config, **kwargs) -> None:
        self._add_config = True

    def add_graph(self, model, data_batch, **kwargs) -> None:
        self._add_graph = True

    def add_image(self, name, image, step=0, **kwargs) -> None:
        self._add_image = True

    def add_scalar(self, name, value, step=0, **kwargs) -> None:
        self._add_scalar = True

    def add_scalars(self,
                    scalar_dict,
                    step=0,
                    file_path=None,
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
            dict(type='MockVisBackend', name='mock1'),
            dict(type='MockVisBackend', name='mock2')
        ]

    def test_init(self):
        visualizer = Visualizer(image=self.image)
        visualizer.get_image()

        # build visualizer without `save_dir`
        visualizer = Visualizer(
            vis_backends=copy.deepcopy(self.vis_backend_cfg))

        visualizer = Visualizer(
            vis_backends=copy.deepcopy(self.vis_backend_cfg),
            save_dir='temp_dir')
        assert isinstance(visualizer.get_backend('mock1'), MockVisBackend)
        assert len(visualizer._vis_backends) == 2

        # The name fields cannot be the same
        with pytest.raises(RuntimeError):
            Visualizer(
                vis_backends=[
                    dict(type='MockVisBackend'),
                    dict(type='MockVisBackend')
                ],
                save_dir='temp_dir')

        with pytest.raises(RuntimeError):
            Visualizer(
                vis_backends=[
                    dict(type='MockVisBackend', name='mock1'),
                    dict(type='MockVisBackend', name='mock1')
                ],
                save_dir='temp_dir')

        # test global init
        instance_name = 'visualizer' + str(time.time())
        visualizer = Visualizer.get_instance(
            instance_name,
            vis_backends=copy.deepcopy(self.vis_backend_cfg),
            save_dir='temp_dir')
        assert len(visualizer._vis_backends) == 2
        visualizer_any = Visualizer.get_instance(instance_name)
        assert visualizer_any == visualizer

        # local backend will not be built without `save_dir` argument
        @VISBACKENDS.register_module()
        class CustomLocalVisBackend:

            def __init__(self, save_dir: str) -> None:
                self._save_dir = save_dir

        with pytest.warns(UserWarning):
            visualizer = Visualizer.get_instance(
                'test_save_dir',
                vis_backends=[dict(type='CustomLocalVisBackend')])
            assert not visualizer._vis_backends

        VISBACKENDS.module_dict.pop('CustomLocalVisBackend')

        visualizer = Visualizer.get_instance(
            'test_save_dir',
            vis_backends=dict(type='CustomLocalVisBackend', save_dir='tmp'))

        visualizer = Visualizer.get_instance(
            'test_save_dir', vis_backends=[CustomLocalVisBackend('tmp')])

        visualizer = Visualizer.get_instance(
            'test_save_dir', vis_backends=CustomLocalVisBackend('tmp'))

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
            image=self.image,
            vis_backends=copy.deepcopy(self.vis_backend_cfg),
            save_dir='temp_dir')

        for name in ['mock1', 'mock2']:
            assert visualizer.get_backend(name)._close is False
        visualizer.close()
        for name in ['mock1', 'mock2']:
            assert visualizer.get_backend(name)._close is True

    def test_draw_points(self):
        visualizer = Visualizer(image=self.image)

        with pytest.raises(TypeError):
            visualizer.draw_points(positions=[1, 2])
        with pytest.raises(AssertionError):
            visualizer.draw_points(positions=np.array([1, 2, 3], dtype=object))
        # test color
        visualizer.draw_points(
            positions=torch.tensor([[1, 1], [3, 3]]),
            colors=['g', (255, 255, 0)])
        visualizer.draw_points(
            positions=torch.tensor([[1, 1], [3, 3]]),
            colors=['g', (255, 255, 0)],
            marker='.',
            sizes=[1, 5])

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
        binary_mask = np.random.randint(0, 2, size=(10, 10)).astype(bool)
        visualizer = Visualizer(image=self.image)
        visualizer.draw_binary_masks(binary_mask)
        visualizer.draw_binary_masks(torch.from_numpy(binary_mask))
        # multi binary
        binary_mask = np.random.randint(0, 2, size=(2, 10, 10)).astype(bool)
        visualizer = Visualizer(image=self.image)
        visualizer.draw_binary_masks(binary_mask, colors=['r', (0, 255, 0)])
        # test the error that the size of mask and image are different.
        with pytest.raises(AssertionError):
            binary_mask = np.random.randint(0, 2, size=(8, 10)).astype(bool)
            visualizer.draw_binary_masks(binary_mask)

        # test non binary mask error
        binary_mask = np.random.randint(0, 2, size=(10, 10, 3)).astype(bool)
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

        # must be Tensor
        with pytest.raises(
                AssertionError,
                match='`featmap` should be torch.Tensor, but got '
                "<class 'numpy.ndarray'>"):
            visualizer.draw_featmap(np.ones((3, 3, 3)))

        # test tensor format
        with pytest.raises(
                AssertionError, match='Input dimension must be 3, but got 4'):
            visualizer.draw_featmap(torch.randn(1, 1, 3, 3))

        # test overlaid_image shape
        with pytest.warns(Warning):
            visualizer.draw_featmap(torch.randn(1, 4, 3), overlaid_image=image)

        # test resize_shape
        featmap = visualizer.draw_featmap(
            torch.randn(1, 4, 3), resize_shape=(6, 7))
        assert featmap.shape[:2] == (6, 7)
        featmap = visualizer.draw_featmap(
            torch.randn(1, 4, 3), overlaid_image=image, resize_shape=(6, 7))
        assert featmap.shape[:2] == (6, 7)

        # test channel_reduction parameter
        # mode only supports 'squeeze_mean' and 'select_max'
        with pytest.raises(AssertionError):
            visualizer.draw_featmap(
                torch.randn(2, 3, 3), channel_reduction='xx')

        featmap = visualizer.draw_featmap(
            torch.randn(2, 3, 3), channel_reduction='squeeze_mean')
        assert featmap.shape[:2] == (3, 3)
        featmap = visualizer.draw_featmap(
            torch.randn(2, 3, 3), channel_reduction='select_max')
        assert featmap.shape[:2] == (3, 3)
        featmap = visualizer.draw_featmap(
            torch.randn(2, 4, 3),
            overlaid_image=image,
            channel_reduction='select_max')
        assert featmap.shape[:2] == (3, 3)

        # test topk parameter
        with pytest.raises(
                AssertionError,
                match='The input tensor channel dimension must be 1 or 3 '
                'when topk is less than 1, but the channel '
                'dimension you input is 6, you can use the '
                'channel_reduction parameter or set topk '
                'greater than 0 to solve the error'):
            visualizer.draw_featmap(
                torch.randn(6, 3, 3), channel_reduction=None, topk=0)

        featmap = visualizer.draw_featmap(
            torch.randn(6, 3, 3), channel_reduction='select_max', topk=10)
        assert featmap.shape[:2] == (3, 3)
        featmap = visualizer.draw_featmap(
            torch.randn(1, 4, 3), channel_reduction=None, topk=-1)
        assert featmap.shape[:2] == (4, 3)

        featmap = visualizer.draw_featmap(
            torch.randn(3, 4, 3),
            overlaid_image=image,
            channel_reduction=None,
            topk=-1)
        assert featmap.shape[:2] == (3, 3)
        featmap = visualizer.draw_featmap(
            torch.randn(6, 3, 3),
            channel_reduction=None,
            topk=4,
            arrangement=(2, 2))
        assert featmap.shape[:2] == (6, 6)
        featmap = visualizer.draw_featmap(
            torch.randn(6, 3, 3),
            channel_reduction=None,
            topk=4,
            arrangement=(1, 4))
        assert featmap.shape[:2] == (3, 12)
        with pytest.raises(
                AssertionError,
                match='The product of row and col in the `arrangement` '
                'is less than topk, please set '
                'the `arrangement` correctly'):
            visualizer.draw_featmap(
                torch.randn(6, 3, 3),
                channel_reduction=None,
                topk=4,
                arrangement=(1, 2))

        # test gray
        featmap = visualizer.draw_featmap(
            torch.randn(6, 3, 3),
            overlaid_image=np.random.randint(
                0, 256, size=(3, 3), dtype='uint8'),
            channel_reduction=None,
            topk=4,
            arrangement=(2, 2))
        assert featmap.shape[:2] == (6, 6)

    def test_chain_call(self):
        visualizer = Visualizer(image=self.image)
        binary_mask = np.random.randint(0, 2, size=(10, 10)).astype(bool)
        visualizer.draw_bboxes(torch.tensor([1, 1, 2, 2])). \
            draw_texts('test', torch.tensor([5, 5])). \
            draw_lines(x_datas=torch.tensor([1, 5]),
                       y_datas=torch.tensor([2, 6])). \
            draw_circles(torch.tensor([1, 5]), radius=torch.tensor([2])). \
            draw_polygons(torch.tensor([[1, 1], [2, 2], [3, 4]])). \
            draw_binary_masks(binary_mask)

    def test_get_backend(self):
        visualizer = Visualizer(
            image=self.image,
            vis_backends=copy.deepcopy(self.vis_backend_cfg),
            save_dir='temp_dir')
        for name in ['mock1', 'mock2']:
            assert isinstance(visualizer.get_backend(name), MockVisBackend)

    def test_add_config(self):
        visualizer = Visualizer(
            vis_backends=copy.deepcopy(self.vis_backend_cfg),
            save_dir='temp_dir')

        cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        visualizer.add_config(cfg)
        for name in ['mock1', 'mock2']:
            assert visualizer.get_backend(name)._add_config is True

    def test_add_graph(self):
        visualizer = Visualizer(
            vis_backends=copy.deepcopy(self.vis_backend_cfg),
            save_dir='temp_dir')

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
            vis_backends=copy.deepcopy(self.vis_backend_cfg),
            save_dir='temp_dir')

        visualizer.add_image('img', image)
        for name in ['mock1', 'mock2']:
            assert visualizer.get_backend(name)._add_image is True

    def test_add_scalar(self):
        visualizer = Visualizer(
            vis_backends=copy.deepcopy(self.vis_backend_cfg),
            save_dir='temp_dir')
        visualizer.add_scalar('map', 0.9, step=0)
        for name in ['mock1', 'mock2']:
            assert visualizer.get_backend(name)._add_scalar is True

    def test_add_scalars(self):
        visualizer = Visualizer(
            vis_backends=copy.deepcopy(self.vis_backend_cfg),
            save_dir='temp_dir')
        input_dict = {'map': 0.7, 'acc': 0.9}
        visualizer.add_scalars(input_dict)
        for name in ['mock1', 'mock2']:
            assert visualizer.get_backend(name)._add_scalars is True

    def test_get_instance(self):

        class DetLocalVisualizer(Visualizer):

            def __init__(self, name):
                super().__init__(name)

        visualizer1 = DetLocalVisualizer.get_instance('name1')
        visualizer2 = Visualizer.get_current_instance()
        visualizer3 = DetLocalVisualizer.get_current_instance()
        assert id(visualizer1) == id(visualizer2) == id(visualizer3)

    def test_data_info(self):
        visualizer = Visualizer()
        visualizer.dataset_meta = {'class': 'cat'}
        assert visualizer.dataset_meta['class'] == 'cat'

    def test_show(self):
        cv2 = MagicMock()
        wait_continue = MagicMock()
        visualizer = Visualizer('test_show')
        img = np.ones([1, 1, 1])
        with patch('mmengine.visualization.visualizer.cv2', cv2), \
             patch('mmengine.visualization.visualizer.wait_continue',
                   wait_continue):
            # test default backend
            visualizer.show(
                drawn_img=img,
                win_name='test_show',
                wait_time=0,
                backend='matplotlib')
            assert hasattr(visualizer, 'manager')
            calls = [
                call(
                    visualizer.manager.canvas.figure,
                    timeout=0,
                    continue_key=' ')
            ]
            wait_continue.assert_has_calls(calls)

            # matplotlib backend
            visualizer.show(
                drawn_img=img,
                win_name='test_show',
                wait_time=0,
                backend='matplotlib')
            assert hasattr(visualizer, 'manager')
            calls = [
                call(
                    visualizer.manager.canvas.figure,
                    timeout=0,
                    continue_key=' '),
                call(
                    visualizer.manager.canvas.figure,
                    timeout=0,
                    continue_key=' ')
            ]
            wait_continue.assert_has_calls(calls)

            # cv2 backend
            visualizer.show(
                drawn_img=img,
                win_name='test_show',
                wait_time=0,
                backend='cv2')
            cv2.imshow.assert_called_once_with(str(id(visualizer)), img)

            # unknown backend
            with pytest.raises(ValueError):
                visualizer.show(
                    drawn_img=img,
                    win_name='test_show',
                    wait_time=0,
                    backend='unknown')
