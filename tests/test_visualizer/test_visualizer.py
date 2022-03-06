# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from mmengine.data import BaseDataSample
from mmengine.visualization import Visualizer


class TestVisualizer(TestCase):

    def setUp(self):
        """Setup the demo image in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.image = np.random.randint(
            0, 256, size=(10, 10, 3)).astype('uint8')

    def test_init(self):
        visualizer = Visualizer(image=self.image)
        visualizer.get_image()

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
            bboxes, alpha=0.5, edgecolors='b', linestyles='-')
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
        with pytest.raises(AssertionError):
            visualizer.draw_bboxes([1, 1, 2, 2])

    def test_close(self):
        visualizer = Visualizer(image=self.image)
        fig_num = visualizer.fig.number
        assert fig_num in plt.get_fignums()
        visualizer.close()
        assert fig_num not in plt.get_fignums()

    def test_draw_texts(self):
        visualizer = Visualizer(image=self.image)

        # only support tensor and numpy
        visualizer.draw_texts('text1', positions=torch.tensor([5, 5]))
        visualizer.draw_texts(['text1', 'text2'],
                              positions=torch.tensor([[5, 5], [3, 3]]))
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
        with pytest.raises(AssertionError):
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
                                  verticalalignments=['top'])
        with pytest.raises(AssertionError):
            visualizer.draw_texts(['text1', 'test2'],
                                  positions=torch.tensor([[5, 5], [3, 3]]),
                                  horizontalalignments=['left'])
        with pytest.raises(AssertionError):
            visualizer.draw_texts(['text1', 'test2'],
                                  positions=torch.tensor([[5, 5], [3, 3]]),
                                  font_sizes=[1])

        # test type valid
        with pytest.raises(AssertionError):
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
            linestyles=['-', '-.'],
            linewidths=[1, 2])
        # test out of bounds
        with pytest.warns(
                UserWarning,
                match='Warning: The line is out of bounds,'
                ' the drawn line may not be in the image'):
            visualizer.draw_lines(
                x_datas=torch.tensor([12, 5]), y_datas=torch.tensor([2, 6]))

        # test incorrect format
        with pytest.raises(AssertionError):
            visualizer.draw_lines(x_datas=[5, 5], y_datas=torch.tensor([2, 6]))
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

        # test filling
        visualizer.draw_circles(
            torch.tensor([[1, 5], [2, 6]]),
            radius=torch.tensor([1, 2]),
            is_filling=True)

        # test config
        visualizer.draw_circles(
            torch.tensor([[1, 5], [2, 6]]),
            radius=torch.tensor([1, 2]),
            edgecolors=['g', 'r'],
            linestyles=['-', '-.'],
            linewidths=[1, 2])

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
        with pytest.raises(AssertionError):
            visualizer.draw_circles([1, 5], radius=torch.tensor([1]))
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
            is_filling=True)
        visualizer.draw_polygons(
            polygons=[
                np.array([[1, 1], [2, 2], [3, 4]]),
                torch.tensor([[1, 1], [2, 2], [3, 4]])
            ],
            edgecolors=['r', 'g'],
            linestyles='-',
            linewidths=[2, 1])

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
        # mode only supports 'mean' and 'max' and 'min
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
        visualizer.draw_featmap(torch.randn(6, 3, 3), mode='min')
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

    def test_register_task(self):

        class DetVisualizer(Visualizer):

            @Visualizer.register_task('instances')
            def draw_instance(self, instances, data_type):
                pass

        assert len(Visualizer.task_dict) == 1
        assert 'instances' in Visualizer.task_dict

        # test registration of the same names.
        with pytest.raises(
                KeyError,
                match=('"instances" is already registered in task_dict, '
                       'add "force=True" if you want to override it')):

            class DetVisualizer1(Visualizer):

                @Visualizer.register_task('instances')
                def draw_instance1(self, instances, data_type):
                    pass

                @Visualizer.register_task('instances')
                def draw_instance2(self, instances, data_type):
                    pass

        Visualizer.task_dict = dict()

        class DetVisualizer2(Visualizer):

            @Visualizer.register_task('instances')
            def draw_instance1(self, instances, data_type):
                pass

            @Visualizer.register_task('instances', force=True)
            def draw_instance2(self, instances, data_type):
                pass

            def draw(self,
                     image: np.ndarray = None,
                     data_sample: 'BaseDataSample' = None,
                     draw_gt: bool = True,
                     draw_pred: bool = True) -> None:
                return super().draw(image, data_sample, draw_gt, draw_pred)

        det_visualizer = DetVisualizer2()
        det_visualizer.draw(data_sample={})
        assert len(det_visualizer.task_dict) == 1
        assert 'instances' in det_visualizer.task_dict
        assert det_visualizer.task_dict[
            'instances'].__name__ == 'draw_instance2'
