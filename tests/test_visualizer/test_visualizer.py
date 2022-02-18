# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import pytest
import torch

from mmengine.visualizer import Visualizer


class TesVisualizer(TestCase):

    def setUp(self):
        """Setup the demo image in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.image = np.random.randint(0, 256, size=(10, 10, 3))

    def assert_img_equal(self, img, ref_img, ratio_thr=0.999):
        assert img.shape == ref_img.shape
        assert img.dtype == ref_img.dtype
        area = ref_img.shape[0] * ref_img.shape[1]
        diff = np.abs(img.astype('int32') - ref_img.astype('int32'))
        assert np.sum(diff <= 1) / float(area) > ratio_thr

    def test_init(self):
        # test `scale` parameter
        # `scale` must be greater than 0.
        with pytest.raises(AssertionError):
            Visualizer(scale=0)

        visualizer = Visualizer(scale=2, image=self.image)
        out_image = visualizer.get_image()
        assert (20, 20, 3) == out_image.shape
        assert visualizer.experiment == visualizer

    def test_set_image(self):
        visualizer = Visualizer()
        visualizer.set_image(self.image)
        # test grayscale image
        visualizer.set_image(self.image[..., 0])

    def test_get_image(self):
        visualizer = Visualizer(image=self.image)
        out_image = visualizer.get_image()
        assert self.assert_img_equal(self.image, out_image)

    def test_draw_bboxes(self):
        visualizer = Visualizer(image=self.image)

        #  only support 4 or nx4 tensor and numpy
        bboxes = torch.tensor([1, 1, 2, 2])
        visualizer.draw_bboxes(bboxes)
        bboxes = torch.tensor([[1, 1, 2, 2], [1, 1.5, 1, 2.5]])
        visualizer.draw_bboxes(
            bboxes, alpha=0.5, edge_color='b', line_style='-')

        bboxes = bboxes.numpy()
        visualizer.draw_bboxes(bboxes)

        # test incorrect bbox format
        # list is not supported
        with pytest.raises(AssertionError):
            bboxes = [1, 1, 2, 2]
            visualizer.draw_bboxes(bboxes)

    def test_draw_texts(self):
        visualizer = Visualizer(image=self.image)
        visualizer.draw_texts('text1', position=[5, 5])
        visualizer.draw_texts(['text1', 'text2'], position=[[5, 5], [3, 3]])
        visualizer.draw_texts(['text1', 'text2'], position=([5, 5], [3, 3]))
        visualizer.draw_texts(('text1', 'text2'), position=([5, 5], [3, 3]))
        visualizer.draw_texts('text1', position=torch.tensor([5, 5]))
        visualizer.draw_texts(['text1', 'text2'],
                              position=torch.tensor([[5, 5], [3, 3]]))
        visualizer.draw_texts('text1', position=np.array([5, 5]))
        visualizer.draw_texts(['text1', 'text2'],
                              position=np.array([[5, 5], [3, 3]]))

        # test length mismatch
        with pytest.raises(AssertionError):
            visualizer.draw_texts(['text1', 'text2'], position=[5, 5])
            visualizer.draw_texts('text1', position=[[5, 5]])
            visualizer.draw_texts('text1', position=[[5, 5], [3, 3]])

    def test_draw_lines(self):
        visualizer = Visualizer(image=self.image)

        visualizer.draw_lines(x_datas=[1, 5], y_datas=[2, 6])
        visualizer.draw_lines(x_datas=[1, 5, 4], y_datas=[2, 6, 6])
        visualizer.draw_lines(x_datas=(1, 5), y_datas=(2, 6))
        visualizer.draw_lines(
            x_datas=torch.tensor([1, 5]), y_datas=torch.tensor([2, 6]))
        visualizer.draw_lines(
            x_datas=np.array([1, 5, 4]), y_datas=np.array([2, 6, 6]))

        # test length mismatch
        with pytest.raises(AssertionError):
            visualizer.draw_lines(x_datas=[1, 5], y_datas=[2, 6, 7])

    def test_draw_circles(self):
        visualizer = Visualizer(image=self.image)
        visualizer.draw_circles([1, 5], radius=1)
        visualizer.draw_circles((1, 5), radius=1)
        visualizer.draw_circles([[1, 5], [2, 6]], radius=[1, 2])
        visualizer.draw_circles(torch.tensor([1, 5]))
        visualizer.draw_circles(np.array([1, 5]))

        # test length mismatch
        with pytest.raises(AssertionError):
            visualizer.draw_circles([1, 5], radius=[1, 2])

    def test_draw_polygons(self):
        visualizer = Visualizer(image=self.image)
        # shape Nx2 or list[Nx2]
        visualizer.draw_polygons(torch.tensor([[1, 1], [2, 2], [3, 4]]))
        visualizer.draw_polygons(np.array([[1, 1], [2, 2], [3, 4]]))
        visualizer.draw_polygons([
            np.array([[1, 1], [2, 2], [3, 4]]),
            torch.tensor([[1, 1], [2, 2], [3, 4]])
        ])

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

        # test non bool error
        binary_mask = np.random.randint(0, 2, size=(10, 10))
        with pytest.raises(AssertionError):
            visualizer.draw_binary_masks(binary_mask)

    def test_draw_featmap(self):
        # Supports tensor in BCHW format where channel C is 1 or 3
        # Supports tensor in BCHW format with channel C greater than 3
        # Support tensor in CHW format
        visualizer = Visualizer()
        with pytest.raises(AssertionError):
            featmap = torch.randn(2, 2, 3, 3)
            visualizer.draw_featmap(featmap)

        featmap = torch.randn(2, 6, 3, 3)
        visualizer.draw_featmap(featmap)
        featmap = torch.randn(2, 1, 3, 3)
        visualizer.draw_featmap(featmap)
        featmap = torch.randn(5, 3, 3)
        visualizer.draw_featmap(featmap, topk=3)

    def test_chain_call(self):
        visualizer = Visualizer(image=self.image)
        binary_mask = np.random.randint(0, 2, size=(10, 10)).astype(np.bool)
        visualizer.draw_bboxes(torch.tensor([1, 1, 2, 2])). \
            draw_texts('test', [5, 5]). \
            draw_lines([1, 5], [2, 6]). \
            draw_circles([1, 5]). \
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
                match=('instances is already registered in task_dict, '
                       'add "force=True" if you want to override it')):

            class DetVisualizer1(Visualizer):

                @Visualizer.register_task('instances')
                def draw_instance1(self, instances, data_type):
                    pass

                @Visualizer.register_task('instances')
                def draw_instance2(self, instances, data_type):
                    pass

        class DetVisualizer2(Visualizer):

            @Visualizer.register_task('instances')
            def draw_instance1(self, instances, data_type):
                pass

            @Visualizer.register_task('instances', force=True)
            def draw_instance2(self, instances, data_type):
                pass

        det_visualizer = DetVisualizer2()
        assert len(det_visualizer.task_dict) == 1
        assert 'instances' in det_visualizer.task_dict
        assert det_visualizer.task_dict[
            'instances'].__name__ == 'draw_instance2'
