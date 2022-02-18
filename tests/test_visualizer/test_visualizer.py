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

        #  4 or nx4
        bboxes = np.array([[1, 1, 2, 2], [1, 1.5, 1, 2.5]])
        visualizer.draw_bboxes(bboxes)

        # [x1,y1,x2,y2] or [[x11,y11,x12,y12],[x21,y21,x22,y22]]
        bboxes = [1, 1, 2, 2]
        visualizer.draw_bboxes(bboxes)
        bboxes = [[1, 1, 2, 2], [1, 1.5, 1, 2.5]]
        visualizer.draw_bboxes(bboxes)

        # test incorrect bbox format
        with pytest.raises(AssertionError):
            bboxes = [1, 1, 2, 2, 1]
            visualizer.draw_bboxes(bboxes)

    def test_draw_texts(self):
        visualizer = Visualizer(image=self.image)
        visualizer.draw_texts('test', [5, 5])

    def test_draw_lines(self):
        visualizer = Visualizer(image=self.image)
        visualizer.draw_lines([1, 5])

    def test_draw_circles(self):
        visualizer = Visualizer(image=self.image)
        visualizer.draw_circles([1, 5])

    def test_draw_polygons(self):
        visualizer = Visualizer(image=self.image)
        visualizer.draw_polygons([1, 5])

    def test_draw_binary_masks(self):
        binary_mask = np.random.randint(0, 2, size=(10, 10, 3)).astype(np.bool)
        visualizer = Visualizer(image=self.image)
        visualizer.draw_binary_masks(binary_mask)

        # test the error that the size of mask and image are different.
        with pytest.raises(AssertionError):
            binary_mask = np.random.randint(
                0, 2, size=(8, 10, 3)).astype(np.bool)
            visualizer.draw_binary_masks(binary_mask)

    # TODO
    def test_draw_featmap(self):
        visualizer = Visualizer()
        with pytest.raises(AssertionError):
            featmap = torch.randn(2, 2, 3, 3)
            visualizer.draw_featmap(featmap)

        featmap = torch.randn(2, 6, 3, 3)
        visualizer.draw_featmap(featmap)
        featmap = torch.randn(2, 1, 3, 3)
        visualizer.draw_featmap(featmap)

    def test_chain_call(self):
        visualizer = Visualizer(image=self.image)
        binary_mask = np.random.randint(0, 2, size=(10, 10, 3)).astype(np.bool)
        visualizer.draw_bboxes([1, 1, 2, 2]). \
            draw_texts('test', [5, 5]). \
            draw_lines([1, 5]). \
            draw_circles([1, 5]). \
            draw_polygons([1, 5]). \
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
