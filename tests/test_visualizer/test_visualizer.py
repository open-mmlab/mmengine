# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import numpy as np
import torch
import tempfile
import os
import os.path as osp
from unittest import TestCase

import mmcv


class TestLocalVisualizer(TestCase):

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
            LocalVisualizer(scale=0)

    def test_set_image(self):
        local_visualizer = LocalVisualizer()
        local_visualizer.set_image(self.image)
        # test grayscale image
        local_visualizer.set_image(self.image[..., 0])

    def test_get_image(self):
        local_visualizer = LocalVisualizer(image=self.image)
        out_image = local_visualizer.get_image()
        assert np.allclose(self.image, out_image)

    def test_save(self):
        out_file = osp.join(tempfile.gettempdir(), 'test.jpg')
        local_visualizer = LocalVisualizer(image=self.image)
        local_visualizer.save(out_file)
        rewrite_img = mmcv.imread(out_file)
        os.remove(out_file)
        self.assert_img_equal(self.image, rewrite_img)

        local_visualizer.save(out_file, local_visualizer.get_image())
        rewrite_img = mmcv.imread(out_file)
        os.remove(out_file)
        self.assert_img_equal(self.image, rewrite_img)

    def test_draw_bboxes(self):
        local_visualizer = LocalVisualizer(image=self.image)

        #  4 or nx4
        bboxes = np.array([[1, 1, 2, 2], [1, 1.5, 1, 2.5]])
        local_visualizer.draw_bboxes(bboxes)

        # [x1,y1,x2,y2] or [[x11,y11,x12,y12],[x21,y21,x22,y22]]
        bboxes = [1, 1, 2, 2]
        local_visualizer.draw_bboxes(bboxes)
        bboxes = [[1, 1, 2, 2], [1, 1.5, 1, 2.5]]
        local_visualizer.draw_bboxes(bboxes)

        # test incorrect bbox format
        with pytest.raises(AssertionError):
            bboxes = [1, 1, 2, 2, 1]
            local_visualizer.draw_bboxes(bboxes)

    def test_draw_texts(self):
        local_visualizer = LocalVisualizer(image=self.image)
        local_visualizer.draw_texts('test', [5, 5])

    def test_draw_lines(self):
        local_visualizer = LocalVisualizer(image=self.image)
        local_visualizer.draw_lines([1, 5])

    def test_draw_circles(self):
        local_visualizer = LocalVisualizer(image=self.image)
        local_visualizer.draw_circles([1, 5])

    def test_draw_polygons(self):
        local_visualizer = LocalVisualizer(image=self.image)
        local_visualizer.draw_polygons([1, 5])

    def test_draw_binary_masks(self):
        binary_mask=np.random.randint(0, 2, size=(10, 10, 3)).astype(np.bool)
        local_visualizer = LocalVisualizer(image=image)
        local_visualizer.draw_binary_masks(binary_mask)

    # TODO
    def test_draw_featmap(self):
        local_visualizer = LocalVisualizer()
        with pytest.raises(AssertionError):
            featmap = torch.randn(2, 2, 3, 3)
            local_visualizer.draw_featmap(featmap)

        featmap = torch.randn(2, 6, 3, 3)
        local_visualizer.draw_featmap(featmap)
        featmap = torch.randn(2, 1, 3, 3)
        local_visualizer.draw_featmap(featmap)

    def test_chain_call(self):
        local_visualizer = LocalVisualizer(image=self.image)
        binary_mask = np.random.randint(0, 2, size=(10, 10, 3)).astype(np.bool)
        local_visualizer.draw_bboxes([1, 1, 2, 2]).\
            draw_texts('test', [5, 5]).\
            draw_lines([1, 5]).\
            draw_circles([1, 5]).\
            draw_polygons([1, 5]).\
            draw_binary_masks(binary_mask)

     def test_register_task(self):
         class DetLocalVisualizer(LocalVisualizer):
             @LocalVisualizer.register_task('instances')
             def draw_instance(self, instances, data_type):
                 pass

         assert len(LocalVisualizer.task_dict)==1
         assert 'instances' in LocalVisualizer.task_dict

         # test registration of the same name.
         with pytest.raises(AssertionError):
             class DetLocalVisualizer(LocalVisualizer):
                 @LocalVisualizer.register_task('instances')
                 def draw_instance1(self, instances, data_type):
                     pass

                 @LocalVisualizer.register_task('instances')
                 def draw_instance2(self, instances, data_type):
                     pass

        class DetLocalVisualizer(LocalVisualizer):
            @LocalVisualizer.register_task('instances')
            def draw_instance1(self, instances, data_type):
                pass

            @LocalVisualizer.register_task('instances',force=True)
            def draw_instance2(self, instances, data_type):
                pass

        assert len(LocalVisualizer.task_dict) == 1
        assert 'instances' in LocalVisualizer.task_dict
        assert LocalVisualizer.task_dict['instances'].__name__ == 'draw_instance2'