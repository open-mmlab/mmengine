# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp

import numpy as np
import pytest
import torch

from mmengine.infer import BaseInferencer
from mmengine.registry import VISUALIZERS, DefaultScope
from mmengine.testing import RunnerTestCase
from mmengine.utils import is_list_of
from mmengine.visualization import Visualizer


def is_imported(package):
    try:
        __import__(package)
        return True
    except ImportError:
        return False


class ToyInferencer(BaseInferencer):
    preprocess_kwargs = {'pre_arg'}
    forward_kwargs = {'for_arg'}
    visualize_kwargs = {'vis_arg'}
    postprocess_kwargs = {'pos_arg'}

    def preprocess(self, inputs, batch_size=1, pre_arg=None, **kwargs):
        return super().preprocess(inputs, batch_size, **kwargs)

    def forward(self, inputs, for_arg=None, **kwargs):
        return super().forward(inputs, **kwargs)

    def visualize(self, inputs, preds, vis_arg=None, **kwargs):
        return inputs

    def postprocess(self,
                    preds,
                    imgs,
                    return_datasamples,
                    pos_arg=None,
                    **kwargs):
        return imgs, preds

    def _init_pipeline(self, cfg):

        def pipeline(img):
            if isinstance(img, str):
                img = np.load(img, allow_pickle=True)
                img = torch.from_numpy(img).float()
            elif isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            else:
                img = torch.tensor(img).float()
            return img

        return pipeline


class ToyVisualizer(Visualizer):
    ...


class TestBaseInferencer(RunnerTestCase):

    def setUp(self) -> None:
        super().setUp()
        runner = self.build_runner(copy.deepcopy(self.epoch_based_cfg))
        runner.train()
        self.cfg_path = osp.join(runner.work_dir, f'{runner.timestamp}.py')
        self.ckpt_path = osp.join(runner.work_dir, 'epoch_1.pth')
        VISUALIZERS.register_module(module=ToyVisualizer, name='ToyVisualizer')

    def test_custom_inferencer(self):
        # Inferencer should not define ***_kwargs with duplicate keys.
        with self.assertRaisesRegex(AssertionError, 'Class define error'):

            class CustomInferencer(BaseInferencer):
                preprocess_kwargs = set('a')
                forward_kwargs = set('a')

    def tearDown(self):
        VISUALIZERS._module_dict.pop('ToyVisualizer')
        return super().tearDown()

    def test_init(self):
        # Pass model as Config
        cfg = copy.deepcopy(self.epoch_based_cfg)
        ToyInferencer(cfg, self.ckpt_path)
        # Pass model as ConfigDict
        ToyInferencer(cfg._cfg_dict, self.ckpt_path)
        # Pass model as normal dict
        ToyInferencer(dict(cfg._cfg_dict), self.ckpt_path)
        # Pass model as string point to path of config
        ToyInferencer(self.cfg_path, self.ckpt_path)

        cfg.model.pretrained = 'fake_path'
        inferencer = ToyInferencer(cfg, self.ckpt_path)
        self.assertNotIn('pretrained', inferencer.cfg.model)

        # Pass invalid model
        with self.assertRaisesRegex(TypeError, 'model must'):
            ToyInferencer([self.epoch_based_cfg], self.ckpt_path)

        # Pass model as model name defined in metafile
        if is_imported('mmdet'):
            from mmdet.utils import register_all_modules

            register_all_modules()
            ToyInferencer(
                'faster-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco',
                'https://download.openmmlab.com/mmdetection/v2.0/resnest/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20200926_125502-20289c16.pth',  # noqa: E501
            )

        checkpoint = self.ckpt_path
        ToyInferencer(weights=checkpoint)

    def test_call(self):
        num_imgs = 12
        imgs = []
        img_paths = []
        for i in range(num_imgs):
            img = np.random.random((1, 2))
            img_path = osp.join(self.temp_dir.name, f'{i}.npy')
            img.dump(img_path)
            imgs.append(img)
            img_paths.append(img_path)

        inferencer = ToyInferencer(self.cfg_path, self.ckpt_path)
        inferencer(imgs)
        inferencer(img_paths)

    @pytest.mark.skipif(
        not is_imported('mmdet'), reason='mmdet is not installed')
    def test_load_model_from_meta(self):
        from mmdet.utils import register_all_modules

        register_all_modules()
        inferencer = ToyInferencer(self.cfg_path, self.ckpt_path)
        inferencer._load_model_from_metafile('retinanet_r18_fpn_1x_coco')
        with self.assertRaisesRegex(ValueError, 'Cannot find model'):
            inferencer._load_model_from_metafile('fake_model')
        # TODO: Test alias

    def test_init_model(self):
        inferencer = ToyInferencer(self.cfg_path, self.ckpt_path)
        model = inferencer._init_model(self.iter_based_cfg, self.ckpt_path)
        self.assertFalse(model.training)

    def test_get_chunk_data(self):
        inferencer = ToyInferencer(self.cfg_path, self.ckpt_path)
        data = list(range(1, 11))
        chunk_data = inferencer._get_chunk_data(data, 3)
        self.assertEqual(
            list(chunk_data), [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]])

    def test_init_visualizer(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        inferencer = ToyInferencer(self.cfg_path, self.ckpt_path)
        visualizer = inferencer._init_visualizer(cfg)
        self.assertIsNone(visualizer, None)
        cfg.visualizer = dict(type='ToyVisualizer')
        visualizer = inferencer._init_visualizer(cfg)
        self.assertIsInstance(visualizer, ToyVisualizer)

        # Visualizer could be built with the same name repeatedly.
        cfg.visualizer = dict(type='ToyVisualizer', name='toy')
        visualizer = inferencer._init_visualizer(cfg)
        visualizer = inferencer._init_visualizer(cfg)

    def test_dispatch_kwargs(self):
        inferencer = ToyInferencer(self.cfg_path, self.ckpt_path)
        kwargs = dict(
            pre_arg=dict(a=1),
            for_arg=dict(c=2),
            vis_arg=dict(b=3),
            pos_arg=dict(d=4))
        pre_arg, for_arg, vis_arg, pos_arg = inferencer._dispatch_kwargs(
            **kwargs)
        self.assertEqual(pre_arg, dict(pre_arg=dict(a=1)))
        self.assertEqual(for_arg, dict(for_arg=dict(c=2)))
        self.assertEqual(vis_arg, dict(vis_arg=dict(b=3)))
        self.assertEqual(pos_arg, dict(pos_arg=dict(d=4)))
        # Test unknown arg.
        kwargs = dict(return_datasample=dict())
        with self.assertRaisesRegex(ValueError, 'unknown'):
            inferencer._dispatch_kwargs(**kwargs)

    def test_preprocess(self):
        inferencer = ToyInferencer(self.cfg_path, self.ckpt_path)
        data = list(range(1, 11))
        pre_data = inferencer.preprocess(data, batch_size=3)
        target_data = [
            [torch.tensor(1),
             torch.tensor(2),
             torch.tensor(3)],
            [torch.tensor(4),
             torch.tensor(5),
             torch.tensor(6)],
            [torch.tensor(7),
             torch.tensor(8),
             torch.tensor(9)],
            [torch.tensor(10)],
        ]
        self.assertEqual(list(pre_data), target_data)
        os.mkdir(osp.join(self.temp_dir.name, 'imgs'))
        for i in range(1, 11):
            img = np.array(1)
            img.dump(osp.join(self.temp_dir.name, 'imgs', f'{i}.npy'))
        # Passing a directory of images.
        inputs = inferencer._inputs_to_list(
            osp.join(self.temp_dir.name, 'imgs'))
        dataloader = inferencer.preprocess(inputs, batch_size=3)
        for data in dataloader:
            self.assertTrue(is_list_of(data, torch.Tensor))

    @pytest.mark.skipif(
        not is_imported('mmdet'), reason='mmdet is not installed')
    def test_list_models(self):
        model_list = BaseInferencer.list_models('mmdet')
        self.assertTrue(len(model_list) > 0)
        DefaultScope._instance_dict.clear()
        with self.assertRaisesRegex(AssertionError, 'scope should be'):
            BaseInferencer.list_models()
        with self.assertRaisesRegex(AssertionError, 'unknown not in'):
            BaseInferencer.list_models('unknown')
