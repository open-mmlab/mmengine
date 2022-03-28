from mmengine.temp_vis import Visualizer
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import cv2
import torch
from mmengine.data import BaseDataElement
import os

print(os.environ.pop('https_proxy'))


class DetLocalVisualizer(Visualizer):

    def add_datasample(self,
                       name,
                       image: Optional[np.ndarray] = None,
                       gt_sample: Optional['BaseDataSample'] = None,
                       pred_sample: Optional['BaseDataSample'] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       step=0) -> None:
        self.set_image(image)
        gt_bboxes = gt_sample.bboxes
        self.draw_bboxes(gt_bboxes)
        if 'masks' in gt_sample:
            self.draw_binary_masks(gt_sample.masks)
        gt_data = self.get_image()

        self.set_image(image)
        pred_bboxes = pred_sample.bboxes
        self.draw_bboxes(pred_bboxes)
        pred_data = self.get_image()

        concat = np.concatenate((gt_data, pred_data), axis=1)
        self.add_image(name, concat, step)


class DetWandbVisualizer(Visualizer):
    def __init__(self,
                 name='visualizer',
                 writers=None,
                 image: Optional[np.ndarray] = None,
                 metadata: Optional[dict] = None,
                 ):
        super().__init__(name, writers, image, metadata)
        if writers is not None:
            self.wandb = self.get_writer(1).experiment
            self.table = self.wandb.Table(columns=["gt", "pred"])
            self.class_id_to_label = {
                1: 'car'
            }
            self.class_set = self.wandb.Classes([{
                'id': 1,
                'name': 'car'
            }])

    def add_datasample(self,
                       name,
                       image: Optional[np.ndarray] = None,
                       gt_sample: Optional['BaseDataSample'] = None,
                       pred_sample: Optional['BaseDataSample'] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       step=0) -> None:
        gt_bboxes = gt_sample.bboxes
        position = dict(
            minX=int(gt_bboxes[0]),
            minY=int(gt_bboxes[1]),
            maxX=int(gt_bboxes[2]),
            maxY=int(gt_bboxes[3]))
        box_data = [{
            'position': position,
            'class_id': 1,
            'box_caption': 'car',
            'domain': 'pixel'
        }]
        box_data = {
            'ground_truth': {
                'box_data': box_data,
                'class_labels': self.class_id_to_label
            }
        }
        if 'masks' in gt_sample:
            mask = gt_sample.masks.astype(np.int32)

            mask_data = {
                'ground_truth': {
                    'mask_data': mask,
                    'class_labels': self.class_id_to_label
                }
            }
            gt_data = self.wandb.Image(img_data, boxes=box_data, masks=mask_data, classes=self.class_set)
        else:
            gt_data = self.wandb.Image(img_data, boxes=box_data, classes=self.class_set)

        pred_bboxes = pred_sample.bboxes
        position = dict(
            minX=int(pred_bboxes[0]),
            minY=int(pred_bboxes[1]),
            maxX=int(pred_bboxes[2]),
            maxY=int(pred_bboxes[3]))
        box_data = [{
            'position': position,
            'class_id': 1,
            'box_caption': 'car',
            'domain': 'pixel'
        }]
        box_data = {
            'predictions': {
                'box_data': box_data,
                'class_labels': self.class_id_to_label
            }
        }

        pred_data = self.wandb.Image(img_data, boxes=box_data, classes=self.class_set)

        self.table.add_data(gt_data, pred_data)
        self.wandb.log({name: self.table}, step=step)


if __name__ == '__main__':

    init_kwargs = {'project': 'hello_my'}

    save_dir = 'work_dir'
    writers = [dict(type='LocalWriter', save_dir=save_dir, img_show=False), dict(type='WandbWriter', init_kwargs=init_kwargs)]
    # writers = [dict(type='LocalWriter', save_dir=save_dir, img_show=True)]
    # writers = [dict(type='WandbWriter')]

    # 暂时不考虑全局唯一性
    # det_local_visualizer = DetLocalVisualizer(writers=writers)
    det_local_visualizer = DetWandbVisualizer(writers=writers)

    # 模拟数据
    img_path = 'demo.jpg'
    img_data = cv2.imread(img_path)[..., ::-1]  # rgb

    # det_local_visualizer.set_image(img_data)
    # det_local_visualizer.draw_bboxes(torch.tensor([100, 100, 500, 500]))
    # det_local_visualizer.show()

    mask = np.zeros_like(img_data)[..., 0]
    bbox = [100, 10, 500, 400]
    mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1
    mask = mask.astype(np.bool)
    gt_instances = BaseDataElement(
        data=dict(bboxes=torch.tensor([100, 100, 500, 500]), masks=mask))

    pred_instances = BaseDataElement(
        data=dict(bboxes=torch.tensor([40, 100, 400, 500])))
    det_local_visualizer.add_datasample('val_image', img_data, gt_instances, pred_instances, step=0)

    gt_instances = BaseDataElement(
        data=dict(bboxes=torch.tensor([100, 110, 500, 500])))
    pred_instances = BaseDataElement(
        data=dict(bboxes=torch.tensor([40, 120, 400, 500])))
    det_local_visualizer.add_datasample('val_image', img_data, gt_instances, pred_instances, step=1)
    # det_local_visualizer.close()

    # import wandb
    # wandb.init()
    # wandb.log({'acc': 2.1, 'loss': 3})
    # wandb.log({'img': wandb.Image(img_data)})

    # det_local_visualizer.show()

    # gt_instances = BaseDataElement(
    #     data=dict(bboxes=torch.tensor([100, 100, 500, 500])))
    # pred_instances = BaseDataElement(
    #     data=dict(bboxes=torch.tensor([40, 100, 400, 500])))
    # for i in range(5):
    #     det_local_visualizer.draw(img_data, gt_instances)
    #     det_local_visualizer.add_image('val_image', step=i)

    # wandb = det_local_visualizer.get_writer(1).experiment
    #
    # table = wandb.Table(columns=["x", "y"])
    #
    # table.add_data(wandb.Image(img_data), wandb.Image(img_data))
    # wandb.log({'image': table}, step=0)
    #
    # img_path = '000000259625.jpg'
    # img_data = cv2.imread(img_path)[..., ::-1]  # rgb
    #
    # table.add_data(wandb.Image(img_data), wandb.Image(img_data))
    # wandb.log({'image': table}, step=1)
    #
    # table.add_data(wandb.Image(img_data), wandb.Image(img_data))
    # wandb.log({'image': table}, step=2)
    #
    # img_path = 'demo.jpg'
    # img_data = cv2.imread(img_path)[..., ::-1]  # rgb
    #
    # table.add_data(wandb.Image(img_data), wandb.Image(img_data))
    # wandb.log({'image': table}, step=3)
