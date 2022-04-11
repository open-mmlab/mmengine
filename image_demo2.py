from mmcv import Config
import mmcv
from mmdet.apis import (inference_detector, init_detector, show_result_pyplot)
from mmengine.visualization import Visualizer
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import cv2
import torch
from mmengine.data import BaseDataElement
import os
from mmdet.core.visualization.palette import get_palette, palette_val

def _get_adaptive_scales(areas, min_area=800, max_area=30000):
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``'min_area'``, the scale is 0.5 while the area is larger than
    ``'max_area'``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Default: 800.
        max_area (int): Upper bound areas for adaptive scales.
            Default: 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


class DetLocalVisualizer(Visualizer):
    def __init__(self,
                 name='visualizer',
                 vis_backends=None,
                 image: Optional[np.ndarray] = None,
                 metadata: Optional[dict] = None,
                 show=True):
        super().__init__(name, vis_backends, image, metadata)
        self._show = show
        if self._metadata:
            self.classes = self._metadata['classes']

    def add_datasample(self,
                       name,
                       image: Optional[np.ndarray] = None,
                       gt_sample: Optional['BaseDataSample'] = None,
                       pred_sample: Optional['BaseDataSample'] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       step=0) -> None:
        gt_data = None
        pred_data = None
        if draw_gt and gt_sample is not None:
            self.set_image(image)
            gt_bboxes = gt_sample.bboxes
            gt_labels = gt_sample.labels
            gt_masks = gt_sample.masks
            self.draw_bboxes(gt_bboxes)
            if 'masks' in gt_sample:
                self.draw_binary_masks(gt_sample.masks)
            gt_data = self.get_image()

        if draw_pred and pred_sample is not None:
            self.set_image(image)
            pred_bboxes = pred_sample.bboxes
            pred_labels = pred_sample.labels
            segms = pred_sample.masks

            max_label = int(max(pred_labels) if len(pred_labels) > 0 else 0)
            text_palette = palette_val(get_palette((200, 200, 200), max_label + 1))
            text_colors = [text_palette[label] for label in pred_labels]

            thickness = 2

            if pred_bboxes is not None:
                num_bboxes = pred_bboxes.shape[0]
                bbox_palette = palette_val(get_palette(None, max_label + 1))
                colors = [bbox_palette[label] for label in pred_labels[:num_bboxes]]
                self.draw_bboxes(pred_bboxes[:, :4], edgecolors=colors, alpha=0.8)

                positions = pred_bboxes[:, :2].astype(np.int32) + thickness
                areas = (pred_bboxes[:, 3] - pred_bboxes[:, 1]) * (pred_bboxes[:, 2] - pred_bboxes[:, 0])
                scales = _get_adaptive_scales(areas)
                scores = pred_bboxes[:, 4] if pred_bboxes.shape[1] == 5 else None

                for i, (pos, label) in enumerate(zip(positions, pred_labels)):
                    label_text = self.classes[label] if self.classes is not None else f'class {label}'
                    if scores is not None:
                        label_text += f'|{scores[i]:.02f}'

                    self.draw_texts(
                        label_text,
                        pos,
                        colors=[list(text_colors[i])],
                        font_sizes=int(13 * scales[i]),
                        bboxes=[{
                            'facecolor': 'black',
                            'alpha': 0.8,
                            'pad': 0.7,
                            'edgecolor': 'none'
                        }])

            if segms is not None:
                mask_palette = get_palette(None, max_label + 1)
                colors = [mask_palette[label] for label in pred_labels]
                colors = np.array(colors, dtype=np.uint8)

                # self.draw_binary_masks(segms, colors, with_edge=True)
                self.draw_binary_masks(segms, colors)

                # if num_bboxes < segms.shape[0]:
                #     segms = segms[num_bboxes:]
                #     horizontal_alignment = 'center'
                #     areas = []
                #     positions = []
                #     for mask in segms:
                #         _, _, stats, centroids = cv2.connectedComponentsWithStats(
                #             mask.astype(np.uint8), connectivity=8)
                #         largest_id = np.argmax(stats[1:, -1]) + 1
                #         positions.append(centroids[largest_id])
                #         areas.append(stats[largest_id, -1])
                #     areas = np.stack(areas, axis=0)
                #     scales = _get_adaptive_scales(areas)
                #     draw_labels(
                #         ax,
                #         labels[num_bboxes:],
                #         positions,
                #         class_names=class_names,
                #         color=text_colors,
                #         font_size=font_size,
                #         scales=scales,
                #         horizontal_alignment=horizontal_alignment)

            pred_data = self.get_image()
            if self._show:
                # TODO
                # import matplotlib.pyplot as plt
                # plt.close()
                # plt.imshow(pred_data)
                # plt.show()
                # plt.close()
                self.show(pred_data, name)

        # if gt_data is not None and pred_data is not None:
        #     concat = np.concatenate((gt_data, pred_data), axis=1)
        #     self.add_image(name, concat, step)
        # elif gt_data is not None:
        #     self.add_image(name, gt_data, step)
        # elif pred_data is not None:
        #     self.add_image(name, pred_data, step)


def main(img, config_path, ckpt_path, score_thr=0.3):
    # vis_backend = [dict(type='LocalWriter', save_dir='a', img_show=True)]

    # build the model from a config file and a checkpoint file
    cfg = Config.fromfile(config_path)
    model = init_detector(cfg, ckpt_path, device='cuda')

    metadata = {'classes': model.CLASSES}
    det_local_visualizer = DetLocalVisualizer(metadata=metadata)

    # test a single image
    result = inference_detector(model, img)

    # 封装为 datasample
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)

    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    img = mmcv.imread(img)
    img = mmcv.bgr2rgb(img)

    pred_instances = BaseDataElement(
        data=dict(bboxes=bboxes, labels=labels, masks=segms))

    # TODO
    det_local_visualizer.add_datasample('image', img, pred_sample=pred_instances, step=0)
    # show_result_pyplot(model, imgs[i], result[i], score_thr=args.score_thr)


if __name__ == '__main__':
    img = 'dog.jpg'
    config = '../configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
    # TODO
    # https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth
    ckpt_path = '/home/PJLAB/huanghaian/checkpoint/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'

    main(img, config, ckpt_path)
