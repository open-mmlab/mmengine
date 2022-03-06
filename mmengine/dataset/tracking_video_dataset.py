from mmengine.dataset import BaseDataset
from mmtrack.datasets.parsers import CocoVID
import os.path as osp
import numpy as np
from mmcv.utils import print_log
import random


class TrackingVideoDataset(BaseDataset):
    META = dict(
        CLASSES=('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus',
                 'car', 'cattle', 'dog', 'domestic_cat', 'elephant', 'fox',
                 'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
                 'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake',
                 'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale',
                 'zebra'))

    def __init__(self,
                 load_as_video=True,
                 ref_img_sampler=dict(
                     num_ref_imgs=2,
                     frame_range=9,
                     filter_key_img=True,
                     method='bilateral_uniform'),
                 seg_prefix=None,
                 proposal_file=None,
                 *args,
                 **kwargs):
        self.load_as_video = load_as_video
        self.ref_img_sampler = ref_img_sampler
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        coco = CocoVID(ann_file)
        self.cat_ids = coco.get_cat_ids(cat_names=self.meta['CLASSES'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        data_infos = []
        vid_ids = coco.get_vid_ids()
        for vid_id in vid_ids[:1]:
            img_ids = coco.get_img_ids_from_vid(vid_id)

            for img_id in img_ids:
                # load img info
                img_info = coco.load_imgs([img_id])[0]
                # img_info['filename'] = osp.join(self.data_prefix['img'],
                #                                 img_info['file_name'])
                img_info['filename'] = img_info['file_name']
                img_info['video_len'] = len(img_ids)

                # load ann info
                ann_ids = coco.get_ann_ids(
                    img_ids=[img_id], cat_ids=self.cat_ids)
                ann_info = coco.load_anns(ann_ids)
                ann_info = self._parse_ann_info(img_info, ann_info)

                # get data_info
                data_info = dict(img_info=img_info, ann_info=ann_info)
                data_infos.append(data_info)
        return data_infos

    def _prepare_data(self, idx):
        data_info = self.get_data_info(idx)
        if self.ref_img_sampler is not None:
            data_infos = self.ref_img_sampling(idx, data_info,
                                               **self.ref_img_sampler)
            results = [
                self.prepare_results(data_info) for data_info in data_infos
            ]
        else:
            results = self.prepare_results(data_info)

        return self.pipeline(results)

    def prepare_results(self, data_info):
        results = data_info.copy()
        results['is_video_data'] = self.load_as_video
        results['img_prefix'] = self.data_prefix['img']
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        return results

    def ref_img_sampling(self,
                         idx,
                         data_info,
                         frame_range,
                         stride=1,
                         num_ref_imgs=1,
                         filter_key_img=True,
                         method='uniform',
                         return_key_img=True):
        assert isinstance(data_info, dict)
        img_info = data_info['img_info']
        assert isinstance(img_info, dict)
        if isinstance(frame_range, int):
            assert frame_range >= 0, 'frame_range can not be a negative value.'
            frame_range = [-frame_range, frame_range]
        elif isinstance(frame_range, list):
            assert len(frame_range) == 2, 'The length must be 2.'
            assert frame_range[0] <= 0 and frame_range[1] >= 0
            for i in frame_range:
                assert isinstance(i, int), 'Each element must be int.'
        else:
            raise TypeError('The type of frame_range must be int or list.')

        if 'test' in method and \
                (frame_range[1] - frame_range[0]) != num_ref_imgs:
            print_log(
                'Warning:'
                "frame_range[1] - frame_range[0] isn't equal to num_ref_imgs."
                'Set num_ref_imgs to frame_range[1] - frame_range[0].',
                logger=self.logger)
            self.ref_img_sampler[
                'num_ref_imgs'] = frame_range[1] - frame_range[0]

        if (not self.load_as_video) or img_info.get('frame_id', -1) < 0 \
                or (frame_range[0] == 0 and frame_range[1] == 0):
            ref_data_infos = []
            for i in range(num_ref_imgs):
                ref_data_infos.append(data_info.copy())
        else:
            vid_id, img_id, frame_id = img_info['video_id'], img_info[
                'id'], img_info['frame_id']
            left = max(0, frame_id + frame_range[0])
            right = min(frame_id + frame_range[1], img_info['video_len'] - 1)
            img_ids = list(range(0, img_info['video_len']))

            ref_img_ids = []
            if method == 'uniform':
                valid_ids = img_ids[left:right + 1]
                if filter_key_img and img_id in valid_ids:
                    valid_ids.remove(img_id)
                num_samples = min(num_ref_imgs, len(valid_ids))
                ref_img_ids.extend(random.sample(valid_ids, num_samples))
            elif method == 'bilateral_uniform':
                assert num_ref_imgs % 2 == 0, \
                    'only support load even number of ref_imgs.'
                for mode in ['left', 'right']:
                    if mode == 'left':
                        valid_ids = img_ids[left:frame_id + 1]
                    else:
                        valid_ids = img_ids[frame_id:right + 1]
                    if filter_key_img and img_id in valid_ids:
                        valid_ids.remove(img_id)
                    num_samples = min(num_ref_imgs // 2, len(valid_ids))
                    sampled_inds = random.sample(valid_ids, num_samples)
                    ref_img_ids.extend(sampled_inds)
            elif method == 'test_with_adaptive_stride':
                if frame_id == 0:
                    stride = float(len(img_ids) - 1) / (num_ref_imgs - 1)
                    for i in range(num_ref_imgs):
                        ref_id = round(i * stride)
                        ref_img_ids.append(img_ids[ref_id])
            elif method == 'test_with_fix_stride':
                if frame_id == 0:
                    for i in range(frame_range[0], 1):
                        ref_img_ids.append(img_ids[0])
                    for i in range(1, frame_range[1] + 1):
                        ref_id = min(round(i * stride), len(img_ids) - 1)
                        ref_img_ids.append(img_ids[ref_id])
                elif frame_id % stride == 0:
                    ref_id = min(
                        round(frame_id + frame_range[1] * stride),
                        len(img_ids) - 1)
                    ref_img_ids.append(img_ids[ref_id])
                img_info['num_left_ref_imgs'] = abs(frame_range[0]) \
                    if isinstance(frame_range, list) else frame_range
                img_info['frame_stride'] = stride
            else:
                raise NotImplementedError

            ref_data_infos = []
            for ref_img_id in ref_img_ids:
                offset = ref_img_id - img_info['frame_id']
                ref_data_info = self.get_data_info(idx + offset)
                ref_data_infos.append(ref_data_info)
            ref_data_infos = sorted(
                ref_data_infos, key=lambda i: i['img_info']['frame_id'])

        if return_key_img:
            return [data_info, *ref_data_infos]
        else:
            return ref_data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotations.

        Args:
            img_anfo (dict): Information of image.
            ann_info (list[dict]): Annotation information of image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
            labels, instance_ids, masks, seg_map. "masks" are raw
            annotations and not decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks = []
        gt_instance_ids = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                if 'segmentation' in ann:
                    gt_masks.append(ann['segmentation'])
                if 'instance_id' in ann:
                    gt_instance_ids.append(ann['instance_id'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks,
            seg_map=seg_map)

        if self.load_as_video:
            ann['instance_ids'] = np.array(gt_instance_ids).astype(np.int)
        else:
            ann['instance_ids'] = np.arange(len(gt_labels))

        return ann


# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadMultiImagesFromFile'),
#     dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
#     dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
#     dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
#     dict(type='SeqNormalize', **img_norm_cfg),
#     dict(type='SeqPad', size_divisor=16),
#     dict(
#         type='VideoCollect',
#         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
#     dict(type='ConcatVideoReferences'),
#     dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
# ]
# import time
# a = time.time()
# self = TrackingVideoDataset(
#     ann_file='annotations/imagenet_vid_train.json',
#     meta=None,
#     data_root='./data/ILSVRC',
#     data_prefix=dict(img='Data/VID'),
#     filter_cfg=None,
#     num_samples=-1,
#     serialize_data=True,
#     pipeline=train_pipeline,
#     test_mode=False,
#     lazy_init=False,
#     max_refetch=1000)
# print(time.time() - a)


# def show_key_shape(data):
#     for k, v in data.items():
#         try:
#             print(k, v.data.shape)
#         except:
#             print(k, type(v.data), len(v))


# for i in range(len(self)):
#     data = self[i]
#     assert data['img_metas'].data['filename'].split(
#         '/')[-2] == data['ref_img_metas'].data[0]['filename'].split('/')[-2]
#     assert data['img_metas'].data['filename'].split(
#         '/')[-2] == data['ref_img_metas'].data[1]['filename'].split('/')[-2]
# import pdb
# pdb.set_trace()
