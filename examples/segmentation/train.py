import argparse
import csv
import os
import os.path as osp
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.optim import AdamW
from torchvision.datasets import VisionDataset
from torchvision.models.segmentation import deeplabv3_resnet50

from mmengine.dist import master_only
from mmengine.evaluator import BaseMetric
from mmengine.hooks import Hook
from mmengine.model import BaseModel
from mmengine.optim import AmpOptimWrapper
from mmengine.runner import Runner


def create_palette(csv_filepath):
    color_to_class = {}
    with open(csv_filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            r, g, b = int(row['r']), int(row['g']), int(row['b'])
            color_to_class[(r, g, b)] = idx
    return color_to_class


class CamVid(VisionDataset):

    def __init__(self,
                 root,
                 img_folder,
                 mask_folder,
                 transform=None,
                 target_transform=None):
        super().__init__(
            root, transform=transform, target_transform=target_transform)
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.images = list(
            sorted(os.listdir(os.path.join(self.root, img_folder))))
        self.masks = list(
            sorted(os.listdir(os.path.join(self.root, mask_folder))))
        self.color_to_class = create_palette(
            os.path.join(self.root, 'class_dict.csv'))

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.img_folder, self.images[index])
        mask_path = os.path.join(self.root, self.mask_folder,
                                 self.masks[index])

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')  # Convert to RGB

        if self.transform is not None:
            img = self.transform(img)

        # Convert the RGB values to class indices
        mask = np.array(mask)
        mask = mask[:, :, 0] * 65536 + mask[:, :, 1] * 256 + mask[:, :, 2]
        labels = np.zeros_like(mask, dtype=np.int64)
        for color, class_index in self.color_to_class.items():
            rgb = color[0] * 65536 + color[1] * 256 + color[2]
            labels[mask == rgb] = class_index

        if self.target_transform is not None:
            labels = self.target_transform(labels)
        data_samples = dict(
            labels=labels, img_path=img_path, mask_path=mask_path)
        return img, data_samples

    def __len__(self):
        return len(self.images)


class MMDeeplabV3(BaseModel):

    def __init__(self, num_classes):
        super().__init__()
        self.deeplab = deeplabv3_resnet50(num_classes=num_classes)

    def forward(self, imgs, data_samples=None, mode='tensor'):
        x = self.deeplab(imgs)['out']
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, data_samples['labels'])}
        elif mode == 'predict':
            return x, data_samples


class IoU(BaseMetric):

    def process(self, data_batch, data_samples):
        preds, labels = data_samples[0], data_samples[1]['labels']
        preds = torch.argmax(preds, dim=1)
        intersect = (labels == preds).sum()
        union = (torch.logical_or(preds, labels)).sum()
        iou = (intersect / union).cpu()
        self.results.append(
            dict(batch_size=len(labels), iou=iou * len(labels)))

    def compute_metrics(self, results):
        total_iou = sum(result['iou'] for result in self.results)
        num_samples = sum(result['batch_size'] for result in self.results)
        return dict(iou=total_iou / num_samples)


class SegVisHook(Hook):

    def __init__(self, data_root, vis_num=1) -> None:
        super().__init__()
        self.vis_num = vis_num
        self.palette = create_palette(osp.join(data_root, 'class_dict.csv'))

    @master_only
    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch=None,
                       outputs=None) -> None:
        if batch_idx > self.vis_num:
            return

        preds, data_samples = outputs
        img_paths = data_samples['img_path']
        mask_paths = data_samples['mask_path']
        _, C, H, W = preds.shape
        preds = torch.argmax(preds, dim=1)
        for idx, (pred, img_path,
                  mask_path) in enumerate(zip(preds, img_paths, mask_paths)):
            pred_mask = np.zeros((H, W, 3), dtype=np.uint8)
            runner.visualizer.set_image(pred_mask)
            for color, class_id in self.palette.items():
                runner.visualizer.draw_binary_masks(
                    pred == class_id,
                    colors=[color],
                    alphas=1.0,
                )
            # Convert RGB to BGR
            pred_mask = runner.visualizer.get_image()[..., ::-1]
            saved_dir = osp.join(runner.log_dir, 'vis_data', str(idx))
            os.makedirs(saved_dir, exist_ok=True)

            shutil.copyfile(img_path,
                            osp.join(saved_dir, osp.basename(img_path)))
            shutil.copyfile(mask_path,
                            osp.join(saved_dir, osp.basename(mask_path)))
            cv2.imwrite(
                osp.join(saved_dir, f'pred_{osp.basename(img_path)}'),
                pred_mask)


def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    num_classes = 32  # Modify to actual number of categories.
    norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(**norm_cfg)])

    target_transform = transforms.Lambda(
        lambda x: torch.tensor(np.array(x), dtype=torch.long))

    train_set = CamVid(
        'data/CamVid',
        img_folder='train',
        mask_folder='train_labels',
        transform=transform,
        target_transform=target_transform)

    valid_set = CamVid(
        'data/CamVid',
        img_folder='val',
        mask_folder='val_labels',
        transform=transform,
        target_transform=target_transform)

    train_dataloader = dict(
        batch_size=3,
        dataset=train_set,
        sampler=dict(type='DefaultSampler', shuffle=True),
        collate_fn=dict(type='default_collate'))
    val_dataloader = dict(
        batch_size=3,
        dataset=valid_set,
        sampler=dict(type='DefaultSampler', shuffle=False),
        collate_fn=dict(type='default_collate'))

    runner = Runner(
        model=MMDeeplabV3(num_classes),
        work_dir='./work_dir',
        train_dataloader=train_dataloader,
        optim_wrapper=dict(
            type=AmpOptimWrapper, optimizer=dict(type=AdamW, lr=2e-4)),
        train_cfg=dict(by_epoch=True, max_epochs=10, val_interval=10),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        val_evaluator=dict(type=IoU),
        launcher=args.launcher,
        custom_hooks=[SegVisHook('data/CamVid')],
        default_hooks=dict(checkpoint=dict(type='CheckpointHook', interval=1)),
    )
    runner.train()


if __name__ == '__main__':
    main()
