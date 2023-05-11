# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import random

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from torch.optim import SGD
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import \
    MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.ops import box_convert
from torchvision.transforms import functional as F

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner


class MMMaskRCNN(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights)

    def forward(self, images, targets, mode):
        if mode == 'loss':
            print(f'TARGET: {targets}')
            loss_dict = self.model(images, targets)
            return loss_dict
        elif mode == 'predict':
            predictions = self.model(images)
            return predictions, targets


class Accuracy(BaseMetric):

    def __init__(self, iou_threshold=0.5):
        super().__init__()
        self.iou_threshold = iou_threshold

    def process(self, data_batch, data_samples):
        predictions, targets = data_samples
        batch_size = targets.size(0)

        # Compute IoU for each prediction and target
        ious = self.calculate_iou(predictions, targets)

        # Count correct predictions based on IoU threshold
        correct = (ious > self.iou_threshold).sum().item()

        self.results.append({'batch_size': batch_size, 'correct': correct})

    def calculate_iou(self, predictions, targets):
        # Compute the intersection and union areas
        intersection = torch.min(predictions[:, 2:], targets[:, 2:]).clamp(0)
        # union = torch.max(predictions[:, 2:], targets[:, 2:]).clamp(0)

        # Calculate areas
        pred_area = (predictions[:, 2] - predictions[:, 0]) * (
            predictions[:, 3] - predictions[:, 1])
        target_area = (targets[:, 2] - targets[:, 0]) * (
            targets[:, 3] - targets[:, 1])

        # Calculate IoU
        iou = intersection.prod(dim=1) / (
            pred_area + target_area - intersection.prod(dim=1) + 1e-6)

        return iou

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        accuracy = 100 * total_correct / total_size
        return {'accuracy': accuracy}


def filterDataset(folder, classes=None, mode='train'):
    # initialize COCO api for instance annotations
    annFile = f'{folder}/{mode}/_annotations.coco.json'
    coco = COCO(annFile)

    images = []
    if classes is not None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given categories
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)

    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)

    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    random.shuffle(unique_images)
    dataset_size = len(unique_images)

    return unique_images, dataset_size, coco


def dataGeneratorCoco(images,
                      annFile,
                      folder,
                      input_image_size=(224, 224),
                      batch_size=32,
                      type='train'):
    img_folder = f'{folder}/{type}'
    coco = COCO(annotation_file=annFile)
    dataset_size = len(images)

    # print(f"imgObjL {images[0]}\n")
    batches = []
    c = 0
    for c in range(0, dataset_size, batch_size):
        img = []
        targets = []

        for i in range(c, c + batch_size):
            imageObj = images[i]
            # Retrieve Image
            # print( f'IMAGE: {imageObj}')
            img_path = f'{img_folder}/{imageObj["file_name"]}'
            # print(f"img_path: {img_path}")
            img_data = Image.open(img_path).convert('RGB')
            img_data = F.resize(img_data, input_image_size)
            img_tensor = F.to_tensor(img_data)
            img.append(img_tensor)

            # Retrieve Object Annotations
            annIds = coco.getAnnIds(imgIds=imageObj['id'], iscrowd=None)
            annotations = coco.loadAnns(annIds)

            # Skip image if there are no annotations
            if len(annotations) == 0:
                continue

            # Convert bounding box coordinates to tensor
            boxes = [ann['bbox'] for ann in annotations]
            # print(f"boxes: {boxes}\n")
            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = box_convert(boxes, in_fmt='xywh', out_fmt='xyxy')
            # boxes = torch.tensor(boxes, dtype=torch.float32)

            # Convert class labels to tensor
            labels = [ann['category_id'] for ann in annotations]
            labels = torch.tensor(labels, dtype=torch.int64)

            # Create target dictionary
            target = {'boxes': boxes, 'labels': labels}
            targets.append(target)

        c += batch_size
        if c + batch_size >= dataset_size:
            c = 0
            random.shuffle(images)

        batches.append((torch.stack(img), targets))

    return batches


def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--batch_size', type=int, default=32, help='batch size for training')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
    folder = 'data/COCO128'
    train_dataset = torchvision.datasets.CocoDetection(
        root=f'{folder}/train',
        annFile=f'{folder}/train/_annotations.coco.json',
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg)
        ]))
    val_dataset = torchvision.datasets.CocoDetection(
        root=f'{folder}/valid',
        annFile=f'{folder}/valid/_annotations.coco.json',
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(**norm_cfg)]))
    print('_____________________________________________________________--')
    trainImages, _, _ = filterDataset(folder, mode='train')
    print(f'trainImages: {trainImages[0]}')
    valImages = filterDataset(folder, mode='valid')
    train_dataloader = dict(
        dataset=train_dataset,
        batch_size=32,
        sampler=dict(type='DefaultSampler', shuffle=False),
        collate_fn=lambda batch: dataGeneratorCoco(
            trainImages,
            f'{folder}/train/_annotations.coco.json',
            folder,
            type='train'))
    val_dataloader = dict(
        dataset=val_dataset,
        batch_size=32,
        sampler=dict(type='DefaultSampler', shuffle=False),
        collate_fn=lambda batch: dataGeneratorCoco(
            valImages,
            f'{folder}/valid/_annotations.coco.json',
            folder,
            type='valid'))

    runner = Runner(
        model=MMMaskRCNN(),
        work_dir='./work_dir',
        train_dataloader=train_dataloader,
        optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
        train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        val_evaluator=dict(type=Accuracy),
        launcher=args.launcher,
    )
    runner.train()


if __name__ == '__main__':
    main()
