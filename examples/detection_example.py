import argparse
import os
import xml.etree.ElementTree as ET  # Connect to the GPU if one exists.

import numpy as np
import PIL
import torch
import torchvision
import torchvision.transforms.functional as F
import torchvision.transforms.transforms as T
# from mmeval import MeanIoU
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner


class MMFasterRCNN(BaseModel):

    def __init__(self, num_classes=3, feature_extraction=True):
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)

        if feature_extraction:
            for p in self.model.parameters():
                p.requires_grad = False

            in_feats = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_feats, num_classes)

    def forward(self, images, targets, mode):
        if mode == 'loss':
            output = self.model(images, targets)
            return output
        elif mode == 'predict':
            predictions = self.model(images)
            return predictions


class Accuracy(BaseMetric):

    def process(self, targets, predictions):
        # miou = MeanIoU(num_classes=11)
        # print(targets)
        # print(predictions)
        label = []
        pred = []
        labelS = []
        preds = []
        for target in targets[1]:
            boxes = target['boxes']
            target_labels = [target['labels'].item()]
            # print("Target Labels:", target_labels)

            for i in range(len(boxes)):
                x_min, y_min, x_max, y_max = boxes[i].tolist()
                label.append([target_labels[i], x_min, y_min])
        for prediction in predictions:
            boxes = prediction['boxes']
            scores = prediction['scores']
            labels = prediction['labels']
            score_threshold = 0.8
            nms_iou_threshold = 0.2
            # print(f"boxes: {boxes}\nscores: {scores}\nlabels: {labels}\n")
            if score_threshold is not None:
                want = scores > score_threshold
                boxes = boxes[want]
                scores = scores[want]
                labels = labels[want]
            if nms_iou_threshold is not None:
                want = torchvision.ops.nms(
                    boxes=boxes,
                    scores=scores,
                    iou_threshold=nms_iou_threshold)
                boxes = boxes[want]
                scores = scores[want]
                labels = labels[want]
            # print(f"boxes: {boxes}\nscores: {scores}\nlabels: {labels}\n")
            # boxes = prediction['boxes']
            prediction_labels = [labels.cpu().numpy()]
            prediction_labels = prediction_labels[0].tolist()
            # print("Prediction Labels:", prediction_labels)
            for i in range(len(boxes)):
                x_min, y_min, x_max, y_max = boxes[i].tolist()
                pred.append([prediction_labels[i], x_min, y_min])

        labelS.append(label)
        preds.append(pred)
        labelS = np.asarray(labelS)
        preds = np.asarray(preds)
        print(f'LABELS:{labelS}, PREDS: {preds}')

        # miou(labels, preds)
        # print(miou)
        # print(f"TARGETS: {targets}\n PREDICTIONS: {predictions}\n")
    def compute_metrics(self, results):
        pass


def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection')
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


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def xml_to_dict(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return {
        'filename': xml_path,
        'image_width': int(root.find('./size/width').text),
        'image_height': int(root.find('./size/height').text),
        'image_channels': int(root.find('./size/depth').text),
        'label': root.find('./object/name').text,
        'x1': int(root.find('./object/bndbox/xmin').text),
        'y1': int(root.find('./object/bndbox/ymin').text),
        'x2': int(root.find('./object/bndbox/xmax').text),
        'y2': int(root.find('./object/bndbox/ymax').text)
    }


def file_to_dict(file_path):
    data_dict = {}
    with open(file_path) as file:
        for index, line in enumerate(file):
            value = line.strip()
            data_dict[index] = value
    return data_dict


def reverse_dict(dictionary):
    return {value: key for key, value in dictionary.items()}


# Convert human readable str label to int.
label_dict = file_to_dict('data/tiny_motorbike_coco/tiny_motorbike/labels.txt')
# Convert label int to human readable str.
reverse_label_dict = reverse_dict(label_dict)


class vehicleDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.files = sorted(
            os.listdir('data/tiny_motorbike_coco/tiny_motorbike/images'))
        for i in range(len(self.files)):
            self.files[i] = self.files[i].split('.')[0]
            self.label_dict = label_dict

    def __getitem__(self, i):
        img = PIL.Image.open(
            os.path.join(self.root,
                         'images/' + self.files[i] + '.jpg')).convert(
                             'RGB')  # Load annotation file from the hard disc.
        ann = xml_to_dict(
            os.path.join(self.root, 'annotations/' + self.files[i] +
                         '.xml'))  # The target is given as a dict.
        target = {}
        target['boxes'] = torch.as_tensor(
            [[ann['x1'], ann['y1'], ann['x2'], ann['y2']]],
            dtype=torch.float32)
        target['labels'] = torch.as_tensor([reverse_label_dict[ann['label']]],
                                           dtype=torch.int64)
        target['image_id'] = torch.as_tensor(
            i)  # Apply any transforms to the data if required.
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.files)


class Compose:

    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(torch.nn.Module):

    def forward(self, image, target=None):
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):

    def forward(self, image, target=None):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                width, _ = F.get_image_size(image)
                target['boxes'][:, [0, 2]] = width - target['boxes'][:, [2, 0]]
        return image, target


def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


# Collate image-target pairs into a tuple.
def collate_fn(batch):
    return tuple(zip(*batch))  # Create the DataLoaders from the Datasets.


def main():
    args = parse_args()
    # norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])

    # Train dataset.
    # Set train = True to apply the training image transforms.
    train_ds = vehicleDataset('data/tiny_motorbike_coco/tiny_motorbike/',
                              get_transform(train=True))  # Validation dataset.
    val_ds = vehicleDataset('data/tiny_motorbike_coco/tiny_motorbike/',
                            get_transform(train=False))  # Test dataset.
    test_ds = vehicleDataset('data/tiny_motorbike_coco/tiny_motorbike/',
                             get_transform(train=False))

    # Randomly shuffle all the data.
    indices = torch.randperm(len(train_ds)).tolist()
    # We split the entire data into 80/20 train-test splits. We further
    # split the train set into 80/20 train-validation splits.
    # Train dataset: 64% of the entire data, or 80% of 80%.
    train_ds = torch.utils.data.Subset(
        train_ds, indices[:int(len(indices) * 0.64)]
    )  # Validation dataset: 16% of the entire data, or 20% of 80%.
    val_ds = torch.utils.data.Subset(
        val_ds,
        indices[int(len(indices) *
                    0.64):int(len(indices) *
                              0.8)])  # Test dataset: 20% of the entire data.
    test_ds = torch.utils.data.Subset(test_ds,
                                      indices[int(len(indices) * 0.8):])

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

    runner = Runner(
        model=MMFasterRCNN(11),
        work_dir='./work_dir',
        train_dataloader=train_dl,
        optim_wrapper=dict(
            optimizer=dict(
                type=torch.optim.SGD,
                lr=0.005,
                momentum=0.9,
                weight_decay=0.0005)),
        train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
        val_dataloader=val_dl,
        val_cfg=dict(),
        val_evaluator=dict(type=Accuracy),
        launcher=args.launcher,
    )
    runner.train()


if __name__ == '__main__':
    main()
