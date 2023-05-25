import argparse
import json
import os
import xml.etree.ElementTree as ET  # Connect to the GPU if one exists.

import numpy as np
import PIL
import torch
import torchvision.transforms.functional as F
import torchvision.transforms.transforms as T
from mmeval import COCODetection
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
        label = []
        pred = []
        score = []
        groundtruth = {}
        predictionS = {}
        labelS = []
        pred_labels = []
        for target in targets[1]:
            boxes = target['boxes']
            width = target['width']
            height = target['height']
            labels = target['labels'].cpu().numpy()
            for i in range(len(boxes)):
                x_min, y_min, x_max, y_max = boxes[i].tolist()
                label.append([x_min, y_min, x_max, y_max])
                labelS.append(labels[i].tolist())
        for prediction in predictions:
            boxes = prediction['boxes']
            scores = prediction['scores'].cpu().numpy()
            pred_label = prediction['labels'].cpu().numpy()
            for i in range(len(boxes)):
                x_min, y_min, x_max, y_max = boxes[i].tolist()
                pred_labels.append(pred_label[i].tolist())
                score.append(scores[i].tolist())
                pred.append([x_min, y_min, x_max, y_max])

        label = np.array(label)
        labelS = np.array(labelS)
        pred = np.array(pred)
        score = np.array(score)
        pred_labels = np.array(pred_labels)

        groundtruth['bboxes'] = label
        groundtruth['labels'] = labelS
        groundtruth['width'] = width
        groundtruth['height'] = height

        predictionS['bboxes'] = pred
        predictionS['scores'] = score
        predictionS['labels'] = pred_labels

        self.results.append({
            'groundtruth': groundtruth,
            'predictions': predictionS
        })

    def compute_metrics(self, results):
        groundtruth = [(item['predictions'], item['groundtruth'])
                       for item in results]
        fake_dataset_metas = {'classes': tuple([str(i) for i in range(73)])}
        coco_det_metric = COCODetection(
            dataset_meta=fake_dataset_metas, metric=['bbox'])
        r = coco_det_metric.compute_metric(groundtruth)
        return r


def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--batch_size', type=int, default=8, help='batch size for training')

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


def json_to_dict(file_path):
    image_name = os.path.basename(file_path)
    json_path = os.path.dirname(file_path)

    with open(json_path) as file:
        json_data = json.load(file)

    images = json_data.get('images', [])
    annotations = json_data.get('annotations', [])

    image_info = next(
        (image for image in images if image['file_name'] == image_name), None)

    if image_info is None:
        return None

    image_id = image_info['id']
    height = image_info['height']
    width = image_info['width']

    boxes = []
    labels = []

    for ann in annotations:
        if ann['image_id'] == image_id:
            x, y, width_box, height_box = ann['bbox']
            xmin = x
            ymin = y
            xmax = x + width_box
            ymax = y + height_box
            boxes.append([xmin, ymin, xmax, ymax])
            category_id = ann['category_id']
            labels.append(category_id)

    if len(boxes) == 0:
        boxes.append([1, 1, 2, 2])
        labels.append(73)

    return {
        'image_width': int(width),
        'image_height': int(height),
        'image_channels': 3,
        'labels': labels,
        'boxes': boxes
    }


def file_to_dict(file_path):
    data_dict = {}
    with open(file_path) as file:
        for index, line in enumerate(file):
            value = line.strip()
            data_dict[index] = value
    return data_dict


def coco_file_to_dict(file_path):
    with open(file_path) as file:
        json_data = json.load(file)

    categories = json_data.get('categories', [])

    data_dict = {category['name']: category['id'] for category in categories}
    data_dict['None'] = 73

    return data_dict


def reverse_dict(dictionary):
    return {value: key for key, value in dictionary.items()}


# Convert human readable str label to int.
label_dict = coco_file_to_dict(
    'examples/detection/train/_annotations.coco.json')
print(label_dict)
# Convert label int to human readable str.
reverse_label_dict = reverse_dict(label_dict)


class vehicleDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.files = sorted(os.listdir(self.root))
        for i in range(len(self.files)):
            self.files[i] = self.files[i].split('.jpg')[0]
            self.label_dict = label_dict
        print(self.files)
        self.files.remove('_annotations.coco.json')

    def __getitem__(self, i):
        img = PIL.Image.open(os.path.join(
            self.root, self.files[i] + '.jpg')).convert(
                'RGB')  # Load annotation file from the hard disc.
        ann = json_to_dict(
            os.path.join(self.root, '_annotations.coco.json/' + self.files[i] +
                         '.jpg'))  # The target is given as a dict.

        target = {}
        target['width'] = ann['image_width']
        target['height'] = ann['image_height']
        target['boxes'] = torch.as_tensor(ann['boxes'], dtype=torch.float32)
        # labels=[label_dict[label] for label in ann['labels']]
        target['labels'] = torch.as_tensor(ann['labels'], dtype=torch.int64)
        target['img_id'] = torch.as_tensor(
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

    # Train dataset.
    # Set train = True to apply the training image transforms.
    train_ds = vehicleDataset('examples/detection/train',
                              get_transform(train=True))  # Validation dataset.
    val_ds = vehicleDataset('examples/detection/valid',
                            get_transform(train=False))  # Valid dataset.

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn)
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn)

    runner = Runner(
        model=MMFasterRCNN(74),
        work_dir='./work_dir',
        train_dataloader=train_dl,
        optim_wrapper=dict(
            optimizer=dict(
                type=torch.optim.SGD,
                lr=0.005,
                momentum=0.9,
                weight_decay=0.0005)),
        train_cfg=dict(by_epoch=True, max_epochs=20, val_interval=1),
        val_dataloader=val_dl,
        val_cfg=dict(),
        val_evaluator=dict(type=Accuracy),
        launcher=args.launcher,
    )
    runner.train()


if __name__ == '__main__':
    main()
