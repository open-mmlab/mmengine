import json
import os

import torch
import torchvision.transforms.functional as F
import torchvision.transforms.transforms as T


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


def coco_file_to_dict(file_path):
    with open(file_path) as file:
        json_data = json.load(file)

    categories = json_data.get('categories', [])

    data_dict = {category['name']: category['id'] for category in categories}
    data_dict['None'] = 73

    return data_dict


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
