import json
import os


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
