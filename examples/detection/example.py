import argparse
import os

import PIL
import torch
from mmeval import COCODetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils import coco_file_to_dict, get_transform, json_to_dict

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
        label_boxes = []
        label_labels = []
        pred_boxes = []
        pred_scores = []
        pred_labels = []
        width = targets[1][0]['width']
        height = targets[1][0]['height']

        for target in targets[1]:
            label_boxes.append(target['boxes'])
            label_labels.append(target['labels'])

        for prediction in predictions:
            pred_boxes.append(prediction['boxes'].cpu())
            pred_scores.append(prediction['scores'].cpu())
            pred_labels.append(prediction['labels'].cpu())

        groundtruth = {
            'bboxes': torch.cat(label_boxes).cpu().numpy(),
            'labels': torch.cat(label_labels).cpu().numpy(),
            'width': width,
            'height': height
        }

        predictionS = {
            'bboxes': torch.cat(pred_boxes).cpu().numpy(),
            'scores': torch.cat(pred_scores).cpu().numpy(),
            'labels': torch.cat(pred_labels).cpu().numpy()
        }

        self.results.append({
            'groundtruth': groundtruth,
            'predictions': predictionS
        })

    def compute_metrics(self, results):
        groundtruth = [(item['predictions'], item['groundtruth'])
                       for item in results]
        fake_dataset_metas = {'classes': tuple(map(str, range(73)))}
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

# Convert human readable str label to int.
label_dict = coco_file_to_dict(
    'examples/detection/train/_annotations.coco.json')


class COCODataset:

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
        target['labels'] = torch.as_tensor(ann['labels'], dtype=torch.int64)
        target['img_id'] = torch.as_tensor(
            i)  # Apply any transforms to the data if required.
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.files)


# Collate image-target pairs into a tuple.
def collate_fn(batch):
    return tuple(zip(*batch))  # Create the DataLoaders from the Datasets.


def main():
    args = parse_args()

    # Train dataset.
    # Set train = True to apply the training image transforms.
    train_ds = COCODataset('examples/detection/train',
                           get_transform(train=True))  # Train dataset.
    val_ds = COCODataset('examples/detection/valid',
                         get_transform(train=False))  # Validation dataset.

    train_dl = dict(
        batch_size=args.batch_size,
        dataset=train_ds,
        sampler=dict(type='DefaultSampler', shuffle=True),
        collate_fn=collate_fn)

    val_dl = dict(
        batch_size=args.batch_size,
        dataset=val_ds,
        sampler=dict(type='DefaultSampler', shuffle=False),
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
