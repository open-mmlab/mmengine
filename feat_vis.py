from mmengine.visualization import Visualizer
from torchvision.models import resnet50
import cv2
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np


def preprocess_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


model = resnet50(pretrained=True)


def _forward_impl(x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x1 = model.layer1(x)
    x2 = model.layer2(x1)
    x3 = model.layer3(x2)
    x4 = model.layer4(x3)
    return x4


model._forward_impl = _forward_impl


if __name__ == '__main__':
    img_path = '/home/PJLAB/huanghaian/Desktop/both.png'
    rgb_img1 = cv2.imread(img_path, 1)[:, :, ::-1]
    print(rgb_img1.shape)
    rgb_img = np.float32(rgb_img1) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    output = model(input_tensor)[0]
    print(output.shape)
    local_visualizer = Visualizer()
    featmap = local_visualizer.draw_featmap(output, image=rgb_img1, mode=None, topk=10, arrangement=(4,3), resize_hw=(rgb_img.shape[0], rgb_img.shape[1]))
    cv2.namedWindow('featmap', 0)
    cv2.imshow('featmap', featmap)
    cv2.waitKey(0)
