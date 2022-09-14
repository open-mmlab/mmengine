# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np


def imfrombytes(content):
    img_np = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    return img
