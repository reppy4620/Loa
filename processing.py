import os
import cv2
import numpy as np

import torch
import torchvision.transforms as T
import torchvision.utils as vutils

from PIL import Image


transforms = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))
])


def color2line(file_name):
    neighbor_hood_8 = np.array([[1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1]],
                               np.uint8)
    img = cv2.imread(os.path.join('uploads', file_name), cv2.IMREAD_GRAYSCALE)
    img_dilate = cv2.dilate(img, neighbor_hood_8, iterations=1)
    img = cv2.absdiff(img, img_dilate)
    img = cv2.bitwise_not(img)
    cv2.imwrite(os.path.join('uploads', file_name), img)
    return Image.fromarray(img)


def line2color(file_name, net, device):
    path = os.path.join('uploads', file_name)
    img = Image.open(path).convert('L')
    img = transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        color = net(img)
    vutils.save_image(
        color,
        path,
        normalize=True
    )


def color2color(file_name, net, device):
    img = color2line(file_name)
    img = transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        color = net(img)
    vutils.save_image(
        color,
        os.path.join('uploads', file_name),
        normalize=True
    )
