import os
import torchvision.transforms as T
import torchvision.utils as vutils
import numpy as np
import cv2
import torch

from PIL import Image


def color_to_line(file_name):
    neighbor_hood_8 = np.array([[1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1]],
                               np.uint8)
    img = cv2.imread(os.path.join('uploads', file_name), 0)
    img_dilate = cv2.dilate(img, neighbor_hood_8, iterations=1)
    img_diff = cv2.absdiff(img, img_dilate)
    img_diff_not = cv2.bitwise_not(img_diff)
    cv2.imwrite(os.path.join('uploads', file_name), img_diff_not)


def line2color(file_name, net, device):
    transforms = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5))
    ])
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
