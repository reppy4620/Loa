import os
import torchvision.transforms as T
import torchvision.utils as vutils
import numpy as np
import cv2

from PIL import Image


transforms = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))
])


def color_to_line(file_name):
    neighbor_hood_8 = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]],
                               np.uint8)
    img = cv2.imread(os.path.join('uploads', file_name), 0)
    img_dilate = cv2.dilate(img, neighbor_hood_8, iterations=1)
    img_diff = cv2.absdiff(img, img_dilate)
    img_diff_not = cv2.bitwise_not(img_diff)
    cv2.imwrite(os.path.join('uploads', file_name), img_diff_not)


def line2color(file_name, net, device):
    path = os.path.join('uploads', file_name)
    img = Image.open(path).convert('L')
    color = net(transforms(img).unsqueeze(0).to(device))
    vutils.save_image(
        color,
        path,
        normalize=True
    )
