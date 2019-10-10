import cv2
import os
import numpy as np
from utils import *


if __name__ == "__main__":
    image_dir = "./../data_raw/train_large"
    mask_dir = "./../data_raw/mask_large"
    data_dir = "./../data/train_large/place2"

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    image_names = path_gen(image_dir)
    mask_names = path_gen(mask_dir)

    n = len(mask_names)

    for item in image_names:
        print(item)
        img_pth = os.path.join(image_dir, item+".jpg")
        rst_pth = os.path.join(data_dir, item+".png")

        img = cv2.imread(img_pth)
        img = cv2.resize(img, (256, 256))
        rate = 0
        while rate < 0.1:
            k = np.random.randint(0, n)
            msk_name = mask_names[k]
            msk_pth = os.path.join(mask_dir, msk_name+".png")
            msk = cv2.imread(msk_pth)
            msk_n = msk // 255
            rate = rate_compute(msk_n)
        print(rate)

        data = np.concatenate((img, msk, img*msk_n), axis=1)
        cv2.imwrite(rst_pth, data)
