# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 09:18:40 2018

@author: liangy
"""


import os
import cv2
import numpy as np
import sys
import time
import random


def crop_imgs(img):
    croped = []
    # Resize to 256 * 256 img
    resized_img = cv2.resize(img, (256, 256))
    # Crop 224 * 224 pieces
    crop_num = 4
    for _ in range(crop_num):
        start_point_y = random.randint(0, 32)
        start_point_x = random.randint(0, 32)
        croped_img = resized_img[start_point_y:start_point_y + 224,
                                 start_point_x:start_point_x + 224]
        croped.append(croped_img)
    return croped


def main(argv):
    output = 'cropedImages'
    if not os.path.exists(output):
        os.makedirs(output)
    styles = os.listdir(argv[1])
    for style in styles:
        if style != ".DS_Store":
            # Create the directory
            if not os.path.exists(output + "/" + style):
                os.makedirs(output + "/" + style)
            # Get imgs for each style
            imgs = os.listdir(argv[1] + "/" + style)
            for img in imgs:
                if img != ".DS_Store":
                    img_data = cv2.imread(argv[1] + "/" + style + "/" + img)
                    croped_imgs = crop_imgs(img_data)
                    for n, croped_img in enumerate(croped_imgs):
                        file_name = output + "/" + style + "/" + \
                            img[:-4] + "_" + str(n) + ".png"
                        cv2.imwrite(file_name, croped_img)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Program accepts the directory parameter!")
        sys.exit(1)

    print("start separation...")
    start_time = time.time()
    main(sys.argv)
    end_time = time.time()
    print('finished:' + str(end_time - start_time) + "s")
