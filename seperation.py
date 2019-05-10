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


possible_styles = ["Baroque", "Expressionism", "Impressionism", "Pointillism", "Romanticism", "Ukiyo_e"]


def crop_imgs(img):
    croped = []
    # Resize to 256 * 256 img
    resized_img = cv2.resize(img, (256, 256))
    # Crop 224 * 224 pieces
    crop_num = 16
    for _ in range(crop_num):
        start_point_y = random.randint(0, 32)
        start_point_x = random.randint(0, 32)
        croped_img = resized_img[start_point_y:start_point_y + 224,
                                 start_point_x:start_point_x + 224]
        croped.append(croped_img)
    return croped


def start_croping(imgs, input_dir, output_dir):
    for img in imgs:
        if img != ".DS_Store":
            img_data = cv2.imread(input_dir + "/" + img)
            croped_imgs = crop_imgs(img_data)
            for n, croped_img in enumerate(croped_imgs):
                file_name = output_dir + "/" + \
                    img[:-4] + "_" + str(n) + ".png"
                cv2.imwrite(file_name, croped_img)


def main(argv):
    output = argv[2]
    if not os.path.exists(output):
        os.makedirs(output)
    styles = os.listdir(argv[1])
    for style in styles:
        if not os.path.exists(output + "/" + style):
            if style != ".DS_Store" and style in possible_styles:
                print("processing style:" + style)
                # Create the directory
                if not os.path.exists(output + "/" + style):
                    os.makedirs(output + "/" + style)
                # Get imgs for each style
                imgs = os.listdir(argv[1] + "/" + style)
                # Shuffle imgs
                random.shuffle(imgs)
                train_imgs = imgs[:450]
                eval_imgs = imgs[450:500]
                train_set_output_dir = output + "/" + style + "/train_set"
                eval_set_output_dir = output + "/" + style + "/eval_set"
                # Create test set folder and folder for storing imgs
                if not os.path.exists(train_set_output_dir):
                    os.makedirs(train_set_output_dir)
                if not os.path.exists(eval_set_output_dir):
                    os.makedirs(eval_set_output_dir)
                input_dir = argv[1] + "/" + style
                start_croping(train_imgs, input_dir, train_set_output_dir)
                start_croping(eval_imgs, input_dir, eval_set_output_dir)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Program accepts the directory parameter and the output directory!")
        sys.exit(1)

    print("start separation...")
    start_time = time.time()
    main(sys.argv)
    end_time = time.time()
    print('finished:' + str(end_time - start_time) + "s")
