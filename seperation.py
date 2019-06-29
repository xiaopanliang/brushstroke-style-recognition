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


def crop_imgs(img, style):
    croped = []

    ratio = 15 / 16
    height, width, _ = img.shape
    
    crop_height = int(height * ratio)
    crop_width = int(width * ratio)
    
    cur_hei_space = height - crop_height
    cur_wi_space = width - crop_width

    # Crop 224 * 224 pieces
    crop_num = 4
    if style == "Ink and wash painting":
        crop_num = 6
    elif style == "Photorealism":
        crop_num = 16
    for _ in range(crop_num):
        start_point_y = random.randint(0, cur_hei_space)
        start_point_x = random.randint(0, cur_wi_space)
        croped_img = img[start_point_y:start_point_y + crop_height,
                                 start_point_x:start_point_x + crop_width]
        croped.append(croped_img)
    
    return croped


def start_croping(imgs, input_dir, output_dir, style):
    for img in imgs:
        if img != ".DS_Store":
            img_data = cv2.imread(input_dir + "/" + img)
            croped_imgs = crop_imgs(img_data, style)
            for n, croped_img in enumerate(croped_imgs):
                file_name = output_dir + "/" + \
                    img[:-4] + "_" + str(n) + ".png"
                cv2.imwrite(file_name, croped_img)


def main(argv):
    output = argv[2]
    if not os.path.exists(output):
        os.makedirs(output)
    input_p = argv[1]
    style = input_p[input_p.rindex('/')+1:]
    if not os.path.exists(output + "/" + style):
        if style != ".DS_Store":
            print("processing style:" + style)
            # Create the directory
            if not os.path.exists(output + "/" + style):
                os.makedirs(output + "/" + style)
            # Get imgs for each style
            imgs = os.listdir(input_p)
            # Shuffle imgs
            random.shuffle(imgs)
            split_index = int(len(imgs) * 0.9)
            train_imgs = imgs[:split_index]
            eval_imgs = imgs[split_index:]
            train_set_output_dir = output + "/" + style + "/train_set"
            eval_set_output_dir = output + "/" + style + "/eval_set"
            # Create test set folder and folder for storing imgs
            if not os.path.exists(train_set_output_dir):
                os.makedirs(train_set_output_dir)
            if not os.path.exists(eval_set_output_dir):
                os.makedirs(eval_set_output_dir)

            start_croping(train_imgs, input_p, train_set_output_dir, style)
            start_croping(eval_imgs, input_p, eval_set_output_dir, style)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Program accepts the style dir parameter and the output directory!")
        sys.exit(1)

    print("start separation...")
    start_time = time.time()
    main(sys.argv)
    end_time = time.time()
    print('finished:' + str(end_time - start_time) + "s")
