# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 09:18:40 2018

@author: liangy
"""


import os
import cv2
import time
import random
import numpy as np
import make_dataset

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def crop_imgs(old_img, style):
    croped = []
    img = np.zeros((256, 256,3))
    # Resize to 256 * 256 img
    for color in range(3):
        img[:,:,color] = cv2.resize(old_img[:,:,color], (256, 256))
    # if len(img.shape) == 3:
    #     img = rgb2gray(img)
    # Crop 224 * 224 pieces
    crop_num = 16
    if style == "Ink and wash painting":
        crop_num = 16
    elif style == "Photorealism":
        crop_num = 64
    for x_index in range(int(np.sqrt(crop_num))):
        for y_index in range(int(np.sqrt(crop_num))):
            start_point_y = y_index * 10
            start_point_x = x_index * 10
            cropped_img = img[start_point_y:start_point_y + 224,start_point_x:start_point_x + 224,:]
            croped.append(cropped_img)
    return croped


def start_croping(imgs, input_dir, output_dir, style):
    for img in imgs:
        if img != ".DS_Store":
            img_data = cv2.imread(input_dir + "/" + img)
            if img_data is None:
                continue
            cropped_imgs = crop_imgs(img_data, style)
            for n, cropped_img in enumerate(cropped_imgs):
                file_name = output_dir + "/" + \
                    img[:-4] + "_" + str(n) + ".jpg"
                cv2.imwrite(file_name, cropped_img)


def main(Input, output):
    if not os.path.exists(output):
        os.makedirs(output)
    styles = os.listdir(Input)
    for style in styles:
        if not os.path.exists(output + "/" + style):
            if style != ".DS_Store":
                print("processing style:" + style)
                # Create the directory
                if not os.path.exists(output + "/" + style):
                    os.makedirs(output + "/" + style)
                # Get imgs for each style
                imgs = os.listdir(Input + "/" + style)
                # Shuffle imgs
                random.shuffle(imgs)
                split_index = int(len(imgs) * 0.8)
                train_imgs = imgs[:split_index]
                eval_imgs = imgs[split_index:]
                train_set_output_dir = output + "/" + style + "/train_set"
                eval_set_output_dir = output + "/" + style + "/eval_set"
                # Create test set folder and folder for storing imgs
                if not os.path.exists(train_set_output_dir):
                    os.makedirs(train_set_output_dir)
                if not os.path.exists(eval_set_output_dir):
                    os.makedirs(eval_set_output_dir)
                input_dir = Input + "/" + style
                start_croping(train_imgs, input_dir, train_set_output_dir, style)
                start_croping(eval_imgs, input_dir, eval_set_output_dir, style)


if __name__ == '__main__':
    print("start separation...")
    start_time = time.time()
    main('old_DataBase','old_Seperation')
    make_dataset.main('Seperation')
    end_time = time.time()
    print('finished:' + str(end_time - start_time) + "s")
