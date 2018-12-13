# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 09:18:40 2018

@author: jpansh
"""

import os
import cv2
import numpy as np
#from threading import Thread
#import matplotlib.image as mpimg

input_height = 512  # 768
input_width = 512  # 1024
rate = 4
sub_rate = 2

# Limit the size of the message to be within the max_size
def resize_content_img(img):
    return cv2.resize(img, dsize=(input_height, input_width), interpolation=cv2.INTER_AREA)


def read_image(path):
    _img_ = cv2.imread(path)
    return _img_.astype(np.float32)


def load_content_img(path):
    _img_ = read_image(path)
    _img_ = resize_content_img(_img_)
    # show_img = img.astype(np.uint8)
    # plt.imshow(show_img)
    # plt.show()
    return _img_

def main():
    base = 'output/color/'
    output = 'output/color_sepe/'
    folders = os.listdir(base)

    for folder in folders:
        if folder != 'Inand wash painting' :
            files = os.listdir(base + folder + '/')
            for num, file in enumerate(files):
                img = load_content_img(base+folder+'/'+file)
#                img *= (255.0/img.max())
                height,width,_ = img.shape
                for index in range(1,rate+1):
                    file_name = file.split('.')[0] + str(index) + '.jpg'
                    if index == 1:
                        new_img = img[0:int(height/32*31),0:int(width/32*31),:]
                    elif index == 2:
                        new_img = img[int(height/32*1):,:int(width/32*31),:]
                    elif index == 3:
                       new_img = img[:int(height/32*31),int(width/32*1):,:]
                    elif index == 4:
                        new_img = img[int(height/32*1):,int(width/32*1):,:]
                    cv2.imwrite(output + folder + '/' + file_name,  new_img)
                    print('write ' + (output + folder + '/' + file_name))


if __name__ == '__main__':
    print('starting making dataset...')
    main()
    print('finished making dataset')