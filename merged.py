# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 21:27:24 2018

@author: jpansh
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 09:18:40 2018

@author: jpansh
"""

import os
import cv2
import numpy as np
#from threading import Thread
# import matplotlib.image as mpimg

input_height = 512  # 768
input_width = 512  # 1024
rate = 4
channel = 3

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
    base = 'output/step44/'
    output = 'output/merged/'
    folders = os.listdir(base)

    for folder in folders:
        if folder != '.DS_Store':
            files = os.listdir(base + folder + '/')
            new_img = np.zeros((input_height,input_width,channel))
            count = 0
            for num, file in enumerate(files):
                img = load_content_img(base+folder+'/'+file)
                new_img[:,:,0] = new_img[:,:,0]+img[:,:,0]
                new_img[:,:,1] = new_img[:,:,1]+img[:,:,1]
                new_img[:,:,2] = new_img[:,:,2]+img[:,:,2]
                count = count+ 1
            new_img[:,:,0] = new_img[:,:,0]/count
            new_img[:,:,1] = new_img[:,:,1]/count
            new_img[:,:,2] = new_img[:,:,2]/count
            cv2.imwrite(output + folder + '/cool.jpg' ,  cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
            print('finished')


if __name__ == '__main__':
    print('starting making dataset...')
    main()
    print('finished making dataset')