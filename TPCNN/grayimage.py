# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 10:35:55 2018

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

# Limit the size of the message to be within the max_size
def resize_content_img(img):
    return cv2.resize(img, dsize=(input_height, input_width), interpolation=cv2.INTER_AREA)


def read_image(path):
    _img_ = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    return _img_.astype(np.float32)


def load_content_img(path):
    _img_ = read_image(path)
    _img_ = resize_content_img(_img_)
    # show_img = img.astype(np.uint8)
    # plt.imshow(show_img)
    # plt.show()
    return _img_

def main():
    base = 'output/step2/'
    output = 'output/Gray/'
    folders = os.listdir(output)

    for folder in folders:
        if folder != '.DS_Store':
            files = os.listdir(base + folder + '/')
            for num, file in enumerate(files):
                try:
                    img = load_content_img(base+folder+'/'+file)

                    file_name = file
                    cv2.imwrite(output + folder + '/' + file_name,  img)
                    print('write ' + (output + folder + '/' + file_name))
                except:
                    print('abandon')

if __name__ == '__main__':
    print('starting making dataset...')
    main()
    print('finished making dataset')