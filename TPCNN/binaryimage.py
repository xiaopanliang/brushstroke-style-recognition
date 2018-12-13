# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 19:09:06 2018

@author: jpansh
"""

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
import colorsys
import random
#from threading import Thread
# import matplotlib.image as mpimg
import sort_pixels as sp
import matplotlib

input_height = 512  # 768
input_width = 512  # 1024
rate = 4

# Limit the size of the message to be within the max_size
def resize_content_img(img):
    return cv2.resize(img, dsize=(input_height, input_width), interpolation=cv2.INTER_AREA)


def read_image(path):
    _img_ = cv2.imread(path)
#    (thresh, _img_) = cv2.threshold(_img_, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return _img_.astype(np.float32)


def load_content_img(path):
    _img_ = read_image(path)
    _img_ = resize_content_img(_img_)
    # show_img = img.astype(np.uint8)
    # plt.imshow(show_img)
    # plt.show()
    return _img_

def main():
    base1 = 'output/step2/'
    base2 = 'output/merged/'
    output = 'output/color/'
    output2 = 'output/true/'
    folders = os.listdir(base2)

    for folder in folders:

        if folder != 'Photoalism':
            files = os.listdir(base2 + folder + '/')
            for num, file in enumerate(files):
                try:
                    img1 = load_content_img(base1 + folder+'/'+file)
                    img2 = np.zeros([512,512,3])
                    h_channel = matplotlib.colors.rgb_to_hsv(img1)[:,:,0]
                    re_h = sp.sort_pixels(h_channel)
                    img2[:,:,0] = re_h*256
                    img2[:,:,1]= 255
                    img2[:,:,2] = 255
                    img3 = matplotlib.colors.hsv_to_rgb(img2/256)*256
                    # img2 = np.uint8(img1)
                    # img2 = cv2.Canny(img2,100,200)
                    # img2 = load_content_img(base2 + folder+'/'+file.split(".")[0]+'.jpg')
                    # img3 = img2
                    # (thresh, img2) = cv2.threshold(img2, 50, 255, cv2.THRESH_BINARY)
                    # img2 = img1
                    # width,height = img2.shape
                    # img2 = img2/255*img1
                    # for x in range(width):
                    #     for y in range(height):
                    #         if img2[x,y] != 0:
                    #             img2[x,y] = random.randint(100,255)
                    #         else:
                    #             img2[x,y] = random.randint(0,20)
                    file_name = file
                    cv2.imwrite(output + folder + '/' + file_name,  img3)
                    # cv2.imwrite(output2 + folder + '/' + file_name, img3)
                    print('write ' + (output + folder + '/' + file_name))
                except:
                    print('abandon')


if __name__ == '__main__':
    print('starting making dataset...')
    main()
    print('finished making dataset')