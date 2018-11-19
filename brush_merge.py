# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 00:50:58 2018

@author: jpansh
"""

import os
import numpy as np
import cv2
import collections
# import matplotlib.pyplot as plt
# import math

input_height = 1024  # 768
input_width = 1024  # 1024
total_height =  input_height*4
total_width = input_width*4

def resize_content_img(img):
    return cv2.resize(img, dsize=(input_height, input_width), interpolation=cv2.INTER_AREA)


def read_image(path):
#    _img_ = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    _img_ = cv2.imread(path)
    return _img_
    
def load_content_img(path):
    _img_ = read_image(path)
    _img_ = resize_content_img(_img_)
    # show_img = img.astype(np.uint8)
    # plt.imshow(show_img)
    # plt.show()
    return _img_
    
def merge_back(_dir_, output):
    files = os.listdir(_dir_)
    files.sort()
    groups = []
    tmp_name = files[0][:-5]
    tmp_group = []
    for file in files:
        if file[:-5] == tmp_name:
            # same img
            file = file[:-5] + '_' + file[-5:]
            tmp_group.append(file)
        elif file[:-6] == tmp_name:
            # same img
            file = file[:-6] + '_' + file[-6:]
            tmp_group.append(file)
        else:
            # another image
            # clear the array
            groups.append(tmp_group)
            tmp_group = []
            tmp_name = file[:-5]
            file = file[:-5] + '_' + file[-5:]
            tmp_group.append(file)
    if not os.path.isdir(output):
        os.mkdir(output)
    # Each group contain 16 pieces of the original image
    for num, group in enumerate(groups):
        try:
            img_map = {}
            for file in group:
                file_num = int(file.split('_')[-1][:-4])
                img_map[file_num] = file
            for i in range(16):
                if i not in img_map:
                    img_map[i] = '0'
            img_map = collections.OrderedDict(sorted(img_map.items()))
            img_data = []
            sizes = []
            name_formed = False
            file_name = ''
            for _, img_name in img_map.items():
                try:
                    if img_name != '0':
                        index = img_name.rfind('_')
                        read_path = _dir_ + img_name[:index] + img_name[index + 1:]
                        if not name_formed:
                            name_formed = True
                            file_name = img_name[:index] + ".jpg"
                        data = load_content_img(read_path)
                        img_data.append(data)
                        sizes.append(data.shape)
                    else:
                        img_data.append(None)
                        sizes.append((0, 0, 0))
                except:
                        img_data.append(None)
                        sizes.append((0, 0, 0))
        #        total_height = 0
            is_row_valid = [False, False, False, False]
            widths = [0, 0, 0, 0]
            heights = [0, 0, 0, 0]
            is_col_valid = [False, False, False, False]
        #        total_width = 0
            i = 0
            # The loop below is for calculating the size of the original image
            col = 0
        #        while i < len(sizes):
        #            sub_array = sizes[i:i + 4]  # This is the array for one row
        #            for n, shape in enumerate(sub_array):
        #                if not is_row_valid[n]:
        #                    height, width, _ = shape
        #                    if width != 0:
        #                        is_row_valid[n] = True
        #                        widths[n] = width
        #                        total_width += width                                
        #                if not is_col_valid[col]:
        #                    height, width, _ = shape
        #                    if height != 0:
        #                        is_col_valid[col] = True
        #                        heights[col] = height
        #                        total_height += height
        #                    else:
        #                        for j in range (1,col):
        #                            if heights[col-j] != 0:
        #                                is_col_valid[col] = True
        #                                heights[col] = height[col-j]
        #                                total_height += height[col-j]
        #            i += 4
        #            col += 1
            empty_img = np.zeros(shape=[total_height, total_width, 3])
            row = 0
            col = 0
            col_index = 0
            row_index = 0
        #            try:
            for n, data in enumerate(img_data):
                Flag = 0
                if data is not None:
                    # plt.imshow(data, cmap='gray')
                    # plt.show()
                    height, width, depth = data.shape
                    height_end = row + height
                    width_end = col + width
                    empty_img[row:height_end, col:width_end] = data
                else:
                    for i in range (1,n+1):
                        previous = img_data[n-i]
                        if previous is not None:
                            height, width, depth = previous.shape
        #                        col += width
        #                        row += height
                            height_end = row + height
                            width_end = col + width
                            empty_img[row:height_end, col:width_end] = previous
                            img_data[n] = previous
                            Flag = 1
                            break
                    if Flag == 0:
                        for i in range (n+1,15):
                            thenext = img_data[i]
                            if thenext is not None:
                                height, width, depth = thenext.shape
                                height_end = row + height
                                width_end = col + width
                                empty_img[row:height_end, col:width_end] = thenext
                                Flag = 1
                                break    
                col += 1024
                col_index += 1
                tmp = n + 1
                if tmp % 4 == 0:
                    row += 1024
                    row_index += 1
                    col_index = 0
                    col = 0
            if len(empty_img)>0:
                cv2.imwrite(output + file_name, empty_img[:,:,0])
                print('write ' + file_name)
#            except:
#                print('abandon' + file_name)
        except:
            print('fail'+file_name)


#merge_back('output/step4/Ink and wash painting/', 'output/step44/Ink and wash painting/')
# merge_back('output/step4/Realism/', 'output/merged/Realism/')
# merge_back('output/step4/Romanticism/', 'output/merged/Romanticism/')
# merge_back('output/step4/Ukiyo_e/', 'output/merged/Ukiyo_e/')
merge_back('output/step4/Ink and wash painting/', 'output/merged/Ink and wash painting/')
# merge_back('output/step4/Photorealism/', 'output/step44/Photorealism/')
# merge_back('output/step4/Pointillism/', 'output/step44/Pointillism/')