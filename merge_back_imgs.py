import os
import numpy as np
import cv2
import collections
import matplotlib.pyplot as plt


style = "Ukiyo_e"


def merge_back(_dir_, output):
    groups = []

    f = open("img_names/img_names_" + style + ".txt", "r")
    for file in f:
        tmp_list = []
        for n in range(16):
            name = file[:-5] + "_" + str(n) + ".png"
            tmp_list.append(name)
        groups.append(tmp_list)

    if not os.path.isdir(output):
        os.mkdir(output)
    # Each group contain 16 pieces of the original image
    f = open("missed_imgs", "w")
    for num, group in enumerate(groups):
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
            if img_name != '0':
                index = img_name.rfind('_')
                read_path = _dir_ + img_name[:index] + img_name[index + 1:]
                if not name_formed:
                    name_formed = True
                    file_name = img_name[:index] + ".jpg"
                data = cv2.imread(read_path)
                if data is None:
                    f.write(img_name + "\n")
                else:
                    img_data.append(data)
                    sizes.append(data.shape)
            else:
                img_data.append(None)
                sizes.append((0, 0, 0))

        total_height = 0
        is_row_valid = [False, False, False, False]
        widths = [0, 0, 0, 0]
        heights = [0, 0, 0, 0]
        is_col_valid = [False, False, False, False]
        total_width = 0
        i = 0
        # The loop below is for calculating the size of the original image
        col = 0
        while i < len(sizes):
            sub_array = sizes[i:i + 4]  # This is the array for one row
            for n, shape in enumerate(sub_array):
                if not is_row_valid[n]:
                    height, width, _ = shape
                    if width != 0:
                        is_row_valid[n] = True
                        widths[n] = width
                        total_width += width
                if not is_col_valid[col]:
                    height, width, _ = shape
                    if height != 0:
                        is_col_valid[col] = True
                        heights[col] = height
                        total_height += height
            i += 4
            col += 1
        empty_img = np.zeros(shape=[total_height, total_width, 3])
        row = 0
        col = 0
        col_index = 0
        row_index = 0
        for n, data in enumerate(img_data):
            if data is not None:
                # plt.imshow(data, cmap='gray')
                # plt.show()
                height, width, depth = data.shape
                height_end = row + height
                width_end = col + width
                empty_img[row:height_end, col:width_end] = data
            col += widths[col_index]
            col_index += 1
            tmp = n + 1
            if tmp % 4 == 0:
                row += heights[row_index]
                row_index += 1
                col_index = 0
                col = 0
        cv2.imwrite(output + file_name, empty_img)
        print('write ' + file_name)
    f.close()


merge_back('output/' + style + '/', 'output/brushstrokes/' + style + '/')
