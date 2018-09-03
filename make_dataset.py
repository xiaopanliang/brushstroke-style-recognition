import os
import cv2
import numpy as np
from threading import Thread
import matplotlib.image as mpimg


input_height = 500  # 768
input_width = 500  # 1024


# Limit the size of the message to be within the max_size
def resize_content_img(img):
    return cv2.resize(img, dsize=(input_height, input_width), interpolation=cv2.INTER_AREA)


def read_image(path):
    _img_ = mpimg.imread(path)
    return _img_.astype(np.float32)


def load_content_img(path):
    _img_ = read_image(path)
    _img_ = resize_content_img(_img_)
    # show_img = img.astype(np.uint8)
    # plt.imshow(show_img)
    # plt.show()
    return _img_


def load_image_data_thread(files, return_vals):
    _img_data_ = []
    for _file_ in files:
        try:
            _img_ = load_content_img(_file_)
            if _img_.shape == (500, 500):
                _img_ = np.stack((_img_,) * 3, -1)  # Make the image to be 3 channel image
            _img_data_.append(_img_)
        except TypeError:
            print('Type error on loading ' + _file_)
    return_vals.append(_img_data_)


def main():
    base = 'output/step4/'
    folders = os.listdir(base)
    label = 0

    train_files = []
    train_labels = []

    eval_files = []
    eval_labels = []

    for folder in folders:
        if folder != '.DS_Store':
            files = os.listdir(base + folder + '/')
            files = files[:10]
            files_num = len(files)
            train_set_num = int(files_num / 10 * 9)
            train_set = files[:train_set_num]
            eval_set = files[train_set_num:]
            for file in train_set:
                train_files.append(base + folder + '/' + file)
                train_labels.append(label)
            for file in eval_set:
                eval_files.append(base + folder + '/' + file)
                eval_labels.append(label)
            label += 1
    np.save('train_imgs', train_files)
    np.save('train_lbs', train_labels)


if __name__ == '__main__':
    print('starting making dataset...')
    main()
    print('finished making dataset')
