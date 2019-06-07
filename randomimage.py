import cv2
import os
import sys
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from random import randint


def random_pixels(img):

    height, width = img.shape
    for h in range(height):
        for w in range(width):
            if img[h][w] != 0:
                # Make sure the pixel range is between 70 and 255
                random_pixel = randint(70, 255)
                img[h][w] = random_pixel
            else:
                # Make sure the pixel range is between 1 and 30
                random_pixel = randint(0, 20)
                img[h][w] = random_pixel


def main(input_path,output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError('The input path cannot be found.')
    if not os.path.exists(output_path):
        print("making output dir...")
        os.mkdir(output_path)
    styles = os.listdir(input_path)
    for style in styles:
        if not os.path.exists(output_path + "/" + style):
            print("processing style:" + style)
            if style != ".DS_Store":
                data_sets = os.listdir(input_path + "/" + style)
                if not os.path.exists(output_path + "/"  + style):
                    os.mkdir(output_path + "/"  + style)
                for data_set in data_sets:
                    if data_set != ".DS_Store":
                        # if not os.path.exists(output_path + "/"  + style + "/" + data_set):
                        #     os.mkdir(output_path + "/"  + style + "/" + data_set)
                        # img_names = os.listdir(input_path + "/"  + style + "/" + data_set)
                        # for img_name in img_names:
                        #     if img_name != ".DS_Store":
                        img = cv2.imread(input_path + "/" + style + "/" + data_set, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (256, 256))
                            (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                            random_pixels(im_bw)
                            cv2.imwrite(output_path + "/"  + style + "/" + data_set[:-4] + ".jpg", im_bw)


if __name__ == '__main__':
    print('Executing...')
    main('DataBase', 'Random')
    print("Finished...")
