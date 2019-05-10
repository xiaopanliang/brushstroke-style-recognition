import cv2
import os
import sys
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from random import randint


def crop_imgs(input_dir, output_dir):
    img_name_i = input_dir.rfind("/")
    img_name = input_dir[img_name_i + 1:]
    img = cv2.imread(input_dir)
    # Resize to 256 * 256 img
    resized_img = cv2.resize(img, (256, 256))
    # Crop 224 * 224 pieces
    crop_num = 16
    for n in range(crop_num):
        start_point_y = randint(0, 32)
        start_point_x = randint(0, 32)
        croped_img = resized_img[start_point_y:start_point_y + 224,
                                 start_point_x:start_point_x + 224]       
        file_name = output_dir + "/" + img_name[:-4] + "_" + str(n) + ".png"
        cv2.imwrite(file_name, croped_img)


def main(argv):
    input_path = argv[1]
    output_path = argv[2]
    if not os.path.exists(input_path):
        raise FileNotFoundError('The input path cannot be found.')
    if not os.path.exists(output_path):
        print("making output dir...")
        os.mkdir(output_path)
    styles = os.listdir(input_path)
    for style in styles:
        if not os.path.exists(output_path + "/"  + style) and style != ".DS_Store":
            data_sets = os.listdir(input_path + "/" + style)
            if not os.path.exists(output_path + "/"  + style):
                os.mkdir(output_path + "/"  + style)
            for data_set in data_sets:
                if data_set != ".DS_Store":
                    if not os.path.exists(output_path + "/"  + style + "/" + data_set):
                        os.mkdir(output_path + "/"  + style + "/" + data_set)
                    img_names = os.listdir(input_path + "/"  + style + "/" + data_set)
                    for img_name in img_names:
                        if img_name != ".DS_Store":
                            crop_imgs(input_path + "/"  + style + "/" + data_set + "/" + img_name, 
                                      output_path + "/"  + style + "/" + data_set)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Program accepts the input path and output path!")
        sys.exit(1)
    main(sys.argv)
    print('Executing...')
