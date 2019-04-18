import cv2
import os
import sys
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def random_pixels(img):
    height, width, _ = img.shape
    for h in range(height):
        for w in range(width):
            if img[h][w][0] > 128:
                img[h][w] = [190, 190, 190]
            else:
                img[h][w] = [50, 50, 50]


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
        if style != ".DS_Store":
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
                            img = cv2.imread(input_path + "/"  + style + "/" + data_set + "/" + img_name)
                            random_pixels(img)
                            cv2.imwrite(output_path + "/"  + style + "/" + data_set + "/" + img_name, img)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Program accepts the input path and output path!")
        sys.exit(1)
    print('Executing...')
    start_time = time.time()
    main(sys.argv)
    end_time = time.time()
    print('finished:' + str(end_time - start_time) + "s")
