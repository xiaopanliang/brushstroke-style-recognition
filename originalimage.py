import cv2
import os
import sys
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from random import randint
from shutil import copyfile


# The method only filter out the image used for brushstrokes
def main(argv):
    input_path = argv[1]
    output_path = argv[2]
    brushstroke_path = argv[3]
    if not os.path.exists(input_path):
        raise FileNotFoundError('The input path cannot be found.')
    if not os.path.exists(brushstroke_path):
        raise FileNotFoundError('The brushstroke path cannot be found.')
    if not os.path.exists(output_path):
        print("making output dir...")
        os.mkdir(output_path)
    styles = os.listdir(brushstroke_path)
    for style in styles:
        if style != ".DS_Store":
            data_sets = os.listdir(brushstroke_path + "/" + style)
            if not os.path.exists(output_path + "/"  + style):
                os.mkdir(output_path + "/"  + style)
            for data_set in data_sets:
                if data_set != ".DS_Store":
                    if not os.path.exists(output_path + "/"  + style + "/" + data_set):
                        os.mkdir(output_path + "/"  + style + "/" + data_set)
                    img_names = os.listdir(brushstroke_path + "/"  + style + "/" + data_set)
                    for img_name in img_names:
                        if img_name != ".DS_Store":
                            div_index = img_name.rfind("_")
                            img_name = img_name[:div_index] + ".jpg"
                            copyfile(input_path + "/"  + style + "/" +  img_name, 
                                     output_path + "/"  + style + "/" + data_set + "/" + img_name)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Program accepts the input path, output path, and brushstrok path!")
        sys.exit(1)
    main(sys.argv)
    print('Executing...')
