import os
import cv2
from multiprocessing import Process
import timeit


expect_styles = {'Baroque', 'Expressionism', 'Impressionism', 'Pointillism', 'Realism', 'Romanticism', 'Ukiyo_e',
                 'Photorealism', 'Ink and wash painting', 'Academicism'}


def split_image(img):
    factor = 4
    height, width, _ = img.shape
    sub_height = int(height / factor)
    sub_width = int(width / factor)
    parts = []
    for i in range(factor):
        y = i * sub_height
        if i >= factor - 1:
            y_end = height
        else:
            y_end = y + sub_height
        for j in range(factor):
            x = j * sub_width
            if j >= factor - 1:
                x_end = width
            else:
                x_end = x + sub_width
            piece = img[y:y_end, x:x_end]
            parts.append(piece)
    return parts


def split_images(ori_file_names, result_file_names, n):
    print('starting the splitting thread ' + str(n))
    _start_ = timeit.default_timer()
    for ori_file_name, result_file_name in zip(ori_file_names, result_file_names):
        if not os.path.isfile(result_file_name):
            print('processing ' + ori_file_name)
            img = cv2.imread(ori_file_name)
            parts = split_image(img)
            for n, part in enumerate(parts):
                cv2.imwrite(result_file_name + str(n) + '.png', part)
        else:
            print('skip ' + ori_file_name)
    _stop_ = timeit.default_timer()
    print('finished thread ' + str(n) + ':' + str(_stop_ - _start_) + 's')


def main():
    cores = 1
    base = 'output/step2/'
    output_folder = 'output/step4/'
    folders = os.listdir(base)
    ori_file_names = []
    result_file_names = []
    for folder in folders:
        if folder != '.DS_Store':
            if not os.path.isdir(output_folder + folder):
                os.makedirs(output_folder + folder)
            for file in os.listdir(base + folder + '/'):
                if file != '.DS_Store':
                    ori_file_names.append(base + folder + '/' + file)
                    result_file_names.append(output_folder + folder + '/' + file[:-4])

    num_files_per_core = round(len(ori_file_names) / cores)
    ori_groups = []
    result_groups = []
    # Split the data for different threads
    for i in range(0, cores):
        start_index = i * num_files_per_core
        if i < cores - 1:
            ori_group = ori_file_names[start_index:start_index + num_files_per_core]
            result_group = result_file_names[start_index:start_index + num_files_per_core]
        else:
            ori_group = ori_file_names[start_index:]
            result_group = result_file_names[start_index:]
        ori_groups.append(ori_group)
        result_groups.append(result_group)

    threads = []
    for i in range(0, cores):
        ori_group = ori_groups[i]
        result_group = result_groups[i]
        thread = Process(target=split_images,
                         args=(ori_group, result_group, i))
        thread.start()
        threads.append(thread)
    for n, thread in enumerate(threads):
        thread.join()


if __name__ == '__main__':
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()
    print('total time:' + str(stop - start) + 's')
