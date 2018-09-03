# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:20:25 2017

@author: pan
"""
import numpy as np
import cv2
from PIL import Image
import os
from hashtable import hashtable
from multiprocessing import Process
import timeit


Qangle = 24
Qstrenth = 3
Qcoherence = 3


expect_styles = {'Baroque', 'Expressionism', 'Impressionism', 'Pointillism', 'Realism', 'Romanticism', 'Ukiyo-e'}


def RAISR_test(ori_files, results, _thread_num_):
    for painting_name, result in zip(ori_files, results):
        try:
            if not os.path.isfile(result):
                print('processing ' + painting_name)
                start = timeit.default_timer()
                mat = cv2.imread(painting_name)
                width, height, _ = mat.shape
                if width <= 500 and height <= 500:
                    fx = 5.5
                    fy = 5.5
                elif width <= 1000 and height <= 1000:
                    fx = 3.3
                    fy = 3.3
                elif width <= 2000 and height <= 2000:
                    fx = 1.3
                    fy = 1.3
                else:
                    fx = 1
                    fy = 1
                mat = cv2.imread(painting_name)
                h = np.load("lowR2.npy")
                mat = cv2.cvtColor(mat, cv2.COLOR_BGR2YCrCb)[:, :, 2]
                LR = cv2.resize(mat, (0, 0), fx=fx, fy=fy)
                LRDirect = np.zeros((LR.shape[0], LR.shape[1]))
                for xP in range(5, LR.shape[0] - 6):
                    for yP in range(5, LR.shape[1] - 6):
                        patch = LR[xP - 5:xP + 6, yP - 5:yP + 6]
                        [angle, strenth, coherence] = hashtable(patch, Qangle, Qstrenth, Qcoherence)
                        j = angle * 9 + strenth * 3 + coherence
                        A = patch.reshape(1, -1)
                        t = xP % 2 * 2 + yP % 2
                        hh = np.matrix(h[j, t])
                        LRDirect[xP][yP] = hh * A.T
                mat = cv2.imread(painting_name)
                mat = cv2.cvtColor(mat, cv2.COLOR_BGR2YCrCb)
                LR = cv2.resize(mat, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
                LRDirectImage = LR
                LRDirectImage[:, :, 2] = LRDirect
                A = cv2.cvtColor(LRDirectImage, cv2.COLOR_YCrCb2RGB)
                im = Image.fromarray(A)
                im.save(result)
                stop = timeit.default_timer()
                print(painting_name + ' running time:' + str(stop - start) + 's')
        except:
            print('error on processing ' + painting_name)
    print('finished RAISR testing thread ' + str(_thread_num_))


def main():
    cores = 8

    base = 'wikiart/'
    output_folder = 'output/step2/'
    folders = os.listdir(base)

    ori_file_names = []
    result_file_names = []
    for folder in folders:
        if folder != '.DS_Store' and folder in expect_styles:
            count = 0
            if not os.path.isdir(output_folder + folder):
                os.makedirs(output_folder + folder)
            for file in os.listdir(base + folder + '/'):
                if count >= 500:
                    break
                ori_file_names.append(base + folder + '/' + file)
                result_file_names.append(output_folder + folder + '/' + file)
                count += 1

    print(str(len(ori_file_names)) + ' images are going to be processed')
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
        thread = Process(target=RAISR_test, args=[ori_group, result_group, i])
        thread.start()
        threads.append(thread)

    for n, thread in enumerate(threads):
        thread.join()


if __name__ == '__main__':
    print('starting the RAISR_test...')
    main()
    print('finished the RAISR testing...')