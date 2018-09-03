"""
Created on Fri Nov 10 18:15:01 2017

@author: pan
"""
import os
import numpy as np
import cv2
from scipy.sparse.linalg import cg
from hashtable import hashtable
import time


def super_resolution_train(mat, Qangle, Qstrenth, Qcoherence):
    Q = np.zeros((Qangle * Qstrenth * Qcoherence, 4, 11 * 11, 11 * 11))
    V = np.zeros((Qangle * Qstrenth * Qcoherence, 4, 11 * 11, 1))
    h = np.zeros((Qangle * Qstrenth * Qcoherence, 4, 11 * 11))
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2YCrCb)[:, :, 2]
    mat = cv2.normalize(mat.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    HR = mat
    LR = cv2.GaussianBlur(HR, (0, 0), 2)  # should use low-resolution image but here use blur image because I am lazy
    for xP in range(5, LR.shape[0] - 6):
        for yP in range(5, LR.shape[1] - 6):
            patch = LR[xP - 5:xP + 6, yP - 5:yP + 6]
            [angle, strenth, coherence] = hashtable(patch, Qangle, Qstrenth, Qcoherence)
            j = angle * 9 + strenth * 3 + coherence
            A = patch.reshape(1, -1)
            b = HR[xP][yP]
            t = xP % 2 * 2 + yP % 2
            Q[j, t] += A * A.T
            V[j, t] += A.T * b
    for t in range(4):
        for j in range(Qangle * Qstrenth * Qcoherence):
            h[j, t] = cg(Q[j, t], V[j, t])[0]
    return h


def main():
    t = time.time()
    Qangle = 24
    Qstrenth = 3
    Qcoherence = 3

    base = 'input/wikipainting/'
    folders = os.listdir(base)
    ori_file_names = []

    for folder in folders:
        i = 1
        if folder != '.DS_Store':
            for file in os.listdir(base + folder + '/'):
                ori_file_names.append(base + folder + '/' + file)
                i = i + 1
                if i >= 11:
                    break
    fileList = []

    for painting_name in ori_file_names:
        fileList.append(painting_name)

    # for i in range(0,2) for the color channel, here we only try to process the channel 2 for simplification
    for file in fileList:
        try:
            print('processing ' + file)
            mat = cv2.imread(file)
            h = super_resolution_train(mat, Qangle, Qstrenth, Qcoherence)
        except:
            print('error happens on processing ' + file)
    print("Train is off")
    np.save("lowR2", h)
    _elapsed_ = time.time() - t
    return _elapsed_


if __name__ == '__main__':
    print('starting RAISR_train...')
    elapsed = main()
    print('finished RAISR train: ' + str(elapsed) + 's')
