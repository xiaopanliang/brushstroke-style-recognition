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


Qangle = 24
Qstrenth = 3
Qcoherence = 3

base = './Alpha/wikipainting/' 

output_folder = './Alpha/output/'

folders = os.listdir(base)

ori_file_names = []

result_file_names = []

for folder in folders:

    if not os.path.isdir(output_folder + folder):

        os.makedirs(output_folder + folder)

    for file in os.listdir(base + folder + '/'):

        ori_file_names.append(base + folder + '/' + file)

        result_file_names.append(output_folder + folder + '/' + file)

for painting_name, result_name in zip(ori_file_names, result_file_names):
    mat = cv2.imread(painting_name)
    [width,height,channel] = mat.shape
    if width<=1000 and height<=1000:
        fx = 2.3
        fy = 2.3
    elif width<=2000 and height<=2000:
        fx = 1.3
        fy = 1.3
    else:
        fx = 1
        fy = 1
    mat = cv2.imread(painting_name)
    [width,height,channel] = mat.shape
    h = np.load("lowR2.npy")
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2YCrCb)[:,:,2]
    LR = cv2.resize(mat,(0,0),fx=fx,fy=fy)
    LRDirect = np.zeros((LR.shape[0],LR.shape[1]))
    for xP in range(5,LR.shape[0]-6):
        for yP in range(5,LR.shape[1]-6):
            patch = LR[xP-5:xP+6,yP-5:yP+6]
            [angle,strenth,coherence] = hashtable(patch,Qangle,Qstrenth,Qcoherence)
            j = angle*9+strenth*3+coherence 
            A = patch.reshape(1,-1)
            t = xP%2*2+yP%2
            hh = np.matrix(h[j,t])
            LRDirect[xP][yP] = hh*A.T
    print("Test is off")
    
    mat = cv2.imread(painting_name)
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2YCrCb)
    LR = cv2.resize(mat,(0,0),fx=fx,fy=fy,interpolation=cv2.INTER_LINEAR)
    LRDirectImage = LR
    LRDirectImage[:,:,2] = LRDirect
    A = cv2.cvtColor(LRDirectImage, cv2.COLOR_YCrCb2RGB);
    im = Image.fromarray(A)
    im.save(painting_name)