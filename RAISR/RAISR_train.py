"""
Created on Fri Nov 10 18:15:01 2017

@author: pan
"""
import os
import numpy as np
import cv2
from scipy.sparse.linalg import cg
from hashtable import hashtable
from matplotlib import pyplot as PLT

Qangle = 24
Qstrenth = 3
Qcoherence = 3
Q = np.zeros((Qangle*Qstrenth*Qcoherence,4,11*11,11*11))
V = np.zeros((Qangle*Qstrenth*Qcoherence,4,11*11,1))
h = np.zeros((Qangle*Qstrenth*Qcoherence,4,11*11))
dataDir="./thetrain"
fileList = []
for parent,dirnames,filenames in os.walk(dataDir):
    for filename in filenames:
        fileList.append(os.path.join(parent,filename))
#for i in range(0,2) for the color channel, here we only try to process the channel 2 for simplification
for file in fileList:
    print("HashMap of %s"%file)
    mat = cv2.imread(file)
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2YCrCb)[:,:,2]
    mat = cv2.normalize(mat.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    HR = mat
    LR = cv2.GaussianBlur(HR,(0,0),2) #should use low-resolution image but here use blur image because I am lazy
    for xP in range(5,LR.shape[0]-6):
        for yP in range(5,LR.shape[1]-6):
            patch = LR[xP-5:xP+6,yP-5:yP+6]
            [angle,strenth,coherence] = hashtable(patch,Qangle,Qstrenth,Qcoherence)
            j = angle*9+strenth*3+coherence
            A = patch.reshape(1,-1)
            b = HR[xP][yP]
            t = xP%2*2+yP%2
            Q[j,t] += A*A.T
            V[j,t] += A.T*b
for t in range(4):
    for j in range(Qangle*Qstrenth*Qcoherence):
        h[j,t] = cg(Q[j,t],V[j,t])[0]
print("Train is off")
np.save("./lowR2",h)