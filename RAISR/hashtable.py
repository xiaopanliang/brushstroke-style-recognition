"""
Created on Thu Nov  9 00:47:51 2017

@author: pan
"""

import numpy as np

def hashtable(patch,Qangle,Qstrenth,Qcoherence):
    [gx,gy] = np.gradient(patch)
    G = np.matrix((gx.ravel(),gy.ravel())).T
    x = G.T*G
    [eigenvalues,eigenvectors] = np.linalg.eig(x)
    angle = np.math.atan2(eigenvectors[0,1],eigenvectors[0,0])
    if angle<0:
        angle += np.pi
    strength = eigenvalues.max()/(eigenvalues.sum()+0.0001)
    lamda1 = np.math.sqrt(eigenvalues.max())
    lamda2 = np.math.sqrt(eigenvalues.min())
    coherence = np.abs((lamda1-lamda2)/(lamda1+lamda2+0.0001))
    angle = np.floor(angle/(np.pi/Qangle)-1)
    strength = np.floor(strength/(1/Qstrenth)-1)
    coherence = np.floor(coherence/(1/Qcoherence)-1)    
    return int(angle),int(strength),int(coherence)