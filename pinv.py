# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:26:33 2018

@author: jpansh
"""

import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def pinv(A, b, reltol=1e-6):
    # Compute the SVD of the input matrix A
    s, u, v = tf.svd(A)

    # Invert s, clear entries lower than reltol*s[0].
    atol = tf.reduce_max(s) * reltol
    s = tf.boolean_mask(s, s > atol)
    s_inv = tf.diag(tf.concat([1. / s, tf.zeros([tf.size(b) - tf.size(s)])], 0))

    # Compute v * s_inv * u_t * b from the left to avoid forming large intermediate matrices.
    tmp = tf.matmul(u, b, transpose_a=True)
    result = tf.matmul(v, tf.matmul(s_inv, tmp))
    return result


