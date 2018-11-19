# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 13:09:47 2018

@author: jpansh
"""

import os
import cv2
import numpy as np
from threading import Thread
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import scipy.io

batch_size = 1
img_height = 500
img_width = 500

def load_imgs(img_path, label):
    img_string = tf.read_file(img_path)
    img_decoded = tf.image.decode_png(img_string)
    img_resized = tf.image.resize_images(img_decoded, [img_height, img_width])
    img = tf.image.grayscale_to_rgb(img_resized)
    return img, label,img_string


def get_iterator():
    img_files = np.load('train_imgs1.npy')
    labels = np.load('train_lbs1.npy')
    dataset = tf.data.Dataset.from_tensor_slices((img_files, labels))
    dataset = dataset.map(map_func=load_imgs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator

itr = get_iterator()
# saveable = tf.contrib.data.make_saveable_from_iterator(itr)
# tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)

next_batch = itr.get_next()

count =  0

with tf.device('/gpu:0'), tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    while True:
        try:
            data, label, img_string = sess.run(next_batch)
            _,height, width, channels = data.shape
            
            if channels == 64:
                print('processing batch:' + str(count))
                print('channel:' + str(channels))
                print('image_name:' + str(img_string))
             
            count += 1
        except tf.errors.OutOfRangeError:
            print("End of dataset")  # ==> "End of dataset"
            break