# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 12:03:23 2018

@author: jpansh
"""

import tensorflow as tf
import numpy as np
import scipy.io

check_pt_path_str = 'checkpoint/'
batch_size = 30
img_height = 200
img_width = 200

def load_imgs(img_path, label):
    img_string = tf.read_file(img_path)
    img_decoded = tf.image.decode_png(img_string)
    img_resized = tf.image.resize_images(img_decoded, [img_height, img_width])
    img = tf.image.grayscale_to_rgb(img_resized)
    return img, label


def get_train_iterator():
    img_files = np.load('gan_imgs.npy')
    labels = np.load('gan_lbs.npy')
    dataset = tf.data.Dataset.from_tensor_slices((img_files, labels))
#    dataset = dataset.shuffle(12000)
    dataset = dataset.map(map_func=load_imgs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator

itr_train = get_train_iterator()
next_train_batch = itr_train.get_next()

with tf.device('/cpu:0'),tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    train_data, train_label = sess.run(next_train_batch)    
