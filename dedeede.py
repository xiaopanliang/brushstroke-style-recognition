# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 23:56:39 2018

@author: jpansh
"""


def conv_layer(name, layer_input, w):
    return tf.nn.conv2d(layer_input, w, strides=[1, 1, 1, 1], padding='SAME', name=name)

def get_weights(name, vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    w = tf.Variable(weights, name=name)
    return w

net['conv5_1'] = conv_layer('conv5_1', net['pool4'], w=get_weights('conv5_1_w', vgg_layers, 28))