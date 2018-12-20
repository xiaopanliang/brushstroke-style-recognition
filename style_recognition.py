# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:10:07 2018

@author: jpansh
"""

import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import math
import os
import cv2

check_pt_path_str = 'checkpointrandombrush/'
batch_size = 16
img_height = 32*4
img_width = 32*4
epochs = 1000
units = 10



def conv_layer(name, layer_input, w):
    return tf.nn.conv2d(layer_input, w, strides=[1, 1, 1, 1], padding='SAME', name=name)


def relu_layer(name, layer_input, b):
    return tf.nn.relu(layer_input + b, name=name)


def pool_layer(name, layer_input):
    return tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def fc_layer(name, layer_input, units, activation):
    return tf.layers.dense(inputs=layer_input, units=units, name=name, activation=activation)  # TODO: Add the units here


def get_weights(name, vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    w = tf.Variable(weights, name=name)
    return w


def get_bias(name, vgg_layers, i):
    bias = vgg_layers[i][0][0][2][0][1]
    b = tf.Variable(np.reshape(bias, bias.size), name=name)
    return b


def calc_texture_logits(net):
    # Load the style image to the model
    layers = ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']
    style_layer_weights = [0.05, 0.05, 0.2, 0.3, 0.4]
    connected_G = None
    n = 0
#    for layer in layers:
    for layer, weight in zip(layers, style_layer_weights):
        layer_data = net[layer]
        index, height, width, channels = layer_data.get_shape()
        M = height.value * width.value
        N = channels.value
        F = tf.reshape(layer_data, [-1, M, N])
#        mean,variance = tf.nn.moments(F,1)
#        mean = tf.reshape(mean,[-1,N,1])
#        variance = tf.reshape(variance,[-1,N,1])
#        MeanMatrix = tf.matmul(tf.transpose(mean,perm=[0,2,1]), mean)
#        VarMatrix = tf.matmul(tf.transpose(variance,perm=[0,2,1]), variance)
#        G = (MeanMatrix + VarMatrix)*weight
        G = tf.matmul(tf.transpose(F,perm=[0, 2, 1]), F)*weight

        # Keep recording gram matrices from different layers
        n += 1
        net['gram_' + str(n)] = G

        _,y, x = G.get_shape()
        G = tf.reshape(G, [-1, y.value * x.value])
        if connected_G is None:
            connected_G = G
        else:
            connected_G = tf.concat([connected_G, G], axis=1)
    net['g_fc1'] = fc_layer('fc_g1', connected_G, 256*2, tf.nn.relu)
    net['g_fc2'] = fc_layer('fc_g2', tf.layers.dropout(net['g_fc1'], 0),512, tf.nn.relu)
    net['g_fc3'] = fc_layer('fc_g3', tf.layers.dropout(net['g_fc2'], 0), units, None)
    return net['g_fc3']


def calc_obj_logits(net):
    # Load the content image to the model
    object_output = net['pool5']
    _, height, width, depth = object_output.get_shape()
    object_output = tf.reshape(object_output, [-1, height.value * width.value * depth.value])
    net['o_fc1'] = fc_layer('fc_o1', object_output, 1024, tf.nn.relu)
    net['o_fc2'] = fc_layer('fc_o2', tf.layers.dropout(net['o_fc1'], 0), 1024, tf.nn.relu)
    net['o_fc3'] = fc_layer('fc_o3', tf.layers.dropout(net['o_fc2'], 0), units, None)
    return net['o_fc3']


net = {}


def cnn_model_fn():
    print('\nBUILDING VGG-19 NETWORK')

    print('loading model weights...')
    vgg_raw_net = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
    vgg_layers = vgg_raw_net['layers'][0]
    print('constructing layers...')
    net['input'] = tf.placeholder(tf.float32, shape=[batch_size, img_height, img_width, 3])
    net['labels'] = tf.placeholder(tf.int32, shape=[batch_size])

    print('LAYER GROUP 1')
    net['conv1_1'] = conv_layer('conv1_1', net['input'], w=get_weights('conv1_1_w', vgg_layers, 0))
#    net['bn1_1'] = tf.layers.batch_normalization(net['conv1_1'], training=False, momentum=0.9)
    net['relu1_1'] = relu_layer('relu1_1', net['conv1_1'], b=get_bias('relu1_1_b', vgg_layers, 0))

    net['conv1_2'] = conv_layer('conv1_2', net['relu1_1'], w=get_weights('conv1_2_w', vgg_layers, 2))
#    net['bn1_2'] = tf.layers.batch_normalization(net['conv1_2'], training=False, momentum=0.9)
    net['relu1_2'] = relu_layer('relu1_2', net['conv1_2'], b=get_bias('relu1_2_b', vgg_layers, 2))

    net['pool1'] = pool_layer('pool1', net['relu1_2'])


    print('LAYER GROUP 2')
    net['conv2_1'] = conv_layer('conv2_1', net['pool1'], w=get_weights('conv2_1_w', vgg_layers, 5))
#    net['bn2_1'] = tf.layers.batch_normalization(net['conv2_1'], training=False, momentum=0.9)
    net['relu2_1'] = relu_layer('relu2_1', net['conv2_1'], b=get_bias('relu2_1_b', vgg_layers, 5))

    net['conv2_2'] = conv_layer('conv2_2', net['relu2_1'], w=get_weights('conv2_2_w', vgg_layers, 7))
#    net['bn2_2'] = tf.layers.batch_normalization(net['conv2_2'], training=False, momentum=0.9)
    net['relu2_2'] = relu_layer('relu2_2', net['conv2_2'], b=get_bias('relu2_2_b', vgg_layers, 7))

    net['pool2'] = pool_layer('pool2', net['relu2_2'])


    
    print('LAYER GROUP 3')
    net['conv3_1'] = conv_layer('conv3_1', net['pool2'], w=get_weights('conv3_1_w', vgg_layers, 10))
#    net['bn3_1'] = tf.layers.batch_normalization(net['conv3_1'], training=False, momentum=0.9)
    net['relu3_1'] = relu_layer('relu3_1', net['conv3_1'], b=get_bias('relu3_1_b', vgg_layers, 10))

    net['conv3_2'] = conv_layer('conv3_2', net['relu3_1'], w=get_weights('conv3_2_w', vgg_layers, 12))
#    net['bn3_2'] = tf.layers.batch_normalization(net['conv3_2'], training=False, momentum=0.9)
    net['relu3_2'] = relu_layer('relu3_2', net['conv3_2'], b=get_bias('relu3_2_b', vgg_layers, 12))

    net['conv3_3'] = conv_layer('conv3_3', net['relu3_2'], w=get_weights('conv3_3_w', vgg_layers, 14))
#    net['bn3_3'] = tf.layers.batch_normalization(net['conv3_3'], training=False, momentum=0.9)
    net['relu3_3'] = relu_layer('relu3_3', net['conv3_3'], b=get_bias('relu3_3_b', vgg_layers, 14))

    net['conv3_4'] = conv_layer('conv3_4', net['relu3_3'], w=get_weights('conv3_4_w', vgg_layers, 16))
#    net['bn3_4'] = tf.layers.batch_normalization(net['conv3_4'], training=False, momentum=0.9)
    net['relu3_4'] = relu_layer('relu3_4', net['conv3_4'], b=get_bias('relu3_4_b', vgg_layers, 16))

    net['pool3'] = pool_layer('pool3', net['relu3_4'])

    
    print('LAYER GROUP 4')
    net['conv4_1'] = conv_layer('conv4_1', net['pool3'], w=get_weights('conv4_1_w', vgg_layers, 19))
#    net['bn4_1'] = tf.layers.batch_normalization(net['conv4_1'], training=False, momentum=0.9)
    net['relu4_1'] = relu_layer('relu4_1', net['conv4_1'], b=get_bias('relu4_1_b', vgg_layers, 19))

    net['conv4_2'] = conv_layer('conv4_2', net['relu4_1'], w=get_weights('conv4_2_w', vgg_layers, 21))
#    net['bn4_2'] = tf.layers.batch_normalization(net['conv4_2'], training=False, momentum=0.9)
    net['relu4_2'] = relu_layer('relu4_2', net['conv4_2'], b=get_bias('relu4_2_b', vgg_layers, 21))

    net['conv4_3'] = conv_layer('conv4_3', net['relu4_2'], w=get_weights('conv4_3_w', vgg_layers, 23))
#    net['bn4_3'] = tf.layers.batch_normalization(net['conv4_3'], training=False, momentum=0.9)
    net['relu4_3'] = relu_layer('relu4_3', net['conv4_3'], b=get_bias('relu4_3_b', vgg_layers, 23))

    net['conv4_4'] = conv_layer('conv4_4', net['relu4_3'], w=get_weights('conv4_4_w', vgg_layers, 25))
#    net['bn4_4'] = tf.layers.batch_normalization(net['conv4_4'], training=False, momentum=0.9)
    net['relu4_4'] = relu_layer('relu4_4', net['conv4_4'], b=get_bias('relu4_4_b', vgg_layers, 25))

    net['pool4'] = pool_layer('pool4', net['relu4_4'])


    
    print('LAYER GROUP 5')
    net['conv5_1'] = conv_layer('conv5_1', net['pool4'], w=get_weights('conv5_1_w', vgg_layers, 28))
#    net['bn5_1'] = tf.layers.batch_normalization(net['conv5_1'], training=False, momentum=0.9)
    net['relu5_1'] = relu_layer('relu5_1', net['conv5_1'], b=get_bias('relu5_1_b', vgg_layers, 28))

    net['conv5_2'] = conv_layer('conv5_2', net['relu5_1'], w=get_weights('conv5_2', vgg_layers, 30))
#    net['bn5_2'] = tf.layers.batch_normalization(net['conv5_2'], training=False, momentum=0.9)
    net['relu5_2'] = relu_layer('relu5_2', net['conv5_2'], b=get_bias('relu5_2', vgg_layers, 30))

    net['conv5_3'] = conv_layer('conv5_3', net['relu5_2'], w=get_weights('conv5_3', vgg_layers, 32))
#    net['bn5_3'] = tf.layers.batch_normalization(net['conv5_3'], training=False, momentum=0.9)
    net['relu5_3'] = relu_layer('relu5_3', net['conv5_3'], b=get_bias('relu5_3', vgg_layers, 32))

    net['conv5_4'] = conv_layer('conv5_4', net['relu5_3'], w=get_weights('conv5_4', vgg_layers, 34))
#    net['bn5_4'] = tf.layers.batch_normalization(net['conv5_4'], training=False, momentum=0.9)
    net['relu5_4'] = relu_layer('relu5_4', net['conv5_4'], b=get_bias('relu5_4', vgg_layers, 34))

    net['pool5'] = pool_layer('pool5', net['relu5_4'])


    
    # Fully connected to get logits
    obj_logits = calc_obj_logits(net)
    texture_logits = calc_texture_logits(net)
#    Denoise_logits = calc_regularization_logits(net)
    logits =  texture_logits
    # sum_logits = logits
    sum_logit1 = tf.reduce_sum(logits[0:4,:units],0,keepdims=True)
    sum_logit2 = tf.reduce_sum(logits[4:8, :units], 0, keepdims=True)
    sum_logit3 = tf.reduce_sum(logits[8:12, :units], 0, keepdims=True)
    sum_logit4 = tf.reduce_sum(logits[12:16, :units], 0, keepdims=True)
    sum_logits = tf.concat([sum_logit1,sum_logit1,sum_logit1,sum_logit1,sum_logit2,sum_logit2,sum_logit2,sum_logit2,sum_logit3,sum_logit3,sum_logit3,sum_logit3,sum_logit4,sum_logit4,sum_logit4,sum_logit4],0)

    
    prediction = tf.argmax(input=logits, axis=1, name='prediction')
    obj_prediction = tf.argmax(input=obj_logits, axis=1, name='obj_prediction')
    tex_prediction = tf.argmax(input=texture_logits, axis=1, name='tex_prediction')
    testing_prediction = tf.argmax(input=sum_logits, axis=1, name='testing_prediction')

    acc, acc_op = tf.metrics.accuracy(labels=net['labels'], predictions=prediction)
    testing_acc, testing_acc_op = tf.metrics.accuracy(labels=net['labels'], predictions=testing_prediction)
    # conf, conf_op = tf.confusion_matrix(labels=net['labels'], predictions=testing_prediction)
    l2_regularizer = tf.contrib.layers.l2_regularizer(
       scale=0.00005, scope=None
    )
    
    weights = tf.trainable_variables() # all vars of your graph
    
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=net['labels'], logits=logits)+ regularization_penalty
#    loss = tf.losses.sparse_softmax_cross_entropy(labels=net['labels'], logits=obj_logits) + regularization_penalty + tf.losses.sparse_softmax_cross_entropy(labels=net['labels'], logits=texture_logits)
#    tex_loss = regularization_penalty + tf.losses.sparse_softmax_cross_entropy(labels=net['labels'], logits=texture_logits)
    return loss, acc, acc_op, prediction,logits, texture_logits ,obj_prediction,tex_prediction, testing_prediction,testing_acc,testing_acc_op

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def load_imgs(img_path, label):
    img_string = tf.read_file(img_path)
    img_decoded = tf.image.decode_jpeg(img_string,0)
    img = tf.image.resize_image_with_pad(img_decoded, img_height, img_width,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # img = tf.image.grayscale_to_rgb(img)
    # img = tf.image.adjust_contrast(img,10)
    img = tf.image.per_image_standardization(img)
    return img, label


def get_train_iterator():
    img_files = np.load('train_imgs.npy')
    labels = np.load('train_lbs.npy')
    dataset = tf.data.Dataset.from_tensor_slices((img_files, labels))
    dataset = dataset.shuffle(20000)
    dataset = dataset.repeat(epochs)
    dataset = dataset.map(map_func=load_imgs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator

def get_vali_iterator():
    img_files = np.load('rbvali_imgs.npy')
    labels = np.load('rbvali_lbs.npy')
    dataset = tf.data.Dataset.from_tensor_slices((img_files, labels))
    dataset = dataset.repeat(epochs)
#    dataset = dataset.shuffle(12000)
    dataset = dataset.map(map_func=load_imgs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator

def get_eval_iterator():
    img_files = np.load('rbeval_imgs.npy')
    labels = np.load('rbeval_lbs.npy')
    dataset = tf.data.Dataset.from_tensor_slices((img_files, labels))
    dataset = dataset.repeat(epochs)
#    dataset = dataset.shuffle(12000)
    dataset = dataset.map(map_func=load_imgs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator


# The function gets the layer output of the neural network
# Param:
# layer: this is the string which tells which layer output should be obtained
def get_layer_output(sess, net, layer, data_batch, label_batch):
    imgs_out = net[layer]
    imgs_out = sess.run(imgs_out, feed_dict={net['input']: data_batch,
                                             net['labels']: label_batch})
    base = 'layer_output/'
    if layer[:4] != 'gram':
        # Enumerate different images
        for n, data in enumerate(zip(label_batch, imgs_out)):
            style = data[0]
            directory = base + layer + '/' + str(style) + '/' + str(n) + '/'
            if not os.path.isdir(directory):
                os.makedirs(directory)
            img_data = data[1]
            height, width, depth = img_data.shape
            for channel in range(depth):
                channel_output = (img_data[:, :, channel]).astype(np.uint8)
                fig = plt.imshow(channel_output, cmap='gray')
                plt.axis('off')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                plt.savefig(directory + str(channel) + '.png', bbox_inches='tight', pad_inches=0)
    else:
        # Enumerate different images
        for n, data in enumerate(zip(label_batch, imgs_out)):
            style = data[0]
            directory = base + layer + '/' + str(style) + '/'
            if not os.path.isdir(directory):
                os.makedirs(directory)
            img_out = data[1]
            # Output data for different color channels
            channel_output = img_out.astype(np.uint8)
            fig = plt.imshow(channel_output, cmap='gray')
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.savefig(directory + str(n) + '.png', bbox_inches='tight', pad_inches=0)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    loss, acc, acc_op, prediction,logits,obj_logits,obj_prediction,tex_prediction,testing_prediction,testing_acc,testing_acc_op = cnn_model_fn()

    variable_summaries(loss, "loss")
    variable_summaries(acc, "acc")
    merged_summary = tf.summary.merge_all()

    optimizer = tf.train.AdamOptimizer(learning_rate=0.00001,epsilon=1e-08,use_locking=False)
 
    train_op = optimizer.minimize(loss)
#    train_op2 = optimizer.minimize(tex_loss)

    # Get the iterator for the data set
    itr_train = get_train_iterator()
    next_train_batch = itr_train.get_next()
    
    # itr_vali = get_vali_iterator()
    # next_vali_batch = itr_vali.get_next()
    #
    # itr_eval = get_eval_iterator()
    # next_eval_batch = itr_eval.get_next()

    # The counter for tracking the number of batches
    count = 0
    arr = []
    saver = tf.train.Saver(tf.trainable_variables())
    with tf.device('/gpu:0'), tf.Session() as sess:
        # train_writer = tf.summary.FileWriter("train_graphs/", sess.graph)
        # eval_writer = tf.summary.FileWriter("eval_graphs/", sess.graph)
        # vali_writer = tf.summary.FileWriter("vali_graphs/", sess.graph)
        # sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())

        print('restoring model...')
#        test_weight = tf.get_default_graph().get_tensor_by_name('conv1_1_w:0')
#        prev = sess.run(test_weight)
        saver.restore(sess, tf.train.latest_checkpoint(check_pt_path_str))
#        after = sess.run(test_weight)
        # Calc difference before read and after read
#        diff = after - prev
#        diff = diff.sum()
#        print('restored:' + str(diff))

        while True: 
            train_data, train_label = sess.run(next_train_batch)
            # eval_data, eval_label = sess.run(next_eval_batch)
            # vali_data, vali_label = sess.run(next_vali_batch)

            print('********************************')
            print('processing batch:' + str(count))
            _, _, _, channel =  train_data.shape
            if channel == 3:
#                 for _ in range(100):
#                     train_dict = {net['input']:  train_data, net['labels']:  train_label}
# ##                        # Train the model
#                     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#                     with tf.control_dependencies(update_ops):
#                         loss_val = sess.run(loss, feed_dict=train_dict)
#                         print('loss:' + str(loss_val))
#                         if loss_val < 0.1:
#                             break
#                         if math.isnan(loss_val):
#                             return
#                         sess.run(train_op, feed_dict=train_dict)
# # #
                train_dict = {net['input']: train_data, net['labels']: train_label}
                train_pre_val = str(sess.run(prediction, feed_dict=train_dict))
                print('train_predic:' + train_pre_val)
#                    Print out the actual labels
                train_actual_val = str( train_label)
                print('train_actual' + train_actual_val)

                # get_layer_output(sess, net, 'relu1_2', train_data, train_label)
                get_layer_output(sess, net, 'relu2_2', train_data, train_label)
                # get_layer_output(sess, net, 'relu3_4', train_data, train_label)
                # get_layer_output(sess, net, 'relu4_4', train_data, train_label)
                # get_layer_output(sess, net, 'relu5_4', train_data, train_label)
                # get_layer_output(sess, net, 'gram_1', train_data, train_label)
                # get_layer_output(sess, net, 'gram_2', train_data, train_label)
                # get_layer_output(sess, net, 'gram_3', train_data, train_label)
                # get_layer_output(sess, net, 'gram_4', train_data, train_label)
                # get_layer_output(sess, net, 'gram_5', train_data, train_label)
                break
#
#                 ##                     Print out the accuracy value
#                 eval_dict = {net['input']: eval_data, net['labels']: eval_label}
#                 sess.run(testing_acc_op, feed_dict=eval_dict)
#                 eval_acc_val = sess.run(testing_acc, feed_dict=eval_dict)
#                 print('eval_accuracy:' + str(eval_acc_val))
#
#                 eval_pre_val = str(sess.run(testing_prediction, feed_dict=eval_dict))
#                 print('eval_predic:' + eval_pre_val)
#
#
#                 # Print out the actual labels
#                 eval_actual_val = str(eval_label)
#                 print('eval_actual:' + eval_actual_val)

# ##                     Print out the accuracy value
#                 vali_dict = {net['input']: vali_data, net['labels']: vali_label}
#                 sess.run(testing_acc_op, feed_dict=vali_dict)
#                 vali_acc_val = sess.run(testing_acc, feed_dict=vali_dict)
#                 print('vali_accuracy:' + str(vali_acc_val))
#
#                 vali_pre_val = str(sess.run(testing_prediction, feed_dict=vali_dict))
#                 print('vali_predic:' + vali_pre_val)
#                 # Print out the actual labels
#                 vali_actual_val = str(vali_label)
#                 print('vali_actual:' + vali_actual_val)

            # if count % 88 == 0:
            #     print('********************************')
            #     summary = sess.run(merged_summary, feed_dict=train_dict)
            #     train_writer.add_summary(summary, count)
            #     summary = sess.run(merged_summary, feed_dict=eval_dict)
            #     eval_writer.add_summary(summary, count)
            #     summary = sess.run(merged_summary, feed_dict=vali_dict)
            #     vali_writer.add_summary(summary, count)
            #     print("saving checkpoint to '" + check_pt_path_str + "'")
            #     saver.save(sess, check_pt_path_str + 'model.ckpt')

            count += 1

if __name__ == "__main__":
    main()