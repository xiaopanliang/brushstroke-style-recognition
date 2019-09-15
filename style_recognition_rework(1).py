# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:10:07 2018

@author: jpansh
"""

import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
import os

check_pt_path_str = 'checkpoints'
batch_size = 16
eval_size = 16
img_height = 256
img_width = 256
epochs = 100
units = np.amax(np.load('train_lbs.npy')) + 1


def conv_layer(name, layer_input, w):
    return tf.nn.conv2d(layer_input, w, strides=[1, 1, 1, 1], padding='SAME', name=name)


def relu_layer(name, layer_input, b):
    return tf.nn.relu(layer_input + b, name=name)


def pool_layer(name, layer_input):
    return tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def fc_layer(name, layer_input, units, activation):
    return tf.layers.dense(inputs=layer_input, units=units, name=name,
                           activation=activation)  # TODO: Add the units here


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
    style_layer_weights = [0.1, 0.1, 0.2, 0.3, 0.3]
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
        G = tf.matmul(tf.transpose(F, perm=[0, 2, 1]), F) * weight

        # Keep recording gram matrices from different layers
        n += 1
        net['gram_' + str(n)] = G

        _, y, x = G.get_shape()
        G = tf.reshape(G, [-1, y.value * x.value])
        if connected_G is None:
            connected_G = G
        else:
            connected_G = tf.concat([connected_G, G], axis=1)
    return connected_G


def calc_obj_logits(net):
    # Load the content image to the model
    object_output = net['pool5']
    _, height, width, depth = object_output.get_shape()
    object_output = tf.reshape(object_output, [-1, height.value * width.value * depth.value])
    return object_output


def conclude_prediction(prediction):
    style_counts = []
    for num in range(units):
        equal_op = tf.equal(prediction, num)
        as_ints = tf.cast(equal_op, tf.int8)
        counts = tf.reduce_sum(as_ints)
        style_counts.append(counts)
    return tf.argmax(input=style_counts)


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
    texture_logits = calc_texture_logits(net)
    net['fc1_t'] = fc_layer('fc1_t', texture_logits, 512, tf.nn.relu)
    net['fc2_t'] = fc_layer('fc2_t', tf.layers.dropout(net['fc1_t'],0), 512, tf.nn.relu)
    net['fc3_t'] = fc_layer('fc3_t', tf.layers.dropout(net['fc2_t'],0), units, None)
    
    logits = net['fc3_t']

    sum_logits = tf.reduce_mean(logits, 0, keepdims=True)
    sum_logits = tf.concat([sum_logits] * batch_size, 0)

    # obj_logits = calc_obj_logits(net)


    # net['fc1_t'] = fc_layer('fc1_tepochs', texture_logits, 512, tf.nn.relu)
    # net['fc2_t'] = fc_layer('fc2_t', tf.layers.dropout(net['fc1_t'],0), 512, tf.nn.relu)
    # net['fc3_t'] = fc_layer('fc3_t', tf.layers.dropout(net['fc2_t'],0), units, None)

    # net['fc1_o'] = fc_layer('fc1_oepochs', obj_logits, 512, tf.nn.relu)
    # net['fc2_o'] = fc_layer('fc2_o', tf.layers.dropout(net['fc1_o'],0), 512, tf.nn.relu)
    # net['fc3_o'] = fc_layer('fc3_o', tf.layers.dropout(net['fc2_o'],0), units, None)
    
    # logits = tf.abs(net['fc3_t'] * 1 + net['fc3_o'] *0)

    # sum_logits1 = tf.reduce_mean(logits[0:3], 0, keepdims=True)
    # sum_logits2 = tf.reduce_mean(logits[4:7], 0, keepdims=True)
    # sum_logits3 = tf.reduce_mean(logits[8:11], 0, keepdims=True)
    # sum_logits4 = tf.reduce_mean(logits[12:15], 0, keepdims=True)
    # sum_logits = tf.concat([sum_logits1]*eval_size+[sum_logits2]*eval_size+ \
    #     [sum_logits3]*eval_size+[sum_logits4]*eval_size, 0)
    # sum_logits1 = tf.reduce_mean(logits, 0, keepdims=True)
    # sum_logits = tf.concat([sum_logits1] * eval_size, 0)
    # sum_logits = logits

    train_prediction = tf.argmax(input=logits, axis=1, name="train_prediction")
    eval_prediction = tf.argmax(input=sum_logits, axis=1, name='eval_prediction')
    
    acc, acc_op = tf.metrics.accuracy(labels=net['labels'], predictions=eval_prediction)

    # l2_regularizer = tf.contrib.layers.l2_regularizer(
    #      scale=0.00005, scope=None
    #  )
    # weights = tf.trainable_variables()  # all vars of your graph
    # regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=net['labels'], logits=logits)
    eval_loss = tf.losses.sparse_softmax_cross_entropy(labels=net['labels'], logits=logits)
    # conf, conf_op = tf.confusion_matrix(labels=net['labels'], predictions=testing_prediction)

  
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=net['labels'], logits=obj_logits) + regularization_penalty + tf.losses.sparse_softmax_cross_entropy(labels=net['labels'], logits=texture_logits)
    # tex_loss = regularization_penalty + tf.losses.sparse_softmax_cross_entropy(labels=net['labels'], logits=texture_logits)
    # Prediction is the prediction for each piece. Predictions are prediction for each image when eval, it is not
    # important under train mode
    # Labels are for the whole image instead of each piece
    return loss, eval_loss,\
           acc, acc_op, \
           net['labels'], \
           train_prediction, \
           eval_prediction, \
           logits, \
           sum_logits


def load_imgs(img_path, label):
    img_string = tf.read_file(img_path)
    img = tf.image.decode_image(img_string, 0)
    img = tf.image.resize_image_with_pad(img, img_height, img_width)
    # img = tf.image.grayscale_to_rgb(img)
    # img = tf.image.adjust_contrast(img,10)
    img = tf.image.per_image_standardization(img)
    return img, label


def get_train_iterator():
    img_files = np.load('train_imgs.npy')
    labels = np.load('train_lbs.npy')
    dataset = tf.data.Dataset.from_tensor_slices((img_files, labels))
    dataset = dataset.shuffle(300000)
    dataset = dataset.map(map_func=load_imgs)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_initializable_iterator()

    return iterator


def get_eval_iterator():
    img_files = np.load('eval_imgs.npy')
    img_files = np.sort(img_files)
    labels = np.load('eval_lbs.npy')
    labels = np.sort(labels)

    dataset = tf.data.Dataset.from_tensor_slices((img_files, labels))
    dataset = dataset.map(map_func=load_imgs)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_initializable_iterator()
    
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


def eval(sess, acc_op, acc, predictions, labels):
    with tf.device('/cpu:0'):
        itr_eval = get_eval_iterator()
        next_eval_batch = itr_eval.get_next()
    batch_num = 0
    with tf.device('/device:XLA_GPU:0'):
        while True:
            try:
                print('********************************')
                print('processing batch:' + str(batch_num))
                eval_data, eval_label = sess.run(next_eval_batch)
                eval_dict = {net['input']: eval_data, net['labels']: eval_label}
                # Update the accuracy
                sess.run(acc_op, feed_dict=eval_dict)
                # Compute labels and predictions
                labels_str = str(sess.run(labels, feed_dict=eval_dict))
                predictions_str = str(sess.run(predictions, feed_dict=eval_dict))
                print("labels:", labels_str)
                print("predictions:", predictions_str)
                # Update the batch number
                batch_num += 1
            except tf.errors.OutOfRangeError:
                # Determine if the end of the eval dataset is reached
                break
    acc_str = str(sess.run(acc))
    print("accuracy:", acc_str)


def main(Command):
    tf.logging.set_verbosity(tf.logging.INFO)

    loss, eval_loss,\
    acc, acc_op, \
    labels, \
    train_prediction, \
    eval_prediction, \
    logits, \
    eval_logit = cnn_model_fn()

    optimizer = tf.train.AdamOptimizer(learning_rate=0.000001, epsilon=1e-9, use_locking=False)

    train_op = optimizer.minimize(loss)

    # Get the iterator for the data set
    itr_train = get_train_iterator()
    next_train_batch = itr_train.get_next()

    itr_eval = get_eval_iterator()
    next_eval_batch = itr_eval.get_next()

    # Define the saver for storing variables
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=0)

    # The counter for tracking the number of batches
    
    with tf.device('/device:XLA_GPU:0'), tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        latest_checkpoint = tf.train.latest_checkpoint(check_pt_path_str)
        if latest_checkpoint is not None:
            # saver.restore(sess, latest_checkpoint)
            saver.restore(sess, 'checkpoints/epoch_70_model.ckpt')

        if Command == "train":            
            epoch_loss_list = []
            eval_loss_list = []
            for idx_epoch in range(epochs):
                # Reset the iterator
                sess.run(itr_train.initializer)
                sess.run(itr_eval.initializer)
                # Var for loss calculation
                batch_count = 0
                batch_loss = 0
                avg_loss = 0
                print("Start Training for epoch:",idx_epoch,"...")
                while True:
                    try:
                        train_data, train_label = sess.run(next_train_batch)
                        _, _, _, channel = train_data.shape

                        if channel == 3:
                            train_dict = {net['input']: train_data, net['labels']: train_label}

                            # Train the model
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                            with tf.control_dependencies(update_ops):
                                loss_val = sess.run(loss, feed_dict=train_dict)
                                sess.run(train_op, feed_dict=train_dict)
                                
                            batch_count += 1
                            batch_loss += loss_val
                            avg_loss = batch_loss/batch_count

                            # print('Epoch', idx_epoch, ' batch:', batch_count, \
                            #     ' Avg_loss:', avg_loss, ' Loss:', loss_val)
                    except:
                        print("Batch Size Error, Skip to next")
                        break

                # print("labels:" + str(sess.run(net['labels'], feed_dict=train_dict)))
                # print("prediction:" + str(sess.run(train_prediction, feed_dict=train_dict)))
                # plt.plot(batch_loss_list) # Plot batch loss
                # plt.show()

                print('Epoch', idx_epoch, ' Avg_loss:', avg_loss)
                epoch_loss_list.append(avg_loss)
                if idx_epoch % 10 == 0:
                    print("saving checkpoint every 2 epoches...")
                    saver.save(sess, check_pt_path_str + '/epoch_' + str(idx_epoch) + '_model.ckpt')
                
                # Calc Eval loss
                batch_count = 0
                batch_loss = 0
                avg_loss = 0
                while True:
                    try:
                        eval_data, eval_label = sess.run(next_eval_batch)
                        eval_dict = {net['input']: eval_data, net['labels']: eval_label}
                        loss_val = sess.run(eval_loss, feed_dict=eval_dict)
                        batch_loss += loss_val
                        batch_count += 1
                    except:
                        break
                avg_loss = batch_loss / batch_count
                eval_loss_list.append(avg_loss)
                print('Epoch', idx_epoch, 'Eval Loss:', avg_loss)
            
            figure(dpi=100)
            plt.plot(epoch_loss_list, linewidth=2)
            plt.plot(eval_loss_list, linewidth=2)
            plt.show()

        elif Command == "eval":
            print("evaluating...")
            sess.run(itr_eval.initializer)
            sess.run(itr_train.initializer)
            ind = 0
            while ind <= 3000:
                eval_data, eval_label = sess.run(next_eval_batch)
                # train_data, train_label = sess.run(next_train_batch)
                # if eval_label[0] != 5 and eval_label[0] != 6:
                print(ind)
                eval_dict = {net['input']: eval_data, net['labels']: eval_label}
                sess.run(acc_op, feed_dict=eval_dict)
                # sess.run(acc_op, feed_dict=eval_dict)
                eval_pre_val = str(sess.run(eval_prediction, feed_dict=eval_dict))
                print('eval_predict:' + eval_pre_val)

                eval_actual_val = str(eval_label)
                print('eval_actual: ' + eval_actual_val)

                # eval_ori_log = str(sess.run(logits, feed_dict=eval_dict))
                # print('eval_ori_log: ' + eval_ori_log)

                # eval_log = str(sess.run(eval_logit, feed_dict=eval_dict))
                # print('eval_log: ' + eval_log)

                eval_acc_val = sess.run(acc, feed_dict=eval_dict)
                print('vali_accuracy:' + str(eval_acc_val))
                ind += 1

        else:
            print("unrecognized mode!!!")


if __name__ == "__main__":
    main('train')
