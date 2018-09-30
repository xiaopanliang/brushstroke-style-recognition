import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

check_pt_path_str = 'checkpoint/'
batch_size = 2
img_height = 64
img_width = 64
epochs = 100
units = 3


def conv_layer(name, layer_input, w):
    return tf.nn.conv2d(layer_input, w, strides=[1, 1, 1, 1], padding='SAME', name=name)


def relu_layer(name, layer_input, b):
    return tf.nn.relu(layer_input + b, name=name)


def pool_layer(name, layer_input):
    return tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def get_weights(name, vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    w = tf.Variable(weights, name=name)
    return w


def get_bias(name, vgg_layers, i):
    bias = vgg_layers[i][0][0][2][0][1]
    b = tf.Variable(np.reshape(bias, bias.size), name=name)  # TODO: check later
    return b


def calc_texture_gram(net):
    # Load the style image to the model
    layers = ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']

    connected_G = None
    for layer in layers:
        layer_data = net[layer]
        _, height, width, channels = layer_data.get_shape()
        M = height.value * width.value
        N = channels.value
        F = tf.reshape(layer_data, [-1, M, N])
        G = tf.matmul(tf.transpose(F, perm=[0, 2, 1]), F)  # [img_num, M, N]
        _, y, x = G.get_shape()
        G = tf.reshape(G, [-1, y.value * x.value])  # [img_num, M * N]
        if connected_G is None:
            connected_G = G
        else:
            connected_G = tf.concat([connected_G, G], axis=1)

    return connected_G


def calc_texture_logits(net, vgg_layers):
    gram = calc_texture_gram(net)
    fc1 = tf.layers.dense(inputs=gram,
                          units=4096,
                          kernel_initializer=tf.constant_initializer(vgg_layers[37][0][0][2][0][0]),
                          bias_initializer=tf.constant_initializer(vgg_layers[37][0][0][2][0][1]),
                          trainable=False,
                          activation=tf.nn.relu)
    fc2 = tf.layers.dense(inputs=fc1,
                          units=4096,
                          kernel_initializer=tf.constant_initializer(vgg_layers[39][0][0][2][0][0]),
                          bias_initializer=tf.constant_initializer(vgg_layers[39][0][0][2][0][1]),
                          trainable=False,
                          activation=tf.nn.relu)
    fc3 = tf.layers.dense(inputs=fc2,
                          units=units,
                          kernel_initializer=tf.constant_initializer(vgg_layers[41][0][0][2][0][0]),
                          bias_initializer=tf.constant_initializer(vgg_layers[41][0][0][2][0][1]),
                          trainable=False,
                          activation=None)
    return fc3


def pinv(A, b, reltol=1e-6):
    # Compute the SVD of the input matrix A
    s, u, v = tf.svd(A)

    # Invert s, clear entries lower than reltol*s[0].
    atol = tf.reduce_max(s) * reltol
    s = tf.boolean_mask(s, s > atol)
    s_inv = tf.diag(tf.concat([1. / s, tf.zeros([tf.size(b) - tf.size(s)])], 0))

    # Compute v * s_inv * u_t * b from the left to avoid forming large intermediate matrices.
    return tf.matmul(v, tf.matmul(s_inv, tf.matmul(u, tf.reshape(b, [-1, 1]), transpose_a=True)))


def calc_label_gram(net, vgg_layers):
    inverse3 = tf.get_variable(name='inverse3',
                               initializer=tf.constant_initializer(
                                   pinv(vgg_layers[41][0][0][2][0][0], vgg_layers[41][0][0][2][0][1])
                               ),
                               trainable=False)
    fc2 = tf.matmul(net['labels'], inverse3)
    inverse2 = tf.get_variable(name='inverse2',
                               initializer=tf.constant_initializer(
                                   pinv(vgg_layers[39][0][0][2][0][0], vgg_layers[39][0][0][2][0][1])
                               ),
                               trainable=False)
    fc1 = tf.matmul(fc2, inverse2)
    inverse1 = tf.get_variable(name='inverse1',
                               initializer=tf.constant_initializer(
                                   pinv(vgg_layers[37][0][0][2][0][0], vgg_layers[37][0][0][2][0][1])
                               ),
                               trainable=False)
    gram_matrix = tf.matmul(fc1, inverse1)
    return gram_matrix


def calc_obj_logits(net, vgg_layers):
    # Load the content image to the model
    object_output = net['pool5']
    _, height, width, depth = object_output.get_shape()
    object_output = tf.reshape(object_output, [-1, height.value * width.value * depth.value])
    fc1 = tf.layers.dense(inputs=object_output,
                          units=4096,
                          kernel_initializer=tf.constant_initializer(vgg_layers[37][0][0][2][0][0]),
                          bias_initializer=tf.constant_initializer(vgg_layers[37][0][0][2][0][1]),
                          trainable=True,
                          activation=tf.nn.relu)
    fc2 = tf.layers.dense(inputs=fc1,
                          units=4096,
                          kernel_initializer=tf.constant_initializer(vgg_layers[39][0][0][2][0][0]),
                          bias_initializer=tf.constant_initializer(vgg_layers[39][0][0][2][0][1]),
                          trainable=True,
                          activation=tf.nn.relu)
    fc3 = tf.layers.dense(inputs=fc2,
                          units=units,
                          kernel_initializer=tf.constant_initializer(vgg_layers[41][0][0][2][0][0]),
                          bias_initializer=tf.constant_initializer(vgg_layers[41][0][0][2][0][1]),
                          trainable=True,
                          activation=None)
    return fc3


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
    net['bn1_2'] = tf.layers.batch_normalization(net['conv1_2'], training=False, momentum=0.9)
    net['relu1_2'] = relu_layer('relu1_2', net['bn1_2'], b=get_bias('relu1_2_b', vgg_layers, 2))

    net['pool1'] = pool_layer('pool1', net['relu1_2'])

    print('LAYER GROUP 2')
    net['conv2_1'] = conv_layer('conv2_1', net['pool1'], w=get_weights('conv2_1_w', vgg_layers, 5))
    #    net['bn2_1'] = tf.layers.batch_normalization(net['conv2_1'], training=False, momentum=0.9)
    net['relu2_1'] = relu_layer('relu2_1', net['conv2_1'], b=get_bias('relu2_1_b', vgg_layers, 5))

    net['conv2_2'] = conv_layer('conv2_2', net['relu2_1'], w=get_weights('conv2_2_w', vgg_layers, 7))
    net['bn2_2'] = tf.layers.batch_normalization(net['conv2_2'], training=False, momentum=0.9)
    net['relu2_2'] = relu_layer('relu2_2', net['bn2_2'], b=get_bias('relu2_2_b', vgg_layers, 7))

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
    net['bn3_4'] = tf.layers.batch_normalization(net['conv3_4'], training=False, momentum=0.9)
    net['relu3_4'] = relu_layer('relu3_4', net['bn3_4'], b=get_bias('relu3_4_b', vgg_layers, 16))

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
    net['bn4_4'] = tf.layers.batch_normalization(net['conv4_4'], training=False, momentum=0.9)
    net['relu4_4'] = relu_layer('relu4_4', net['bn4_4'], b=get_bias('relu4_4_b', vgg_layers, 25))

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
    net['bn5_4'] = tf.layers.batch_normalization(net['conv5_4'], training=False, momentum=0.9)
    net['relu5_4'] = relu_layer('relu5_4', net['bn5_4'], b=get_bias('relu5_4', vgg_layers, 34))

    net['pool5'] = pool_layer('pool5', net['relu5_4'])

    # Fully connected to get logits
    obj_logits = calc_obj_logits(net, vgg_layers)
    texture_logits = calc_obj_logits(net, vgg_layers)

    texture_gram = calc_texture_gram(net)
    label_gram = calc_label_gram(net, vgg_layers)


    prediction = tf.argmax(input=logits, axis=1, name='prediction')
    acc, acc_op = tf.metrics.accuracy(labels=net['labels'], predictions=prediction)

    l1_regularizer = tf.contrib.layers.l2_regularizer(
        scale=0.0001, scope=None
    )

    weights = tf.trainable_variables()  # all vars of your graph

    regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=net['labels'], logits=logits) + regularization_penalty

    return loss, acc, acc_op, prediction, logits, lo


def load_imgs(img_path, label):
    img_string = tf.read_file(img_path)
    img_decoded = tf.image.decode_png(img_string)
    img_resized = tf.image.resize_images(img_decoded, [img_height, img_width])
    img = tf.image.grayscale_to_rgb(img_resized)
    return img, label


def get_train_iterator():
    img_files = np.load('train_imgs.npy')
    labels = np.load('train_lbs.npy')
    dataset = tf.data.Dataset.from_tensor_slices((img_files, labels))
    dataset = dataset.repeat(epochs)
    dataset = dataset.shuffle(len(labels))
    dataset = dataset.map(map_func=load_imgs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator


def get_vali_iterator():
    img_files = np.load('vali_imgs.npy')
    labels = np.load('vali_lbs.npy')
    dataset = tf.data.Dataset.from_tensor_slices((img_files, labels))
    dataset = dataset.shuffle(1)
    dataset = dataset.map(map_func=load_imgs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    loss, acc, acc_op, prediction, logits, lo = cnn_model_fn()

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001 / 2, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                       use_locking=False)

    train_op = optimizer.minimize(loss)

    # Get the iterator for the data set
    itr_train = get_train_iterator()
    next_train_batch = itr_train.get_next()

    itr_vali = get_vali_iterator()
    next_vali_batch = itr_vali.get_next()

    # The counter for tracking the number of batches
    count = 0

    saver = tf.train.Saver(tf.trainable_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print('restoring model...')
        test_weight = tf.get_default_graph().get_tensor_by_name('conv1_1_w:0')
        prev = sess.run(test_weight)
        # saver.restore(sess, tf.train.latest_checkpoint(check_pt_path_str))
        after = sess.run(test_weight)
        # Calc difference before read and after read
        diff = after - prev
        diff = diff.sum()
        print('restored:' + str(diff))

        while True:
            try:
                train_data, train_label = sess.run(next_train_batch)
                vali_data, vali_label = sess.run(next_vali_batch)
                print('********************************')
                print('processing batch:' + str(count))
                _, _, _, channel = train_data.shape
                if channel == 3:
                    train_dict = {net['input']: train_data, net['labels']: train_label}

                    sess.run(train_op, feed_dict=train_dict)

                    # Update the accuracy
                    sess.run(acc_op, feed_dict=train_dict)
                    # Print out the accuracy value
                    acc_val = sess.run(acc, feed_dict=train_dict)
                    loss_val = sess.run(loss, feed_dict=train_dict)
                    print('train_accuracy:' + str(acc_val))
                    print('train_loss:' + str(loss_val))
                    # Print out the prediction
                    pre_val = str(sess.run(prediction, feed_dict=train_dict))
                    print('train_prediction:' + pre_val)
                    # Print out the actual labels
                    actual_val = str(train_label)
                    print('train_actual:' + actual_val)

                    # vali_dict = {net['input']: vali_data, net['labels']: vali_label}
                    #
                    # vali_pre_lo = str(sess.run(lo, feed_dict=vali_dict))
                    # print('vali_pred_lo:' + vali_pre_lo)
                    # vali_pre_val = str(sess.run(prediction, feed_dict=vali_dict))
                    # print('vali_predic:' + vali_pre_val)
                    # # Print out the actual labels
                    # vali_actual_val = str(vali_label)
                    # print('vali_actual:' + vali_actual_val)

                if count % 10 == 0:
                    print('********************************')
                    print("saving checkpoint to '" + check_pt_path_str + "'")
                    saver.save(sess, check_pt_path_str + 'model.ckpt')

                count += 1
            except tf.errors.OutOfRangeError:
                print("End of dataset")  # ==> "End of dataset"
                break
            except:
                print('Error occurs on processing batch:' + str(count))


if __name__ == "__main__":
    main()
