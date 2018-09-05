import tensorflow as tf
import numpy as np
import scipy.io

check_pt_path_str = 'checkpoint'
batch_size = 2
img_height = 500
img_width = 500


def conv_layer(name, layer_input, w):
    return tf.nn.conv2d(layer_input, w, strides=[1, 1, 1, 1], padding='SAME', name=name)


def relu_layer(name, layer_input, b):
    return tf.nn.relu(layer_input + b, name=name)


def pool_layer(name, layer_input):
    return tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def fc_layer(name, layer_input):
    return tf.layers.dense(inputs=layer_input, units=2, name=name)  # TODO: Add the units here


def get_weights(name, vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    return tf.Variable(weights, name=name)


def get_bias(name, vgg_layers, i):
    bias = vgg_layers[i][0][0][2][0][1]
    b = tf.Variable(np.reshape(bias, bias.size), name=name)
    return b


def calc_texture_logits(net):
    # Load the style image to the model
    layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

    connected_G = None
    for layer in layers:
        layer_data = net[layer]
        _, height, width, channels = layer_data.get_shape()
        M = height.value * width.value
        N = channels.value
        F = tf.reshape(layer_data, [-1, M, N])
        G = tf.matmul(tf.transpose(F, perm=[0, 2, 1]), F)
        _, y, x = G.get_shape()
        G = tf.reshape(G, [-1, y.value * x.value])
        if connected_G is None:
            connected_G = G
        else:
            connected_G = tf.concat([connected_G, G], axis=1)

    net['g_fc'] = fc_layer('g', connected_G)
    return net['g_fc']


def calc_obj_logits(net):
    # Load the content image to the model
    object_output = net['pool5']
    _, height, width, depth = object_output.get_shape()
    object_output = tf.reshape(object_output, [-1, height.value * width.value * depth.value])
    net['o_fc'] = fc_layer('o', object_output)
    return net['o_fc']


net = {}


def cnn_model_fn():
    print('\nBUILDING VGG-19 NETWORK')

    print('loading model weights...')
    vgg_raw_net = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
    vgg_layers = vgg_raw_net['layers'][0]
    print('constructing layers...')
    net['input'] = tf.Variable(np.zeros([batch_size, img_height, img_width, 3]), dtype=tf.float32, trainable=False)
    net['labels'] = tf.Variable(np.zeros([batch_size]), dtype=tf.int32, trainable=False)

    print('LAYER GROUP 1')
    net['conv1_1'] = conv_layer('conv1_1', net['input'], w=get_weights('conv1_1_w', vgg_layers, 0))
    net['relu1_1'] = relu_layer('relu1_1', net['conv1_1'], b=get_bias('relu1_1_b', vgg_layers, 0))

    net['conv1_2'] = conv_layer('conv1_2', net['relu1_1'], w=get_weights('conv1_2_w', vgg_layers, 2))
    net['relu1_2'] = relu_layer('relu1_2', net['conv1_2'], b=get_bias('relu1_2_b', vgg_layers, 2))

    net['pool1'] = pool_layer('pool1', net['relu1_2'])

    print('LAYER GROUP 2')
    net['conv2_1'] = conv_layer('conv2_1', net['pool1'], w=get_weights('conv2_1_w', vgg_layers, 5))
    net['relu2_1'] = relu_layer('relu2_1', net['conv2_1'], b=get_bias('relu2_1_b', vgg_layers, 5))

    net['conv2_2'] = conv_layer('conv2_2', net['relu2_1'], w=get_weights('conv2_2_w', vgg_layers, 7))
    net['relu2_2'] = relu_layer('relu2_2', net['conv2_2'], b=get_bias('relu2_2_b', vgg_layers, 7))

    net['pool2'] = pool_layer('pool2', net['relu2_2'])

    print('LAYER GROUP 3')
    net['conv3_1'] = conv_layer('conv3_1', net['pool2'], w=get_weights('conv3_1_w', vgg_layers, 10))
    net['relu3_1'] = relu_layer('relu3_1', net['conv3_1'], b=get_bias('relu3_1_b', vgg_layers, 10))

    net['conv3_2'] = conv_layer('conv3_2', net['relu3_1'], w=get_weights('conv3_2_w', vgg_layers, 12))
    net['relu3_2'] = relu_layer('relu3_2', net['conv3_2'], b=get_bias('relu3_2_b', vgg_layers, 12))

    net['conv3_3'] = conv_layer('conv3_3', net['relu3_2'], w=get_weights('conv3_3_w', vgg_layers, 14))
    net['relu3_3'] = relu_layer('relu3_3', net['conv3_3'], b=get_bias('relu3_3_b', vgg_layers, 14))

    net['conv3_4'] = conv_layer('conv3_4', net['relu3_3'], w=get_weights('conv3_4_w', vgg_layers, 16))
    net['relu3_4'] = relu_layer('relu3_4', net['conv3_4'], b=get_bias('relu3_4_b', vgg_layers, 16))

    net['pool3'] = pool_layer('pool3', net['relu3_4'])

    print('LAYER GROUP 4')
    net['conv4_1'] = conv_layer('conv4_1', net['pool3'], w=get_weights('conv4_1_w', vgg_layers, 19))
    net['relu4_1'] = relu_layer('relu4_1', net['conv4_1'], b=get_bias('relu4_1_b', vgg_layers, 19))

    net['conv4_2'] = conv_layer('conv4_2', net['relu4_1'], w=get_weights('conv4_2_w', vgg_layers, 21))
    net['relu4_2'] = relu_layer('relu4_2', net['conv4_2'], b=get_bias('relu4_2_b', vgg_layers, 21))

    net['conv4_3'] = conv_layer('conv4_3', net['relu4_2'], w=get_weights('conv4_3_w', vgg_layers, 23))
    net['relu4_3'] = relu_layer('relu4_3', net['conv4_3'], b=get_bias('relu4_3_b', vgg_layers, 23))

    net['conv4_4'] = conv_layer('conv4_4', net['relu4_3'], w=get_weights('conv4_4_w', vgg_layers, 25))
    net['relu4_4'] = relu_layer('relu4_4', net['conv4_4'], b=get_bias('relu4_4_b', vgg_layers, 25))

    net['pool4'] = pool_layer('pool4', net['relu4_4'])

    print('LAYER GROUP 5')
    net['conv5_1'] = conv_layer('conv5_1', net['pool4'], w=get_weights('conv5_1_w', vgg_layers, 28))
    net['relu5_1'] = relu_layer('relu5_1', net['conv5_1'], b=get_bias('relu5_1_b', vgg_layers, 28))

    net['conv5_2'] = conv_layer('conv5_2', net['relu5_1'], w=get_weights('conv5_2', vgg_layers, 30))
    net['relu5_2'] = relu_layer('relu5_2', net['conv5_2'], b=get_bias('relu5_2', vgg_layers, 30))

    net['conv5_3'] = conv_layer('conv5_3', net['relu5_2'], w=get_weights('conv5_3', vgg_layers, 32))
    net['relu5_3'] = relu_layer('relu5_3', net['conv5_3'], b=get_bias('relu5_3', vgg_layers, 32))

    net['conv5_4'] = conv_layer('conv5_4', net['relu5_3'], w=get_weights('conv5_4', vgg_layers, 34))
    net['relu5_4'] = relu_layer('relu5_4', net['conv5_4'], b=get_bias('relu5_4', vgg_layers, 34))

    net['pool5'] = pool_layer('pool5', net['relu5_4'])

    # Fully connected to get logits
    obj_logits = calc_obj_logits(net)
    texture_logits = calc_texture_logits(net)
    logits = obj_logits * 0.05 + texture_logits * 0.95 + 1e-8

    prediction = tf.argmax(input=logits, axis=1, name='prediction')
    acc, acc_op = tf.metrics.accuracy(labels=net['labels'], predictions=prediction)

    #    Denoise_Loss = tf.image.total_variation(net['input']) * 0.001
    loss = tf.losses.sparse_softmax_cross_entropy(labels=net['labels'], logits=logits)
    # loss = tf.Print(loss, [loss], 'loss:')

    return loss, acc, acc_op, prediction


def load_imgs(img_path, label):
    img_string = tf.read_file(img_path)
    img_decoded = tf.image.decode_png(img_string)
    img_resized = tf.image.resize_images(img_decoded, [img_height, img_width])
    img = tf.image.grayscale_to_rgb(img_resized)
    return img, label


def get_iterator():
    img_files = np.load('train_imgs.npy')
    labels = np.load('train_lbs.npy')
    dataset = tf.data.Dataset.from_tensor_slices((img_files, labels))
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(map_func=load_imgs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    loss, acc, acc_op, prediction = cnn_model_fn()

    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                       method='L-BFGS-B',
                                                       options={'maxiter': 3000,
                                                                'disp': 50,
                                                                'eps': 1e-08,
                                                                'gtol': 1e-06,
                                                                'ftol': 2.220446049250313e-10})
    # Get the iterator for the data set
    itr = get_iterator()
    next_batch = itr.get_next()

    # The counter for tracking the number of batches
    count = 0

    saver = tf.train.Saver(tf.trainable_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print('restoring model...')
        test_weight = tf.get_default_graph().get_tensor_by_name('conv1_1_w:0')
        prev = sess.run(test_weight)
        saver.restore(sess, tf.train.latest_checkpoint('checkpoint/'))
        after = sess.run(test_weight)
        # Calc difference before read and after read
        diff = after - prev
        diff = diff.sum()
        print('restored:' + str(diff))

        while True:
            try:
                print('processing batch:' + str(count))
                data, label = sess.run(next_batch)
                _, _, _, channel = data.shape
                if channel == 3:
                    sess.run(net['input'].assign(data))
                    sess.run(net['labels'].assign(label))
                    # sess.run(train_op)
                    optimizer.minimize(sess)
                    sess.run(acc_op)
                    acc_val = str(sess.run(acc))
                    print('accuracy:' + acc_val)

                    pre_val = str(sess.run(prediction))
                    print('prediction:' + pre_val)

                    actual_val = str(label)
                    print('actual:' + actual_val)

                    if count % 10 == 0:
                        saver.save(sess, 'checkpoint/model.ckpt')

                    count += 1
            except tf.errors.OutOfRangeError:
                print("End of dataset")  # ==> "End of dataset"
                break
            except:
                print('Error occurs on processing batch:' + str(count))


if __name__ == "__main__":
    main()
