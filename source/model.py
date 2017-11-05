import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

def weight_variable(shape):
    # standard deviation = 0.1
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def conv_neural_network(x, y_, height=28, width=28, dimension=1, classes=None):

    print("\n***** Initialize CNN Layers *****")

    print("\n* Layer 1 Init")
    # 5, 5: window size
    # 1: number of input channel
    # 32: number of output channel
    W_conv1 = weight_variable([5, 5, dimension, 32])
    b_conv1 = bias_variable([32])

    # Convolusion x(input data) and W(weight) and add b(bias)
    # And apply relu function
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    # Apply max pooling on (h = x conv W + b)
    h_pool1 = max_pool_2x2(h_conv1)
    print(" "+str(h_pool1.shape))

    print("\n* Layer 2 Init")
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    print(" "+str(h_pool2.shape))

    """
    One benefit of replacing a fully connected layer with a convolutional layer is that the number of parameters to adjust are reduced due to the fact that the weights are shared in a convolutional layer.
    This means faster and more robust learning. Additionally max pooling can be used just after a convolutional layer to reduce the dimensionality of the layer.
    This means improved robustness to distortions in input stimuli and a better overall performance.
    reference: https://www.quora.com/What-are-the-benefits-of-converting-a-fully-connected-layer-in-a-deep-neural-network-to-an-equivalent-convolutional-layer
    """
    print("\n* Fully connected Layer Init")
    # 7*7: frame size : 28*28 -> 14*14 -> 7*7 (caused by max pool)
    # 64: number of output channel of Layer 2
    full_flat = int(h_pool2.shape[1] * h_pool2.shape[2] * h_pool2.shape[3])
    full_con = 1024
    W_fc1 = weight_variable([full_flat, full_con])
    b_fc1 = bias_variable([full_con])

    h_pool2_flat = tf.reshape(h_pool2, [-1, full_flat])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    print(" "+str(h_fc1.shape))

    print("\n* Dropout Layer Init")
    # Prevention overfitting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    print(" "+str(h_fc1_drop.shape))

    print("\n* Softmax Layer Init")
    W_fc2 = weight_variable([full_con, classes])
    b_fc2 = bias_variable([classes])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    print(" "+str(y_conv.shape))

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # return

    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # return

    return keep_prob, train_step, accuracy
