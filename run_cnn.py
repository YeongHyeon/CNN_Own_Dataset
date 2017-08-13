print("\n***** Load modules *****")
import os, inspect, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# custom module
import dataset_loader
import utility
print(" Load module complete")

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

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

def request_dataset(path):

    print("\n***** Load dataset *****")

    dataset, classes = dataset_loader.load_dataset(path=PACK_PATH+"/images", img_h=FLAGS.height, img_w=FLAGS.width)

    num_train = dataset.train.amount
    num_test = dataset.test.amount
    print(" Num of Train images : "+str(num_train))
    print(" Num of Test images  : "+str(num_test))
    return dataset, classes, min(num_train, num_test)

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

#============================================================================
def main():
    dataset, classes, min_b = request_dataset(path="images/")
    # Separate composition and execute
    sess = tf.InteractiveSession()

    # Initialize placeholdersshape[0]
    # x is image, y_ is label
    #x = tf.placeholder(tf.float32, shape=[None, img_length])
    #y_ = tf.placeholder(tf.float32, shape=[None, classes])
    height, width, dimension = dataset.train.shape
    x = tf.placeholder(tf.float32, shape=[None, height, width, dimension])
    y_ = tf.placeholder(tf.float32, shape=[None, classes])

    keep_prob, train_step, accuracy = conv_neural_network(x, y_, height=height, width=width, dimension=dimension, classes=classes)

    print("\n***** Training with CNN *****")
    sess.run(tf.global_variables_initializer())
    print(" Initialized all variables")

    #Create a saver object which will save all the variables
    saver = tf.train.Saver()

    batch_size = FLAGS.batch
    if(batch_size > min_b):
        batch_size = min_b

    epochs=FLAGS.epochs
    epoch_step = 10
    if(epochs <= 10):
        epoch_step = 1

    train_acc_list = []
    test_acc_list = []
    print("\n Training... epochs: %d, batch size: %d\n" %(epochs, batch_size))

    for i in range(epochs):
        # dataset.train.next_batch returns images, labels
        train = dataset.train.next_batch(batch_size)
        if i%epoch_step == 0:
            test = dataset.test.next_batch(batch_size)

            train_accuracy = accuracy.eval(feed_dict={x:train[0], y_:train[1], keep_prob: 1.0})
            test_accuracy = accuracy.eval(feed_dict={x:test[0], y_:test[1], keep_prob: 1.0})
            train_acc_list.append(train_accuracy)
            test_acc_list.append(test_accuracy)
            print(" Step %d\n\ttrain acc, test acc | %.5f, %.5f\r" %(i, train_accuracy, test_accuracy))
        train_step.run(feed_dict={x:train[0], y_:train[1], keep_prob: 0.5}) # dropout 50%

    test = dataset.test.next_batch(batch_size)
    print("\n Final Test accuracy %g"%accuracy.eval(feed_dict={x:test[0], y_:test[1], keep_prob: 1.0}))

    if(not(os.path.exists("./checkpoint"))):
        os.mkdir("./checkpoint")
    else:
        pass
    saver.save(sess, "./checkpoint/checkpoint",global_step=1000)

    utility.show_graph(train_acc_list, test_acc_list)
    utility.save_graph_as_image(train_acc_list, test_acc_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Default: 100')
    parser.add_argument('--batch', type=int, default=10, help='Default: 10. Batches per iteration, the number of data to be training and testing.')
    parser.add_argument('--height', type=float, default=1, help='Default: 28. Enter the size to resize images what you want')
    parser.add_argument('--width', type=float, default=1, help='Default: 28. Enter the size to resize images what you want')
    FLAGS, unparsed = parser.parse_known_args()

    main()
