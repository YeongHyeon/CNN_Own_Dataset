print("\n***** Load modules *****")
import os, inspect
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import dataset_loader # custom module
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

def request_dataset(image_dir, rgb=True):

    print("\n***** Load dataset *****")

    dataset, classes = dataset_loader.load_dataset(image_dir=PACK_PATH+"/images", test_percentage=10, validation_percentage=10, imsize=28, rgb=True)

    num_train = dataset.train.images.shape[0]
    num_test = dataset.test.images.shape[0]
    img_length = dataset.train.images[0].shape[0]
    img_sqrt = int(math.sqrt(img_length))
    print(" Num of Train images : "+str(num_train))
    print(" Num of Test images  : "+str(num_test))
    return dataset, classes

def conv_neural_network(img_shape, x, y_, classes=None, rgb=True):
    if(rgb):
        c = 3
        h = int(math.sqrt(image_shape[1]/c))
        w = h
    else:
        c = 3
        h = int(math.sqrt(image_shape[1]/c))
        w = h
    # -1: don't know how long
    # img_sqrt, img_sqrt: image width, height
    # 1: number of color channel
    x_image = tf.reshape(x, [-1, h, w, c])

    print("\n***** Initialize CNN Layers *****")

    print("\n* Layer 1 Init")
    # 5, 5: window size
    # 1: number of input channel
    # 32: number of output channel
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])

    # Convolusion x(input data) and W(weight) and add b(bias)
    # And apply relu function
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # Apply max pooling on (h = x conv W + b)
    h_pool1 = max_pool_2x2(h_conv1)
    print(" "+str(h_pool1))

    print("\n* Layer 2 Init")
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    print(" "+str(h_pool2))

    """
    One benefit of replacing a fully connected layer with a convolutional layer is that the number of parameters to adjust are reduced due to the fact that the weights are shared in a convolutional layer.

    This means faster and more robust learning. Additionally max pooling can be used just after a convolutional layer to reduce the dimensionality of the layer.

    This means improved robustness to distortions in input stimuli and a better overall performance.
    reference: https://www.quora.com/What-are-the-benefits-of-converting-a-fully-connected-layer-in-a-deep-neural-network-to-an-equivalent-convolutional-layer
    """
    print("\n* Fully connected Layer Init")
    # 7*7: frame size : 28*28 -> 14*14 -> 7*7 (caused by max pool)
    # 64: number of output channel of Layer 2
    full_flat = 7 * 7 * 64
    full_con = 1024
    W_fc1 = weight_variable([full_flat, full_con])
    b_fc1 = bias_variable([full_con])

    h_pool2_flat = tf.reshape(h_pool2, [-1, full_flat])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    print(" "+str(h_fc1))

    print("\n* Dropout Layer Init")
    # Prevention overfitting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    print(" "+str(h_fc1_drop))

    print("\n* Softmax Layer Init")
    W_fc2 = weight_variable([full_con, classes])
    b_fc2 = bias_variable([classes])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    print(" "+str(y_conv))

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # return

    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # return

    return keep_prob, train_step, accuracy

def show_train_graph(train_acc_list, test_acc_list):
    # draw graph
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.show()


#============================================================================

dataset, classes = request_dataset(image_dir="images/", rgb=True)
# Separate composition and execute
sess = tf.InteractiveSession()

# Initialize placeholders
# x is image, y_ is label
#x = tf.placeholder(tf.float32, shape=[None, img_length])
#y_ = tf.placeholder(tf.float32, shape=[None, classes])
image_shape = dataset.train.images.shape
x = tf.placeholder(tf.float32, shape=[None, image_shape[1]])
y_ = tf.placeholder(tf.float32, shape=[None, classes])

keep_prob, train_step, accuracy = conv_neural_network(image_shape, x, y_, classes=classes, rgb=True)

print("\n***** Training with CNN *****")
sess.run(tf.global_variables_initializer())

#Create a saver object which will save all the variables
saver = tf.train.Saver()

epochs = 1000;
epoch_step = epochs/10
train_acc_list = []
test_acc_list = []
print("\n Training")

for i in range(epochs):
    # dataset.train.next_batch returns images, labels
    batch = dataset.train.next_batch(50)
    if i%epoch_step == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        test_accuracy = accuracy.eval(feed_dict={x: dataset.test.images, y_: dataset.test.labels, keep_prob: 1.0})
        train_acc_list.append(train_accuracy)
        test_acc_list.append(test_accuracy)
        print(" Step %d, train acc, test acc | %.5f, %.5f\r" %(i, train_accuracy, test_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) # dropout 50%

print("\n Final Test accuracy %g"%accuracy.eval(feed_dict={
    x: dataset.test.images, y_: dataset.test.labels, keep_prob: 1.0}))

saver.save(sess, 'cnn_own_dataset',global_step=1000)

show_train_graph(train_acc_list, test_acc_list)
