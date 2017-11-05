import os, inspect
import source.utility as util

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def training(steps=None, batch_size=None, dataset=None, sess=None, keep_prob=None, train_step=None, accuracy=None, x=None, y_=None, saver=None):

    print_step = 10
    if(steps <= 10):
        print_step = 1

    train_acc_list = []
    test_acc_list = []
    print("\n Training... steps: %d, batch size: %d\n" %(steps, batch_size))

    for i in range(steps):
        # dataset.train.next_batch returns images, labels
        train = dataset.train.next_batch(batch_size)
        if i%print_step == 0:
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

    util.show_graph(train_acc_list, test_acc_list)
    util.save_graph_as_image(train_acc_list, test_acc_list)
