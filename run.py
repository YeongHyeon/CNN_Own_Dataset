import source.developed as dev
dev.print_stamp()

print("\n***** Load modules *****")
import os, argparse
import tensorflow as tf

# custom module
import source.constructor as cntrt
import source.model as model
import source.sub_procedure as subp
print(" Load module complete")

def main():
    usr_path = ""
    print("Enter the path where the images are located.")
    usr_path = input(">> ")

    if(os.path.exists(usr_path)):
        dataset, classes, min_b = cntrt.load_dataset(path=usr_path, img_h=FLAGS.height, img_w=FLAGS.width)

        # Separate composition and execute
        sess = tf.InteractiveSession()

        # Initialize placeholdersshape[0]
        # x is image, y_ is label
        #x = tf.placeholder(tf.float32, shape=[None, img_length])
        #y_ = tf.placeholder(tf.float32, shape=[None, classes])
        height, width, dimension = dataset.train.shape
        x = tf.placeholder(tf.float32, shape=[None, height, width, dimension])
        y_ = tf.placeholder(tf.float32, shape=[None, classes])

        keep_prob, train_step, accuracy = model.conv_neural_network(x, y_, height=height, width=width, dimension=dimension, classes=classes)

        print("\n***** Training with CNN *****")
        sess.run(tf.global_variables_initializer())
        print(" Initialized all variables")

        #Create a saver object which will save all the variables
        saver = tf.train.Saver()

        batch_size = FLAGS.batch
        if(batch_size > min_b):
            batch_size = min_b
        subp.training(steps=FLAGS.steps, batch_size=batch_size, dataset=dataset, sess=sess, keep_prob=keep_prob, train_step=train_step, accuracy=accuracy, x=x, y_=y_, saver=saver)


    else:
        print("Invalid path.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000, help='Default: 1000')
    parser.add_argument('--batch', type=int, default=10, help='Default: 10. Batches per iteration, the number of data to be training and testing.')
    parser.add_argument('--height', type=float, default=1, help='Default: 28. Enter the size to resize images what you want')
    parser.add_argument('--width', type=float, default=1, help='Default: 28. Enter the size to resize images what you want')
    FLAGS, unparsed = parser.parse_known_args()

    main()
