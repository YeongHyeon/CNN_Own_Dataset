from datetime import datetime
import hashlib
import inspect, os
import random
import re
import struct
import sys
import shutil

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.platform import gfile
from tensorflow.python.framework import dtypes
from tensorflow.python.util import compat

import cv2
import matplotlib.image as mpimg

FILE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      #if reshape:
        #assert images.shape[3] == 1
        #images = images.reshape(images.shape[0],
        #                        images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """Builds a list of training images from the file system.
    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.
    Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.
    Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
    """
    print("\n***** Create image lists *****")

    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        print(" Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            print(' No files found')
            continue
        if len(file_list) < 20:
            print(' WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            print(' WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # We want to ignore anything after '_nohash_' in the file name when
            # deciding which set to put an image in, the data set creator has a way of
            # grouping photos that are close variations of each other. For example
            # this is used in the plant disease data set to group multiple pictures of
            # the same leaf.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                              (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                             (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result

def key_from_dictionary(dictionary):
    print("\n***** Extract keys *****")
    master_key = list(dictionary.keys())
    sub_key = list(dictionary[master_key[0]].keys())

    print(" Master Key is...")
    print(" "+str(master_key))
    print(" Sub Key is...")
    print(" "+str(sub_key))

    return master_key, sub_key

def image_save(path, imagename, matrix):
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(path+imagename, matrix)

def imagelist_to_dataset(image_dir, image_lists, imsize=28, rgb=True):

    master_key, sub_key = key_from_dictionary(image_lists)
    classes = len(master_key)


    print("\n***** Make image list *****")
    result_dir = "dataset/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        shutil.rmtree(result_dir)
        os.makedirs(result_dir)

    x_train = np.empty((0, imsize * imsize * 3), int)
    t_train = np.empty((0, classes), int)
    x_test = np.empty((0, imsize * imsize * 3), int)
    t_test = np.empty((0, classes), int)
    x_valid = np.empty((0, imsize * imsize * 3), int)
    t_valid = np.empty((0, classes), int)

    for key_i in [0, 1, 3]:
        if key_i == 0:
            result_name = "train"
        elif key_i == 1:
            result_name = "test"
        else:
            result_name = "valid"
        sys.stdout.write(" Make \'"+result_name+" list\'...")

        # m: class
        for m in master_key:

                for i in range(len(image_lists[m][sub_key[key_i]])):
                    # m: category
                    # image_lists[m][sub_key[key_i]][i]: image name
                    image_path = image_dir+"/"+m+"/"+image_lists[m][sub_key[key_i]][i]
                    # Read jpg images and resizing it.
                    origin_image = cv2.imread(image_path)
                    resized_image = cv2.resize(origin_image, (imsize, imsize))
                    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

                    image_save(result_dir+"origin/"+result_name+"/", image_lists[m][sub_key[key_i]][i], origin_image)
                    image_save(result_dir+"resize/"+result_name+"/", image_lists[m][sub_key[key_i]][i], resized_image)
                    image_save(result_dir+"gray/"+result_name+"/", image_lists[m][sub_key[key_i]][i], grayscale_image)

                    img_length = 1
                    for idx in range(len(resized_image.shape)):
                        img_length *= resized_image.shape[idx]

                    if(rgb):
                        if key_i == 0:
                            x_train = np.append(x_train, resized_image.reshape(1, img_length), axis=0)
                            t_train = np.append(t_train, np.eye(classes)[int(np.asfarray(m))].reshape(1, classes), axis=0)
                        elif key_i == 1:
                            x_test = np.append(x_test, resized_image.reshape(1, img_length), axis=0)
                            t_test = np.append(t_test, np.eye(classes)[int(np.asfarray(m))].reshape(1, classes), axis=0)
                        else:
                            x_valid = np.append(x_valid, resized_image.reshape(1, img_length), axis=0)
                            t_valid = np.append(t_valid, np.eye(classes)[int(np.asfarray(m))].reshape(1, classes), axis=0)
                    else:
                        if key_i == 0:
                            x_train = np.append(x_train, grayscale_image.reshape(1, img_length), axis=0)
                            t_train = np.append(t_train, np.eye(classes)[int(np.asfarray(m))].reshape(1, classes), axis=0)
                        elif key_i == 1:
                            x_test = np.append(x_test, grayscale_image.reshape(1, img_length), axis=0)
                            t_test = np.append(t_test, np.eye(classes)[int(np.asfarray(m))].reshape(1, classes), axis=0)
                        else:
                            x_valid = np.append(x_valid, grayscale_image.reshape(1, img_length), axis=0)
                            t_valid = np.append(t_valid, np.eye(classes)[int(np.asfarray(m))].reshape(1, classes), axis=0)
        print(" complete.")
    x_train = np.asarray(x_train)
    t_train = np.asarray(t_train)
    x_test = np.asarray(x_test)
    t_test = np.asarray(t_test)
    return (x_train, t_train), (x_test, t_test), classes

def imagelist_to_tensor(image_dir, image_lists, imsize=28):
    X_data = []
    files = glob.glob ("*.jpg")
    for myFile in files:
        image = cv2.imread (myFile)
        X_data.append (image)

    print('X_data shape:', np.array(X_data).shape)

def load_dataset(image_dir="/images", test_percentage=10, validation_percentage=10, imsize=28, reshape=True, rgb=True):
    dtype=dtypes.float32

    test_percentage = 10
    validation_percentage = 10
    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(image_dir, test_percentage, validation_percentage)
    (x_train, t_train), (x_test, t_test), classes = imagelist_to_dataset(image_dir=image_dir, image_lists=image_lists, imsize=imsize, rgb=True)
    print("\n Data set is ready!")
    print(" Data for train : " + str(x_train.shape[0]))
    print(" Data for test  : " + str(x_test.shape[0]))

    #return (x_train, t_train), (x_test, t_test)
    train = DataSet(x_train, t_train, dtype=dtype, reshape=reshape)
    test = DataSet(x_test, t_test, dtype=dtype, reshape=reshape)
    validation = DataSet(x_test, t_test, dtype=dtype, reshape=reshape)

    return base.Datasets(train=train, test=test, validation=validation), classes

#=========================================
#              *** main ***
#=========================================
if __name__ == "__main__":
    load_dataset(image_dir="./images", test_percentage=10, validation_percentage=10, imsize=28, reshape=True)
