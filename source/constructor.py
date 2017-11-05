import tensorflow as tf
import numpy as np
import os, sys, glob, shutil, random, inspect
import scipy.misc

import source.utility as util

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

class DataSet(object):

    def __init__(self, who_am_i, datas, labels, class_len, height=0, width=0):
        self._who_am_i = who_am_i

        combined = list(zip(datas, labels))
        random.shuffle(combined)
        self._datas, self._labels = zip(*combined)

        self._class_len = class_len

        form = scipy.misc.imread(self._datas[0])
        dimension = 0
        if(len(form.shape) < 3):
            dimension = 1
        else:
            dimension = form.shape[2]


        if((height == 0) or (width == 0)):
            self._height = form.shape[0]
            self._width = form.shape[1]
        else:
            self._height = height
            self._width = width
        self._dimension = dimension

    @property
    def amount(self):
        return int(len(self._datas))

    @property
    def shape(self):
        return self._height, self._width, self._dimension

    def next_batch(self, batch_size=10):

        datas = np.empty((0, self._height, self._width, self._dimension), int)
        labels = np.empty((0, self._class_len), int)


        for idx in range(batch_size):
            random.randint(0, len(self._datas)-1)
            tmp_img = scipy.misc.imread(self._datas[idx])
            tmp_img = scipy.misc.imresize(tmp_img, (self._height, self._width))
            tmp_img = tmp_img.reshape(1, self._height, self._width, self._dimension)

            datas = np.append(datas, tmp_img, axis=0)
            labels = np.append(labels, np.eye(self._class_len)[int(np.asfarray(self._labels[idx]))].reshape(1, self._class_len), axis=0)


        return datas, labels

def split_data(path=None):

    util.directory_check(PACK_PATH+"/train")
    util.directory_check(PACK_PATH+"/test")
    util.directory_check(PACK_PATH+"/valid")

    directories = []
    for dirname in os.listdir(path):
        directories.append(dirname)

    for di in directories:
        if(not(os.path.exists(PACK_PATH+"/train/"+di))):
            os.mkdir("./train/"+di)
        if(not(os.path.exists(PACK_PATH+"/test/"+di))):
            os.mkdir(PACK_PATH+"/test/"+di)
        if(not(os.path.exists(PACK_PATH+"/valid/"+di))):
            os.mkdir(PACK_PATH+"/valid/"+di)

    extensions = [".jpg",".JPG",".jpeg",".JPEG"]
    for di in directories:
        files = []
        for ex in extensions:
            for filename in glob.glob(path+"/"+di+"/*"+ex):
                files.append(filename)
        random.shuffle(files)

        tr_point = int(len(files)*0.8)
        te_point = int(len(files)*0.9)
        va_point = int(len(files)*1.0)

        train = files[:tr_point]
        test = files[tr_point:te_point]
        valid = files[te_point:va_point]

        util.copy_file_as_image(train, PACK_PATH+"/train/"+di)
        util.copy_file_as_image(test, PACK_PATH+"/test/"+di)
        util.copy_file_as_image(valid, PACK_PATH+"/valid/"+di)


def path_to_dirlist(path=None):

    directories = []
    for dirname in os.listdir(path):
        directories.append(dirname)

    return directories

def dirlist_to_dataset(path=None, dirlist=None):

    extensions = [".jpg",".JPG",".jpeg",".JPEG"]

    height = 0
    width = 0
    dimension = 0
    classes = len(dirlist)

    for di in dirlist:
        for ex in extensions:
            for fi in glob.glob(path+"/"+di+"/*"+ex):
                sample = scipy.misc.imread(fi)
                if(len(sample.shape) < 3):
                    dimension = 1
                else:
                    dimension = sample.shape[2]
                height = sample.shape[0]
                width = sample.shape[1]
                break

    data_list = []
    label_list = []

    label_num = 0
    for di in dirlist:
        for ex in extensions:
            for fi in glob.glob(path+"/"+di+"/*"+ex):
                data_list.append(fi)
                label_list.append(label_num)
        label_num = label_num + 1

    return data_list, label_list, classes

def load_dataset(path=None, img_h=28, img_w=28):

    print("\n***** Load dataset *****")

    split_data(path=path)

    dirlist = path_to_dirlist(path=PACK_PATH+"/train")
    if(len(dirlist) > 0):
        train_datas, train_labels, classes = dirlist_to_dataset(path="./train", dirlist=dirlist)

    dirlist = path_to_dirlist(path=PACK_PATH+"/test")
    if(len(dirlist) > 0):
        test_datas, test_labels, classes = dirlist_to_dataset(path="./test", dirlist=dirlist)

    dirlist = path_to_dirlist(path=PACK_PATH+"/valid")
    if(len(dirlist) > 0):
        valid_datas, valid_labels, classes = dirlist_to_dataset(path="./valid", dirlist=dirlist)

    train = DataSet(who_am_i="train", datas=train_datas, labels=train_labels, class_len=classes, height=img_h, width=img_w)
    test = DataSet(who_am_i="test", datas=test_datas, labels=test_labels, class_len=classes, height=img_h, width=img_w)
    validation = DataSet(who_am_i="valid", datas=valid_datas, labels=valid_labels, class_len=classes, height=img_h, width=img_w)

    num_train = train.amount
    num_test = test.amount
    print(" Num of Train images : "+str(num_train))
    print(" Num of Test images  : "+str(num_test))

    return base.Datasets(train=train, test=test, validation=validation), classes, min(num_train, num_test)
