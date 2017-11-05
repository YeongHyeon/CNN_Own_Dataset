# import modules
import os
import numpy as np
import matplotlib.pyplot as plt
import psutil, shutil
import scipy.misc

# from modules
from datetime import datetime

def directory_check(dir_name):
    if(os.path.exists(dir_name)):
        shutil.rmtree(dir_name)
        os.mkdir(dir_name)
    else:
        os.mkdir(dir_name)

def copy_file_as_image(origin, copy):
    count = 0
    for ori in origin:
        shutil.copy(ori, copy+"/"+str(count)+".jpg")
        count = count + 1

def memory_check():
    pid = os.getpid()
    proc = psutil.Process(pid)
    used_mem = proc.memory_info()[0]

    print(" Memory Used: %.2f GB\t( %.2f MB )" %(used_mem/(2**30), used_mem/(2**20)))

    return used_mem

def show_graph(train_acc_list, test_acc_list):
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

def save_graph_as_image(train_list, test_list, ylabel=""):

    x = np.arange(len(train_list))
    plt.clf()
    plt.plot(x, train_list, label="train "+ylabel)
    plt.plot(x, test_list, label="test "+ylabel, linestyle='--')
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.ylim(-0.1, max([1, max(train_list), max(test_list)])*1.1)
    plt.legend(loc='lower right')

    if(not(os.path.exists("./graph"))):
        os.mkdir("./graph")
    else:
        pass
    now = datetime.now()

    print(" Save "+ylabel+" graph in ./graph")
    plt.savefig("./graph/"+ylabel+"_"+now.strftime('%Y%m%d_%H%M%S%f')+".png")

def save_matrix_as_image(filename, matrix):
    matrix = matrix.real

    min_idx = matrix.argmin()
    bias = matrix[int(min_idx/matrix.shape[0])][int(min_idx%matrix.shape[0])]
    matrix = matrix + abs(bias)

    scipy.misc.imsave(filename+".jpg", matrix)
