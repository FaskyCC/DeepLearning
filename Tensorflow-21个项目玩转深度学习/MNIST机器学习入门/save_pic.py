#coding: utf-8
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc as misc
import os
import numpy as np

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

save_dir = './mnist_pic/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

for i in range(10):
    image_array = mnist.train.images[i, :]

    image_array = image_array.reshape(28, 28)

    filename = save_dir+'mnist_train_{0}.jpg'.format(i)

    misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)

    one_hot_label = mnist.train.labels[i, :]

    label = np.argmax(one_hot_label)

    with open('./mnist_pic/label.txt', 'a') as f:
        f.write('mnist_train_{0}.jpg label: {1} \n'.format(i, label))
