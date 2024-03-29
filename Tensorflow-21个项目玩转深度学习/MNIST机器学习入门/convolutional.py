# coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev =0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding ='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding ='SAME')

#第一层卷积层
W_conv1 = weight_variable([5,5,1,32])# 5,5表示卷积核大小,1是in_size,32是out_size 28*28
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1) # 28*28*32
h_pool1 = max_pool_2x2(h_conv1) # 14*14*32

#第二层卷积层
W_conv2 =weight_variable([5,5,32,64])# 5,5表示卷积核大小,32是in_size,64是out_size
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)# 14*14*32
h_pool2 = max_pool_2x2(h_conv2) #7*7*64

#全连接层1
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#全连接层2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2)+b_fc2

y_conv_softmax = tf.nn.softmax(y_conv)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv_softmax), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#定义测试准确率
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch = mnist.train.next_batch(100)
        if i % 10 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0
            })
            print("step{0},training accuracy {1}".format(i, train_accuracy))
        train_step.run(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 0.5
        })
    print('training is done')
    test_accuracy = sess.run(accuracy, feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
    })
    print("testining accuracy {0}".format(test_accuracy))
    # test_accuracy = sess.run(accuracy, feed_dict={
    #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
    # })
    # print('test accuracy:{0}'.format(test_accuracy))
