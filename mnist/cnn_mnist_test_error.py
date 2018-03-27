from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import pickle as pk
import time
import random
import error_insertion as ei
from math import *
from tqdm import tqdm

start_time = time.time()
IL = 2
FL = 4
WL = IL + FL
unit = 2
batch_size = 1
testing_img_number = 1
error_list = pk.load(open('Error_file/Err_file_3.3.pkl', 'rb'))
#error_list = pk.load(open('Error_file/996.p', 'rb'))

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create Model
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
x_image = tf.reshape(x, [-1, 28, 28, 1])
# ==Convolution layer== #
with tf.name_scope('Conv1'):
    W_conv1	= tf.Variable(tf.truncated_normal([3,3,1,16], stddev=0.1), name='weight') # 3x3,in=1,out=16
    b_conv1 = tf.Variable(tf.zeros([16]), name='bias')
    #h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME', name='convolution') + b_conv1
    h_conv1 = ei.crossbar(x_image, W_conv1, IL, FL, WL, unit, error_list, 0) + b_conv1
    h_conv1 = tf.nn.relu(h_conv1, name='relu') # 28x28x16
    layer1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pooling') # 14x14x16
with tf.name_scope('Conv2'):
    W_conv2	= tf.Variable(tf.truncated_normal([3,3,16,32], stddev=0.1), name='weight') # 3x3, in=16, output_size=32
    b_conv2 = tf.Variable(tf.zeros([32]), name='bias')
    #h_conv2 = tf.nn.conv2d(layer1, W_conv2, strides=[1,1,1,1], padding='SAME', name='convolution') + b_conv2
    h_conv2 = ei.crossbar(layer1, W_conv2, IL, FL, WL, unit, error_list, 0) + b_conv2
    h_conv2 = tf.nn.relu(h_conv2, name='relu') # 28x28x32
    layer2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pooling') # 7x7x32
# ==Fully connected layer== #
with tf.name_scope('Dense1'):
    flatten = tf.reshape(layer2, [-1,7*7*32], name='flatten')
    W_fcon1 = tf.Variable(tf.truncated_normal([7*7*32, 128], stddev=0.1), name='weight')
    b_fcon1 = tf.Variable(tf.zeros([128]), name='bias')
    #h_fcon1 = tf.matmul(flatten, W_fcon1, name='formula') + b_fcon1
    h_fcon1 = ei.crossbar(flatten, W_fcon1, IL, FL, WL, unit, error_list, 1) + b_fcon1
    h_fcon1 = tf.nn.relu(h_fcon1, name='relu')
    layer3 = tf.nn.dropout(h_fcon1, 0.5, name='dropout')
with tf.name_scope('Dense2'):
    W_fcon2 = tf.Variable(tf.truncated_normal([128, 10], stddev=0.1), name='weight')
    b_fcon2 = tf.Variable(tf.zeros([10]), name='bias')
    #h_fcon2 = tf.matmul(layer3, W_fcon2, name='formula') + b_fcon2
    h_fcon2 = ei.crossbar(layer3, W_fcon2, 3, 13, 16, unit, error_list, 1) + b_fcon2
    layer4 = tf.nn.relu(h_fcon2, name='relu')
prediction = tf.identity(layer4, name='prediction')

# initialize Graph
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Restore Model
saver = tf.train.Saver()
saver.restore(sess, "./model/baseline.ckpt")

# Testing
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Visualize
#writer = tf.summary.FileWriter('TensorBoard/', graph=sess.graph)

batch_num = int(mnist.test.num_examples / batch_size)
test_accuracy = 0
test_img = mnist.test.images
test_label = mnist.test.labels
test_img = test_img[0:testing_img_number]
test_label = test_label[0:testing_img_number]
batch_num = int(testing_img_number/batch_size)

for i in tqdm(range(batch_num)):
    batch_img = test_img[i*batch_size:i*batch_size+batch_size]
    batch_label = test_label[i*batch_size:i*batch_size+batch_size]
    test_accuracy += accuracy.eval(feed_dict={x: batch_img, y_: batch_label})

test_accuracy /= batch_num

print('IL:', IL)
print('FL:', FL)
print('batch size:', batch_size)
print('# of testing data:', testing_img_number)
print('accuracy: %g'%test_accuracy)
print('execution time: %ss' % (time.time() - start_time))
