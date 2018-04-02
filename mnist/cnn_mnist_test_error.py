from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import pickle as pk
import time
import random
import error_insertion as ei
import sys
from math import *
from tqdm import tqdm
start_time = time.time()

# Parameter
IL = 2
FL = 4
WL = IL + FL
unit = int(sys.argv[1])
batch_size = 1
testing_img_number = 100
error_list = pk.load(open('Error_file/Err_file_mean_3_var_0.02_0.38_SA_4.pkl', 'rb'))
#error_list = pk.load(open('Error_file/perfect.p', 'rb'))

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Create Model
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
x_image = tf.reshape(x, [-1, 28, 28, 1])
# ==Convolution layer== #
with tf.name_scope('Conv1'):
    W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,16], stddev=0.1), name='weight') # 5x5,in=1,out=16
    b_conv1 = tf.Variable(tf.zeros([16]), name='bias')
    #h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME', name='convolution') + b_conv1
    h_conv1 = ei.crossbar(x_image, W_conv1, IL, FL, WL, unit, error_list, 0) + b_conv1
    h_conv1 = tf.nn.relu(h_conv1, name='relu') # 28x28x16
    layer1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pooling') # 14x14x16
# ==Fully connected layer== #
with tf.name_scope('Dense1'):
    flatten = tf.reshape(layer1, [-1,14*14*16], name='flatten')
    W_fcon1 = tf.Variable(tf.truncated_normal([14*14*16, 100], stddev=0.1), name='weight')
    b_fcon1 = tf.Variable(tf.zeros([100]), name='bias')
    #h_fcon1 = tf.matmul(flatten, W_fcon1, name='matmul') + b_fcon1
    h_fcon1 = ei.crossbar(flatten, W_fcon1, IL, FL, WL, unit, error_list, 1) + b_fcon1
    layer2 = tf.nn.relu(h_fcon1, name='relu')
with tf.name_scope('Dense2'):
    W_fcon2 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1), name='weight')
    b_fcon2 = tf.Variable(tf.zeros([10]), name='bias')
    #h_fcon2 = tf.matmul(layer2, W_fcon2, name='matmul') + b_fcon2
    h_fcon2 = ei.crossbar(layer2, W_fcon2, IL, FL, WL, unit, error_list, 1) + b_fcon2
    layer3 = tf.nn.relu(h_fcon2, name='relu')
prediction = tf.identity(layer3, name='prediction')

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

test_accuracy = 0
test_img = mnist.test.images[0:testing_img_number]
test_label = mnist.test.labels[0:testing_img_number]
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
