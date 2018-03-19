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
Act_unit = 10
batch_size = 1
testing_img_number = 100

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create Model
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
x_image = tf.reshape(x, [-1, 28, 28, 1])
# ==Convolution layer== #    
with tf.name_scope('Conv1'):
    with tf.name_scope('Weights'):
        W_conv1	= tf.Variable(tf.truncated_normal([3,3,1,32], stddev=0.1)) # 3x3, input_size=1, output_size=32  # 5x5
    with tf.name_scope('Biases'):
        b_conv1 = tf.Variable(tf.zeros([32]))     
    with tf.name_scope('Convolution'):
        x_image = tf.py_func(ei.decomposition, [x_image, IL, FL, WL], tf.float32)
        W_conv1_list = tf.py_func(ei.decomposition, [W_conv1, IL, FL, WL], tf.float32) 
        W_conv1_list = tf.py_func(ei.activation_unit, [W_conv1_list, Act_unit, 0], tf.float32)   
        h_conv1 = ei.composition(x_image, W_conv1_list, IL, FL, WL, W_conv1.shape, Act_unit, computeType='Conv2d')
        #h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME')
        h_conv1 += b_conv1
    with tf.name_scope('Relu'):
        h_conv1 = tf.nn.relu(h_conv1) # 28x28x32
with tf.name_scope('Maxpooling'):
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 14x14x32

with tf.name_scope('Conv2'):
    with tf.name_scope('Weights'):
        W_conv2 = tf.Variable(tf.truncated_normal([3,3,32,64], stddev=0.1)) # 5x5
    with tf.name_scope('Biases'):
        b_conv2 = tf.Variable(tf.zeros(([64])))
    with tf.name_scope('Convolution'):
        h_pool1 = tf.py_func(ei.decomposition, [h_pool1, IL, FL, WL], tf.float32) 
        W_conv2_list = tf.py_func(ei.decomposition, [W_conv2, IL, FL, WL], tf.float32) 
        W_conv2_list = tf.py_func(ei.activation_unit, [W_conv2_list, Act_unit, 0], tf.float32)
        h_conv2 = ei.composition(h_pool1, W_conv2_list, IL, FL, WL, W_conv2.shape, Act_unit, computeType='Conv2d')
        #h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME')
        h_conv2 += b_conv2
    with tf.name_scope('Relu'): 
        h_conv2 = tf.nn.relu(h_conv2) # 14x14x64
with tf.name_scope('Maxpooling'):
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 7x7x64
# ==Fully connected layer== #
with tf.name_scope('Dense1'):
    with tf.name_scope('Weights'):
        W_fcon1 = tf.Variable(tf.truncated_normal([7*7*64, 128], stddev=0.1)) # 1024
    with tf.name_scope('Biases'):
        b_fcon1 = tf.Variable(tf.zeros([128])) # 1024
    with tf.name_scope('Flatten'):
        flatten = tf.reshape(h_pool2, [-1,7*7*64])
    with tf.name_scope('Formula'):
        flatten = tf.py_func(ei.decomposition, [flatten, IL, FL, WL], tf.float32)
        W_fcon1_list = tf.py_func(ei.decomposition, [W_fcon1, IL, FL, WL], tf.float32)
        W_fcon1_list = tf.py_func(ei.activation_unit, [W_fcon1_list, Act_unit, 1], tf.float32)
        h_fcon1 = ei.composition(flatten, W_fcon1_list, IL, FL, WL, W_fcon1.shape, Act_unit, computeType='Matmul')
        h_fcon1 += b_fcon1
        #h_fcon1 = tf.matmul(flatten, W_fcon1) + b_fcon1
    with tf.name_scope('Relu'):
        h_fcon1 = tf.nn.relu(h_fcon1)
    with tf.name_scope('Dropout'):
        h_drop1 = tf.nn.dropout(h_fcon1, 0.5)

with tf.name_scope('Dense2'):
    with tf.name_scope('Weights'):
        W_fcon2 = tf.Variable(tf.zeros([128, 10]))
    with tf.name_scope('Biases'):
        b_fcon2 = tf.Variable(tf.zeros([10]))
    with tf.name_scope('Formula'):
        h_drop1 = tf.py_func(ei.decomposition, [h_drop1, IL, FL, WL], tf.float32)
        W_fcon2_list = tf.py_func(ei.decomposition, [W_fcon2, IL, FL, WL], tf.float32)
        W_fcon2_list = tf.py_func(ei.activation_unit, [W_fcon2_list, Act_unit, 1], tf.float32)
        h_fcon2 = ei.composition(h_drop1, W_fcon2_list, IL, FL, WL, W_fcon2.shape, Act_unit, computeType='Matmul')
        h_fcon2 += b_fcon2
        #h_fcon2 = tf.matmul(h_drop1, W_fcon2) + b_fcon2
        prediction = h_fcon2

prediction = tf.identity(prediction, name='prediction')

# initialize Graph
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Restore Model
saver = tf.train.Saver()
saver.restore(sess, "./model/cnn_model.ckpt")

# Testing
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Visualize
writer = tf.summary.FileWriter('TensorBoard/', graph=sess.graph)

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
