import tensorflow as tf
import numpy as np
import pickle as pk
import random
import error_insertion as ei

'''
err_list = pk.load(open('Err_file.p', 'rb'))
'''

#x = tf.constant([[[[1,1],[0,1],[1,2]],
#                  [[1,2],[0,1],[2,1]],
#                  [[0,1],[1,1],[1,1]]]], tf.float32) #NHWC #shape=(1,3,3,2)
x = tf.constant([[[[1],[0],[1]],
                  [[1],[0],[2]],
                  [[0],[1],[1]]]], tf.float32) #NHWC #shape=(1,3,3,1)
filt = tf.constant([[[[1,0]], [[1,1]]],
                    [[[1,0]], [[0,1]]]], tf.float32) #[filter_height, filter_width, in_channels, out_channels] #shape=(2,2,1,2)
Act_unit = 3
IL = 3
FL = 1
WL = IL + FL

org_result = tf.nn.conv2d(x, filt, strides=[1,1,1,1], padding='SAME')
x_err = tf.py_func(ei.decomposition, [x, IL, FL, WL], tf.float32)
filt_err = tf.py_func(ei.decomposition, [filt, IL, FL, WL], tf.float32)
filt_err = tf.py_func(ei.activation_unit, [filt_err, Act_unit, 0], tf.float32)
err_result = ei.composition(x_err, filt_err, IL, FL, WL, filt.shape, Act_unit, 'Conv2d')

'''
x1 = tf.constant([[[[1],[2],[1]],
                   [[1],[3],[2]],
                   [[2],[1],[1]]]], tf.float32) #NHWC
x2 = tf.constant([[[[1],[1],[2]],
                   [[2],[1],[2]],
                   [[2],[3],[1]]]], tf.float32) #NHWC
filt1 = tf.constant([[[[1]], [[2]]],
                     [[[1]], [[2]]]], tf.float32) #[filter_height, filter_width, in_channels, out_channels]
filt2 = tf.constant([[[[3]], [[1]]],
                     [[[2]], [[1]]]], tf.float32) #[filter_height, filter_width, in_channels, out_channels]

filt3 = tf.constant([[[[3]], [[1]]],
                     [[[2]], [[0]]]], tf.float32) #[filter_height, filter_width, in_channels, out_channels]

'''
#x0 = tf.mod(x, 2)
#x1 = tf.floordiv(x, 2)
#w = tf.Variable(tf.truncated_normal([2,2,1,2], stddev=0.1))

'''
result = tf.nn.conv2d(x, filist[0], strides=[1,1,1,1], padding='VALID')
'''
#result += tf.nn.conv2d(x, filist[1], strides=[1,1,1,1], padding='VALID')
#result += tf.nn.conv2d(x, filist[2], strides=[1,1,1,1], padding='VALID')
'''iterate = filt.shape[0] * filt.shape[1] * filt.shape[2] // Act_unit + 1
for i in range(1, iterate, 1):
    result += tf.nn.conv2d(x, filist[i], strides=[1,1,1,1], padding='VALID')
'''
#result1 = tf.nn.conv2d(x1, filt1, strides=[1,1,1,1], padding='VALID')
#result2 = tf.nn.conv2d(x2, filt2, strides=[1,1,1,1], padding='VALID')
#result3 = tf.add(result1, result2)
#resulttt= tf.nn.conv2d(x, filt3, strides=[1,1,1,1], padding='VALID')
#result0 = tf.nn.conv2d(x0, filt, strides=[1,1,1,1], padding='VALID')
#result1 = tf.nn.conv2d(x1, filt, strides=[1,1,1,1], padding='VALID')
#result1 = tf.multiply(result1, 2)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

#print('x0',sess.run(x0))
#print('x1',sess.run(x1))
#print('x', sess.run(tf.shape(x)))
#print('filter', sess.run(tf.shape(filt)))
#print('result', sess.run(result))
print('org_result', sess.run(org_result))
print('org_result.shape', sess.run(tf.shape(org_result)))
#print('x_err', sess.run(x_err))
#print('filt_err', sess.run(filt_err))
print('err_result', sess.run(err_result))
print('err_result.shape', sess.run(tf.shape(err_result)))
#print('filist', sess.run(filist))
#print('filist.shape', sess.run(tf.shape(filist)[0]))
#print('result3', sess.run(result3))
#print('test', sess.run(resulttt))
# Visualize
#writer = tf.summary.FileWriter('TensorBoard/', graph=sess.graph)
sess.close()
