import tensorflow as tf
import numpy as np
import error_insertion as ei
import pickle as pk

unit = 2 # 10
IL = 2
FL = 0
WL = IL + FL
error_list = pk.load(open('Err_file.p', 'rb')) # unit = 10 100%

x = tf.constant([[[[1],[1],[1]],
                  [[1],[1],[1]],
                  [[1],[1],[1]]]], tf.float32) #NHWC #shape=(1,3,3,1)
filt = tf.constant([[[[1,1]], [[1,1]]],
                    [[[1,1]], [[1,1]]]], tf.float32) #[filter_height, filter_width, in_channels, out_channels] #shape=(2,2,1,2)
y = tf.nn.conv2d(x, filt, strides=[1,1,1,1], padding='SAME')
y_err = ei.crossbar(x, filt, IL, FL, WL, unit, error_list, 0)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

print('x',sess.run(x))
print('x.shape', sess.run(tf.shape(x)))
print('filter', sess.run(tf.shape(filt)))
print('y', sess.run(y))
print('y.shape', sess.run(tf.shape(y)))
print('y_err', sess.run(y_err))

# Visualize
#writer = tf.summary.FileWriter('TensorBoard/', graph=sess.graph)
sess.close()
