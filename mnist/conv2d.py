import tensorflow as tf
import numpy as np
import error_insertion as ei


x = tf.constant([[[[-1],[-1],[-1]],
                  [[-1],[-1],[-1]],
                  [[-1],[-1],[-1]]]], tf.float32) #NHWC #shape=(1,3,3,1)
filt = tf.constant([[[[1,0]], [[1,1]]],
                    [[[1,0]], [[0,1]]]], tf.float32) #[filter_height, filter_width, in_channels, out_channels] #shape=(2,2,1,2)
Act_unit = 3
IL = 2
FL = 1
WL = IL + FL

org_result = tf.nn.conv2d(x, filt, strides=[1,1,1,1], padding='SAME')
x_bit = tf.py_func(ei.decomposition, [x, IL, FL, WL], tf.float32)
filt_bit = tf.py_func(ei.decomposition, [filt, IL, FL, WL], tf.float32)
x_act = tf.py_func(ei.new_activation_unit, [x_bit, Act_unit, 0], tf.float32)
#err_result = ei.composition(x_err, filt_err, IL, FL, WL, filt.shape, Act_unit, 'Conv2d')

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

print('x',sess.run(x))
print('x.shape', sess.run(tf.shape(x)))
print('filter', sess.run(tf.shape(filt)))
print('org_result', sess.run(org_result))
print('org_result.shape', sess.run(tf.shape(org_result)))
print('x_bit', sess.run(x_bit))
print('filt_bit', sess.run(filt_bit))
print('x_act', sess.run(x_act))

#print('x_err', sess.run(x_err))
#print('filt_err', sess.run(filt_err))
#print('err_result', sess.run(err_result))
#print('err_result.shape', sess.run(tf.shape(err_result)))
#print('filist', sess.run(filist))
#print('filist.shape', sess.run(tf.shape(filist)[0]))
#print('result3', sess.run(result3))
#print('test', sess.run(resulttt))
# Visualize
#writer = tf.summary.FileWriter('TensorBoard/', graph=sess.graph)
sess.close()
