import tensorflow as tf
import numpy as np
import pickle as pk
import random

print(random.uniform(0,1))

def activation_unit(W, unit):
    Act_list = []
    for j in range(W.shape[0]):
        S = W[j].reshape((-1, W[j].shape[3])) # straighten
        act_list = []
        index = 0
        for i in range(0, S.shape[0], unit):
            act_list.append(np.zeros(S.shape))
            if (i+unit) < S.shape[0]:
                en = i + unit
            else:
                en = S.shape[0]
            act_list[index][i:en] = S[i:en]
            index += 1
        Act_list.append(act_list)
    Act_list = np.array(Act_list)
    Act_list = Act_list.reshape(W.shape[0], -1, W.shape[1], W.shape[2], W.shape[3], W.shape[4])
    return np.float32(Act_list)

def insert_error(ideal, err_list):
    for x in np.nditer(ideal, op_flags=['readwrite']):
        for u in range(11): # only consider 10 cells accumulation
            if random.uniform(0,1) < err_list[9][int(x[...])][u]:
                x[...] = u
    return ideal

err_list = pk.load(open('Err_file.p', 'rb'))
print(err_list[9])
x = tf.constant([[[[1,1],[0,1],[1,2]],
                  [[1,2],[0,1],[2,1]],
                  [[0,1],[1,1],[1,1]]]], tf.float32) #NHWC

filt = tf.constant([[[[1,0],[1,1]], [[1,1],[1,0]]],
                    [[[1,0],[1,1]], [[0,1],[1,1]]]], tf.float32) #[filter_height, filter_width, in_channels, out_channels]


Act_unit = 3
org_result = tf.nn.conv2d(x, filt, strides=[1,1,1,1], padding='VALID')
err_result = tf.py_func(insert_error, [org_result, err_list], tf.float32)
#filist = tf.py_func(activation_unit, [filt, Act_unit], tf.float32)
#print(ll)
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
#print('x.shape', sess.run(tf.shape(x)))
#print('filter.shape', sess.run(tf.shape(filt)))
#print('result', sess.run(result))
print('org_result', sess.run(org_result))
print('org_result.shape', sess.run(tf.shape(org_result)))
print('err_result', sess.run(err_result))
#print('filist', sess.run(filist))
#print('filist.shape', sess.run(tf.shape(filist)[0]))
#print('result3', sess.run(result3))
#print('test', sess.run(resulttt))
# Visualize
#writer = tf.summary.FileWriter('TensorBoard/', graph=sess.graph)
sess.close()
