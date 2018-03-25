import tensorflow as tf 
import numpy as np
import error_insertion as ei
import pickle as pk

IL = 2
FL = 0
WL = 2
unit = 2 # 10
error_list = pk.load(open('Err_file.p', 'rb')) # unit=10 100%

x = tf.placeholder(tf.float32, shape=[None, 3], name='x')
w = tf.Variable(tf.ones([3, 2]))
y = tf.matmul(x, w)
y_err = ei.crossbar(x, w, IL, FL, WL, unit, error_list, 1)
# initialize Graph
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

feed = np.ones((2,3))
feed *= -1

print('x:', sess.run(x, feed_dict={x: feed}))
print('w:', sess.run(w))
print('y:', sess.run(y, feed_dict={x: feed}))
print('y_err:', sess.run(y_err, feed_dict={x :feed}))
