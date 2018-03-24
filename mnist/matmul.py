import tensorflow as tf 
import numpy as np
import error_insertion as ei

IL = 2
FL = 1
WL = 3
unit = 2

x = tf.placeholder(tf.float32, shape=[None, 3], name='x')
w = tf.Variable(tf.ones([3, 2]))
y = tf.matmul(x, w)
 
# decompose
x_decomposed = tf.py_func(ei.decomposition, [x, IL, FL, WL], tf.float32)
w_decomposed = tf.py_func(ei.decomposition, [w, IL, FL, WL], tf.float32)
# act x
x_decomposed_act = tf.py_func(ei.new_activation_unit, [x_decomposed, unit, 1], tf.float32)
# initialize Graph
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

feed = np.ones((2,3))
feed *= -1

print('x:', sess.run(x, feed_dict={x: feed}))
print('w:', sess.run(w))
print('y:', sess.run(y, feed_dict={x: feed}))

print('x_decomposed:', sess.run(x_decomposed, feed_dict={x: feed}))
print('w_decomposed:', sess.run(x_decomposed, feed_dict={x: feed}))
print('x_decomposed_act', sess.run(x_decomposed_act, feed_dict={x: feed}))
print('x_decomposed_act.shape', sess.run(tf.shape(x_decomposed_act), feed_dict={x: feed}))
