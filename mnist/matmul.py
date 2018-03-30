import tensorflow as tf 
import numpy as np
import error_insertion as ei
import pickle as pk

IL = 3
FL = 0
WL = IL+FL
unit = 2
error_list = pk.load(open('Error_file/perfect.p', 'rb'))

#x = tf.placeholder(tf.float32, shape=[None, 2], name='x')
x = tf.constant([[1.,3.,3.]])
w = tf.Variable([[1.],[3.],[3.]])
y = tf.matmul(x, w)
y_err = ei.crossbar(x, w, IL, FL, WL, unit, error_list, 1)
# initialize Graph
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#feed = np.array([[2.5,1.25],[1.5,-2.25]])
#feed = np.ones((2,3))
#feed *= -1

print('x:', sess.run(x))
print('w:', sess.run(w))
print('y:', sess.run(y))
print('y_err:', sess.run(y_err))
