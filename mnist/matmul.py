import tensorflow as tf 
import numpy as np
import error_insertion as ei
import pickle as pk

IL = 3
FL = 2
WL = IL+FL
unit = 1
error_list = pk.load(open('perfect.p', 'rb'))

#x = tf.placeholder(tf.float32, shape=[None, 2], name='x')
x = tf.constant([[2.5,1.25],[1.5,-2.25]])
w = tf.Variable([[1.,1.],[1., 1.]])
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
