import error_insertion as ei
import tensorflow as tf

IL = 3
FL = 1
WL = 4
Act_unit = 2
a = tf.constant([1,2,3], shape=[1,3])
b = tf.constant([1,2,3,3,2,1], shape=[3,2])


a_l = tf.py_func(ei.decomposition, [a, IL, FL, WL], tf.float32)
b_l = tf.py_func(ei.decomposition, [b, IL, FL, WL], tf.float32)
b_l = tf.py_func(ei.activation_unit, [b_l, Act_unit,'Matmaul'], tf.float32)
c = tf.matmul(a, b)
c_err = ei.composition(a_l, b_l, IL, FL, WL, b.shape, Act_unit, computeType='Matmul')   

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

print('a', sess.run(a))
print('b', sess.run(b))
print('c', sess.run(c))
#print('a_l', sess.run(a_l))
#print('b_l', sess.run(b_l))
#print('a_l.shape', sess.run(tf.shape(a_l)))
#print('b_l.shape', sess.run(tf.shape(b_l)))
print('c_err', sess.run(c_err))
