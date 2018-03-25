from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

def error(x):
   return np.sinh(x) 

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Create the model
x = tf.placeholder(tf.float32, shape=[None, 784])
W1 = tf.Variable(tf.random_normal([784, 100], stddev=0.1))
b1 = tf.Variable(tf.zeros([100]))
W2 = tf.Variable(tf.random_normal([100, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W1)
y = tf.py_func(error, [y], tf.float32) # calculate error
y = y + b1
y = tf.nn.relu(y)

y = tf.matmul(y, W2)
y = tf.py_func(error, [y], tf.float32) # calcuate error
y = y + b2

y_ = tf.placeholder(tf.float32, shape=[None, 10])

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Restore Model
saver = tf.train.Saver()
saver.restore(sess, "./model/model.ckpt")

# Testing
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("accuracy: ",accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels}))
