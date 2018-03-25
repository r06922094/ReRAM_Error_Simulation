import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Testing data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Restore Model
sess = tf.InteractiveSession()

saver = tf.train.import_meta_graph('./model/cnn_model.ckpt.meta')
#saver.restore(sess, './model/cnn_model.ckpt')
saver.restore(sess, tf.train.latest_checkpoint( './model/'))


graph = tf.get_default_graph()

#print(sess.run(graph.get_tensor_by_name('b:0')))
#print(sess.run('w:0'))

x = graph.get_tensor_by_name('x:0')
#y_ = graph.get_tensor_by_name('y_:0')
prediction = graph.get_tensor_by_name('prediction:0')


# Testing
pre = sess.run(prediction, feed_dict={x:mnist.test.images})
y_ = tf.placeholder(tf.float32, shape=[None, 10])
correct_prediction = tf.equal(tf.argmax(pre,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(pre)
print("accuracy: ",sess.run(accuracy, feed_dict={y_:mnist.test.labels}))

# Visualize
writer = tf.summary.FileWriter('TensorBoard/', graph=sess.graph)

sess.close()

