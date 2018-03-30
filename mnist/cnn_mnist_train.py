from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 55,000 training data
# 10,000 testing data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
iteration = 200000

# Create Model
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
x_image = tf.reshape(x, [-1, 28, 28, 1])

# ==Convolution layer== #
with tf.name_scope('Conv1'):
    W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,16], stddev=0.1), name='weight') # 5x5,in=1,out=16
    b_conv1 = tf.Variable(tf.zeros([16]), name='bias')
    h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME', name='convolution') + b_conv1
    h_conv1 = tf.nn.relu(h_conv1, name='relu') # 28x28x16
    layer1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pooling') # 14x14x16
# ==Fully connected layer== #
flatten = tf.reshape(layer1, [-1,14*14*16], name='flatten')
with tf.name_scope('Dense1'):
    W_fcon1 = tf.Variable(tf.truncated_normal([14*14*16, 100], stddev=0.1), name='weight')
    b_fcon1 = tf.Variable(tf.zeros([100]), name='bias')
    h_fcon1 = tf.matmul(flatten, W_fcon1, name='matmul') + b_fcon1
    layer2 = tf.nn.relu(h_fcon1, name='relu')
    layer2 = tf.nn.dropout(layer2, 0.5, name='dropout')
with tf.name_scope('Dense2'):
    W_fcon2 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1), name='weight')
    b_fcon2 = tf.Variable(tf.zeros([10]), name='bias')
    h_fcon2 = tf.matmul(layer2, W_fcon2, name='matmul') + b_fcon2
    layer3 = tf.nn.relu(h_fcon2, name='relu')
prediction = tf.identity(layer3, name='prediction')

with tf.name_scope('Train'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=prediction))
    train = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Session
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()

# Training
sess.run(init)
for step in range(iteration):
    batch = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x: batch[0], y_:batch[1]})
    if step %100 == 0:
        print('Step(%d/%d) '%(step,iteration), end='')
        print('Loss:', sess.run(loss, feed_dict={x: batch[0], y_: batch[1]}))

# Save model
saver = tf.train.Saver()
save_path = saver.save(sess, "./model/baseline.ckpt")
print("Model saved in file: %s" % save_path)

# Testing
pre = sess.run(prediction, feed_dict={x: mnist.test.images})
correct_prediction = tf.equal(tf.argmax(pre,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("accuracy: ", sess.run(accuracy, feed_dict={y_:mnist.test.labels}))

# Visualize
#writer = tf.summary.FileWriter('TensorBoard/', graph=sess.graph)

sess.close()
