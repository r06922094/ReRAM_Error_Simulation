from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 55,000 training data
# 10,000 testing data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
iteration = 20000

# Create Model
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
x_image = tf.reshape(x, [-1, 28, 28, 1])

# ==Convolution layer== #
with tf.name_scope('Conv1'):
    W_conv1	= tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1), name='weight') # 5x5, input_size=1, output_size=32
    b_conv1 = tf.Variable(tf.zeros([32]), name='bias')
    h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME', name='convolution') + b_conv1
    h_conv1 = tf.nn.relu(h_conv1, name='relu') # 28x28x32
    layer_output1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pooling') # 14x14x32
with tf.name_scope('Conv2'):
    W_conv2	= tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1), name='weight') # 5x5, input_size=32, output_size=64
    b_conv2 = tf.Variable(tf.zeros([64]), name='bias')
    h_conv2 = tf.nn.conv2d(layer_output1, W_conv2, strides=[1,1,1,1], padding='SAME', name='convolution') + b_conv2
    h_conv2 = tf.nn.relu(h_conv2, name='relu') # 28x28x32
    layer_output2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pooling') # 7x7x64

# ==Fully connected layer== #
with tf.name_scope('Dense1'):
    flatten = tf.reshape(layer_output2, [-1,7*7*64], name='flatten')
    W_fcon1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1), name='weight')
    b_fcon1 = tf.Variable(tf.zeros([1024]), name='bias')
    h_fcon1 = tf.matmul(flatten, W_fcon1, name='formula') + b_fcon1
    h_fcon1 = tf.nn.relu(h_fcon1, name='relu')
    layer_output3 = tf.nn.dropout(h_fcon1, 0.5, name='dropout')
with tf.name_scope('Dense2'):
    W_fcon2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), name='weight')
    b_fcon2 = tf.Variable(tf.zeros([10]), name='bias')
    h_fcon2 = tf.matmul(layer_output3, W_fcon2, name='formula') + b_fcon2
    layer_output4 = tf.nn.relu(h_fcon2, name='relu')

prediction = tf.identity(layer_output4, name='prediction')
with tf.name_scope('Train'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=prediction))
    train = tf.train.AdamOptimizer(1e-4).minimize(loss)

# initialize Graph
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
save_path = saver.save(sess, "./model/cnn_large_model_new.ckpt")
print("Model saved in file: %s" % save_path)

# Testing
pre = sess.run(prediction, feed_dict={x: mnist.test.images})
correct_prediction = tf.equal(tf.argmax(pre,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("accuracy: ", sess.run(accuracy, feed_dict={y_:mnist.test.labels}))

# Visualize
writer = tf.summary.FileWriter('TensorBoard/', graph=sess.graph)

sess.close()
