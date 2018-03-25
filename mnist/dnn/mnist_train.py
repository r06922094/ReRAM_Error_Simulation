from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 55,000 training data
# 10,000 testing data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, shape=[None, 784])
W1 = tf.Variable(tf.random_normal([784, 100], stddev=0.1))
b1 = tf.Variable(tf.zeros([100]))
W2 = tf.Variable(tf.random_normal([100, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W1) + b1
y = tf.nn.relu(y)
y = tf.matmul(y, W2) + b2
y_ = tf.placeholder(tf.float32, shape=[None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Training
for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_:batch[1]})

# Save model
saver = tf.train.Saver()
save_path = saver.save(sess, "./model/model.ckpt")
print("Model saved in file: %s" % save_path)

# Testing
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("accuracy: ",accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels}))
