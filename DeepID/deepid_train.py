import tensorflow as tf
import ReRam_model
import os
from vec import load_data
import numpy as np


def get_weights():
    #return np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()])
    return np.sum([np.prod(v.get_shape().as_list()) \
			for v in tf.global_variables()])

iteration = 200000

testX1, testX2, testY, validX, validY, trainX, trainY = load_data()
class_num = np.max(trainY) + 1
print('class_num: ', class_num)

with tf.name_scope('input'):
    h0 = tf.placeholder(tf.float32, [None, 55, 47, 3], name='x')
    y_ = tf.placeholder(tf.float32, [None, class_num], name='y')

y = ReRam_model.model_DeepID(h0, class_num, False, True)

print(get_weights())

with tf.name_scope('Train'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train = tf.train.AdamOptimizer(1e-4).minimize(loss)




if __name__ == '__main__':
    def get_batch(data_x, data_y, start):
        end = (start + 128) % data_x.shape[0]
        if start < end:
            return data_x[start:end], data_y[start:end], end
        return np.vstack([data_x[start:], data_x[:end]]), np.vstack([data_y[start:], data_y[:end]]), end
    
    data_x = trainX
    data_y = (np.arange(class_num) == trainY[:,None]).astype(np.float32)
    validY = (np.arange(class_num) == validY[:,None]).astype(np.float32)
    
    # initialize Graph
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()

    # Training
    sess.run(init)
    
    
    idx = 0
    for i in range(iteration):
        batch_x, batch_y, idx = get_batch(data_x, data_y, idx)
        train_loss, _ = sess.run([loss, train], {h0: batch_x, y_: batch_y})

    
        if i % 100 == 0:
            print(i, train_loss)

    # Save model
    if not os.path.exists('./model/'):
        os.makedirs('./model/')
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model/model_deepID.ckpt")
    print("Model saved in file: %s" % save_path)
