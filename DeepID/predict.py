#! /usr/bin/python
import pickle
import tensorflow as tf
import ReRam_model
from scipy.spatial.distance import cosine, euclidean
from vec import read_csv_pair_file
import time
import numpy as np

if __name__ == '__main__':


    print('load data...')
    load_start = time.time()
    testX1, testX2, testY = read_csv_pair_file('data/test_set.csv')
    testX1 = np.asarray(testX1[0:5], dtype='float32')
    testX2 = np.asarray(testX2[0:5], dtype='float32')
    testY = np.asarray(testY[0:5], dtype='float32')


    print('build model...')
    with tf.name_scope('input'):
        h0 = tf.placeholder(tf.float32, [None, 55, 47, 3], name='x')

    h5 = ReRam_model.model_DeepID(h0, 0, Err_insert = True, Training = False)



    

    print('run eval...')
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './model/model_deepID.ckpt')

        for i in range(5):
            start_time = time.time()



            h1 = sess.run(h5, {h0: [testX1[i]]})
            h2 = sess.run(h5, {h0: [testX2[i]]})

            pre_y = np.array([1-cosine(x, y) for x, y in zip(h1, h2)])
            print('pre_y: ', pre_y, 'testY: ', testY)

            print('time: ', time.time()-start_time)

    '''
    def part_mean(x, mask):
        z = x * mask
        return float(np.sum(z) / np.count_nonzero(z))
    
    
    true_mean = part_mean(pre_y, testY)
    false_mean = part_mean(pre_y, 1-testY)
    print(true_mean, false_mean)
    
    print(np.mean((pre_y < (true_mean + false_mean)/2) == testY.astype(bool)))
    '''