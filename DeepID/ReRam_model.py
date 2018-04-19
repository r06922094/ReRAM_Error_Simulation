import tensorflow as tf
import error_insertion_v2 as ei
import pickle as pk
import numpy as np

# mnist: (2, 4)
# cifar: (3, 6)
# deepid:(3, 6) 0.90
IL = 3
FL = 6
WL = IL + FL
unit  = 9

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name = 'Weights')

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = 'Biases')

def _bn(x, is_train, global_step=None, name='bn'):
    moving_average_decay = 0.9
    # moving_average_decay = 0.99
    # moving_average_decay_init = 0.99
    with tf.variable_scope(name):
        decay = moving_average_decay
        # if global_step is None:
            # decay = moving_average_decay
        # else:
            # decay = tf.cond(tf.greater(global_step, 100)
                            # , lambda: tf.constant(moving_average_decay, tf.float32)
                            # , lambda: tf.constant(moving_average_decay_init, tf.float32))
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        with tf.device('/CPU:0'):
            mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                            initializer=tf.zeros_initializer(), trainable=False)
            sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer(), trainable=False)
            beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                            initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer())
        # BN when training
        update = 1.0 - decay
        # with tf.control_dependencies([tf.Print(decay, [decay])]):
            # update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_sigma = sigma.assign_sub(update*(sigma - batch_var))
        #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
        #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

        mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
                            lambda: (mu, sigma))
        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

        # bn = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-5)

        # bn = tf.contrib.layers.batch_norm(inputs=x, decay=decay,
                                          # updates_collections=[tf.GraphKeys.UPDATE_OPS], center=True,
                                          # scale=True, epsilon=1e-5, is_training=is_train,
                                          # trainable=True)
    return bn

def _conv(x, filter_size, out_channel, strides, pad='SAME', name='conv'):
    in_channel = x.get_shape().as_list()[3]
    n = filter_size*filter_size*out_channel
    with tf.variable_scope(name):
        W_conv = weight_variable([filter_size,filter_size,in_channel, out_channel], np.sqrt(2.0/n))
        x = tf.nn.conv2d(x, W_conv, strides=[1,strides,strides,1], padding=pad)
    return x


def _residual_block_first(x, out_channel, strides, is_train, name="unit"):
    in_channel = np.asarray(x.get_shape().as_list()[-1])
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)
        # Shortcut connection
        if in_channel == out_channel:
            shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
        else:
            shortcut = _conv(x, 1, out_channel, strides, name='shortcut')
        # Residual

        x = _conv(x, 3, out_channel, strides, name='conv_1')
        x = _bn(x, is_train, name='bn_1')
        x = tf.nn.relu(x, name='relu_1')
        x = _conv(x, 3, out_channel, 1, name='conv_2')
        x = _bn(x, is_train, name='bn_2')
        # Merge
        x = x + shortcut
        x = tf.nn.relu(x, name='relu_2')
    return x
def _residual_block(x, is_train, name="unit"):
    num_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)
        # Shortcut connection
        shortcut = x
        # Residual
        x = _conv(x, 3, num_channel, 1, name='conv_1')
        x = _bn(x, is_train, name='bn_1')
        x = tf.nn.relu(x, name='relu_1')
        x = _conv(x, 3, num_channel, 1, name='conv_2')
        x = _bn(x, is_train, name='bn_2')
        x = x + shortcut
        x = tf.nn.relu(x, name='relu_2')
    return x
def model_mnist(x_image, keep_prob, Err_insert = True, err_path=''):
    
    error_list = []
    if err_path:
        error_list = pk.load(open(err_path, 'rb'))
        print('load ', err_path)
    
        # ==Convolution layer== #
    with tf.name_scope('Conv1'):
        print('build Conv1...')
        W_conv1 = weight_variable([3,3,1,32])
        b_conv1 = bias_variable([32])
        if Err_insert :
            h_conv1 = ei.crossbar(x_image, W_conv1, IL, FL, WL, unit, error_list, 0) + b_conv1
        else:
            h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1
        h_conv1 = tf.nn.relu(h_conv1) # 28x28x32
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 14x14x32

    with tf.name_scope('Conv2'):
        print('build Conv2...')
        W_conv2 = weight_variable([3,3,32,64])
        b_conv2 = bias_variable([64])
        if Err_insert :
            h_conv2 = ei.crossbar(h_pool1, W_conv2, IL, FL, WL, unit, error_list, 0) + b_conv2
        else:
            h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2
        h_conv2 = tf.nn.relu(h_conv2) # 14x14x64
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 7x7x64


    # ==Fully connected layer== #
    with tf.name_scope('Dense1'):
        print('build Dense1...')
        W_fcon1 = weight_variable([7*7*64, 128])
        b_fcon1 = bias_variable([128])
        flatten = tf.reshape(h_pool2, [-1,7*7*64], name = 'Flatten')
        with tf.name_scope('Formula'):
            if Err_insert :
                h_fcon1 = ei.crossbar(flatten, W_fcon1, IL, FL, WL, unit, error_list, 1) + b_fcon1
            else:
                h_fcon1 = tf.matmul(flatten, W_fcon1) + b_fcon1
        h_fcon1 = tf.nn.relu(h_fcon1)
        h_drop1 = tf.nn.dropout(h_fcon1, keep_prob)
    with tf.name_scope('Dense2'):
        print('build Dense2...')
        W_fcon2 = weight_variable([128, 10])
        b_fcon2 = bias_variable([10])
        with tf.name_scope('Formula'):
            if Err_insert :
                h_fcon2 = ei.crossbar(h_drop1, W_fcon2, 3, 13, 16, unit, error_list, 1) + b_fcon2
            else:
                h_fcon2 = tf.matmul(h_drop1, W_fcon2) + b_fcon2
    
    return h_fcon2


def model_cifar10(x_image, err_path = '', Err_insert = True, test_bit = False, half=False):
    error_list = []
    if err_path:
        error_list = pk.load(open(err_path, 'rb'))
        print('load ', err_path)
    with tf.name_scope('Conv1'):
        W_conv1 = weight_variable([5,5,3,64])
        b_conv1 = bias_variable([64])
        if Err_insert :
            print('build Conv1...')
            if test_bit:
                x_image = tf.py_func(ei.just_convert_bit, [x_image, IL, FL, WL], tf.float32)
                W_conv1 = tf.py_func(ei.just_convert_bit, [W_conv1, IL, FL, WL], tf.float32)
                h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1
            else:
                h_conv1 = ei.crossbar(x_image, W_conv1, IL, FL, WL, unit, error_list, 0) + b_conv1
        else:
            h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1
        h_conv1 = tf.nn.relu(h_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    with tf.name_scope('Conv2'):
        W_conv2 = weight_variable([5,5,64,64])
        b_conv2 = bias_variable([64])
        if Err_insert :
            print('build Conv2...')
            if test_bit:
                h_pool1 = tf.py_func(ei.just_convert_bit, [h_pool1, IL, FL, WL], tf.float32)
                W_conv2 = tf.py_func(ei.just_convert_bit, [W_conv2, IL, FL, WL], tf.float32)
                h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2
            else:
                h_conv2 = ei.crossbar(h_pool1, W_conv2, IL, FL, WL, unit, error_list, 0) + b_conv2
        else:
            h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2
        h_conv2 = tf.nn.relu(h_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 6x6x64

    with tf.name_scope('Conv3'):
        W_conv3 = weight_variable([3,3,64,64])
        b_conv3 = bias_variable([64])
        if Err_insert :
            print('build Conv3...')
            if test_bit:
                h_pool2 = tf.py_func(ei.just_convert_bit, [h_pool2, IL, FL, WL], tf.float32)
                W_conv3 = tf.py_func(ei.just_convert_bit, [W_conv3, IL, FL, WL], tf.float32)
                h_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1,1,1,1], padding='SAME') + b_conv3
            else:
                h_conv3 = ei.crossbar(h_pool2, W_conv3, IL, FL, WL, unit, error_list, 0) + b_conv3
        else:
            h_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1,1,1,1], padding='SAME') + b_conv3

        h_conv3 = tf.nn.relu(h_conv3)
        h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 6x6x64

    if half:
        return h_pool3
    # ==Fully connected layer== #
    with tf.name_scope('Dense1'):

        flatten = tf.reshape(h_pool3, [-1, 576], name = 'Flatten')
        W_fcon1 = weight_variable([576, 384])
        b_fcon1 = bias_variable([384])

        with tf.name_scope('Formula'):
            if Err_insert :
                print('build Dense1...')
                if test_bit:
                    flatten = tf.py_func(ei.just_convert_bit, [flatten, IL, FL, WL], tf.float32)
                    W_fcon1 = tf.py_func(ei.just_convert_bit, [W_fcon1, IL, FL, WL], tf.float32)
                    h_fcon1 = tf.matmul(flatten, W_fcon1) + b_fcon1
                else:
                    h_fcon1 = ei.crossbar(flatten, W_fcon1, IL, FL, WL, unit, error_list, 1) + b_fcon1
            else:
                h_fcon1 = tf.matmul(flatten, W_fcon1) + b_fcon1
        h_fcon1 = tf.nn.relu(h_fcon1)
    

    

    with tf.name_scope('Dense2'):
    
        W_fcon2 = weight_variable([384, 192])
        b_fcon2 = bias_variable([192])
        with tf.name_scope('Formula'):
            if Err_insert :
                print('build Dense2...')
                if test_bit:
                    h_fcon1 = tf.py_func(ei.just_convert_bit, [h_fcon1, IL, FL, WL], tf.float32)
                    W_fcon2 = tf.py_func(ei.just_convert_bit, [W_fcon2, IL, FL, WL], tf.float32)
                    h_fcon2 = tf.matmul(h_fcon1, W_fcon2) + b_fcon2
                else:
                    h_fcon2 = ei.crossbar(h_fcon1, W_fcon2, IL, FL, WL, unit, error_list, 1) + b_fcon2
            else:
                h_fcon2 = tf.matmul(h_fcon1, W_fcon2) + b_fcon2

        h_fcon2 = tf.nn.relu(h_fcon2)

    with tf.name_scope('Dense3'):
    
        W_fcon3 = weight_variable([192, 10])
        b_fcon3 = bias_variable([10])
        with tf.name_scope('Formula'):
            if Err_insert :
                print('build Dense3...')
                if test_bit:
                    h_fcon2 = tf.py_func(ei.just_convert_bit, [h_fcon2, IL, FL, WL], tf.float32)
                    W_fcon3 = tf.py_func(ei.just_convert_bit, [W_fcon3, IL, FL, WL], tf.float32)
                    h_fcon3 = tf.matmul(h_fcon2, W_fcon3) + b_fcon3
                else:
                    h_fcon3 = ei.crossbar(h_fcon2, W_fcon3, 3, 13, 16, unit, error_list, 1) + b_fcon3
            else:
                h_fcon3 = tf.matmul(h_fcon2, W_fcon3) + b_fcon3

    return h_fcon3
        
def model_DeepID(x_image, class_num, Err_insert = True, Training = False, err_path = ''):
    error_list = []
    if err_path:
        error_list = pk.load(open(err_path, 'rb'))
        print('load ', err_path)
    x_image = tf.cast(x_image, tf.float32) / 255.

    with tf.name_scope('Conv1'):
        W_conv1 = weight_variable([4,4,3,20])
        b_conv1 = bias_variable([20])
        if Err_insert :
            print('build Conv1...')
            #x_image = tf.py_func(ei.just_convert_bit, [x_image, IL, FL, WL], tf.float32)
            #W_conv1 = tf.py_func(ei.just_convert_bit, [W_conv1, IL, FL, WL], tf.float32)
            #h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='VALID') + b_conv1
            h_conv1 = ei.crossbar(x_image, W_conv1, IL, FL, WL, unit, error_list, 0, padding_type='VALID') + b_conv1
        else:
            h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='VALID') + b_conv1
        h_conv1 = tf.nn.relu(h_conv1) # 28x28x32
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    with tf.name_scope('Conv2'):
        W_conv2 = weight_variable([3,3,20,40])
        b_conv2 = bias_variable([40])
        if Err_insert :
            print('build Conv2...')
            #h_pool1 = tf.py_func(ei.just_convert_bit, [h_pool1, IL, FL, WL], tf.float32)
            #W_conv2 = tf.py_func(ei.just_convert_bit, [W_conv2, IL, FL, WL], tf.float32)
            #h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='VALID') + b_conv2
            h_conv2 = ei.crossbar(h_pool1, W_conv2, IL, FL, WL, unit, error_list, 0, padding_type='VALID') + b_conv2
        else:
            h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='VALID') + b_conv2
        h_conv2 = tf.nn.relu(h_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    with tf.name_scope('Conv3'):
        W_conv3 = weight_variable([3,3,40,60])
        b_conv3 = bias_variable([60])
        if Err_insert :
            print('build Conv3...')
            #h_pool2 = tf.py_func(ei.just_convert_bit, [h_pool2, IL, FL, WL], tf.float32)
            #W_conv3 = tf.py_func(ei.just_convert_bit, [W_conv3, IL, FL, WL], tf.float32)
            #h_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1,1,1,1], padding='VALID') + b_conv3
            h_conv3 = ei.crossbar(h_pool2, W_conv3, IL, FL, WL, unit, error_list, 0, padding_type='VALID') + b_conv3
        else:
            h_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1,1,1,1], padding='VALID') + b_conv3
        h_conv3 = tf.nn.relu(h_conv3)
        h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    with tf.name_scope('Conv4'):
        W_conv4 = weight_variable([2,2,60,80])
        b_conv4 = bias_variable([80])
        if Err_insert :
            print('build Conv4...')
            #h_pool3 = tf.py_func(ei.just_convert_bit, [h_pool3, IL, FL, WL], tf.float32)
            #W_conv4 = tf.py_func(ei.just_convert_bit, [W_conv4, IL, FL, WL], tf.float32)
            #h_conv4 = tf.nn.conv2d(h_pool3, W_conv4, strides=[1,1,1,1], padding='VALID') + b_conv4
            h_conv4 = ei.crossbar(h_pool3, W_conv4, IL, FL, WL, unit, error_list, 0, padding_type='VALID') + b_conv4
        else:
            h_conv4 = tf.nn.conv2d(h_pool3, W_conv4, strides=[1,1,1,1], padding='VALID') + b_conv4
        h_conv4 = tf.nn.relu(h_conv4)
    
    # ==Fully connected layer== #
    with tf.name_scope('Dense1'):

        flatten_h3 = tf.reshape(h_pool3, [-1, 5*4*60], name = 'Flatten')
        W_fcon_h3 = weight_variable([5*4*60, 160])

        flatten_h4 = tf.reshape(h_conv4, [-1, 4*3*80], name = 'Flatten')
        W_fcon_h4 = weight_variable([4*3*80, 160])

        b_fcon1 = bias_variable([160])


        with tf.name_scope('Formula'):
            if Err_insert :
                print('build Dense1...')
                #flatten_h3 = tf.py_func(ei.just_convert_bit, [flatten_h3, IL, FL, WL], tf.float32)
                #W_fcon_h3 = tf.py_func(ei.just_convert_bit, [W_fcon_h3, IL, FL, WL], tf.float32)
                #flatten_h4 = tf.py_func(ei.just_convert_bit, [flatten_h4, IL, FL, WL], tf.float32)
                #W_fcon_h4 = tf.py_func(ei.just_convert_bit, [W_fcon_h4, IL, FL, WL], tf.float32)

                #h_fcon1 = tf.matmul(flatten_h3, W_fcon_h3) + tf.matmul(flatten_h4, W_fcon_h4) + b_fcon1

                h_fcon1 = ei.crossbar(flatten_h3, W_fcon_h3, IL, FL, WL, unit, error_list, 1) \
                        + ei.crossbar(flatten_h4, W_fcon_h4, IL, FL, WL, unit, error_list, 1) \
                        + b_fcon1
            else:
                h_fcon1 = tf.matmul(flatten_h3, W_fcon_h3) + tf.matmul(flatten_h4, W_fcon_h4) + b_fcon1
        h_fcon1 = tf.nn.relu(h_fcon1)
    
    if Training:
        with tf.name_scope('Dense2'):
            W_fcon2 = weight_variable([160, class_num])
            b_fcon2 = bias_variable([class_num])
            h_fcon2 = tf.matmul(h_fcon1, W_fcon2) + b_fcon2

        return h_fcon2
    else:
        return h_fcon1


def model_resNet18(x_image, class_num, is_train):

    filters = [64, 64, 128, 256, 512]
    kernels = [7, 3, 3, 3, 3]
    strides = [2, 0, 2, 2, 2]
    

    with tf.variable_scope('Conv1'):
        h_conv1 = _conv(x_image, 7, 64, 2)
        h_conv1 = _bn(h_conv1, is_train)
        h_conv1 = tf.nn.relu(h_conv1) # 28x28x32
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    with tf.variable_scope('Conv2'):
        h_conv2_1 = _residual_block(h_pool1, is_train, name='conv2_1')
        h_conv2_2 = _residual_block(h_conv2_1, is_train, name='conv2_2')
    with tf.variable_scope('Conv3'):
        h_conv3_1 = _residual_block_first(h_conv2_2, 128, 2, is_train, name='conv3_1')
        h_conv3_2 = _residual_block(h_conv3_1, is_train, name='conv3_2')
    with tf.variable_scope('Conv4'):
        h_conv4_1 = _residual_block_first(h_conv3_2, 256, 2, is_train, name='conv4_1')
        h_conv4_2 = _residual_block(h_conv4_1, is_train, name='conv4_2')
    with tf.variable_scope('Conv5'):
        h_conv5_1 = _residual_block_first(h_conv4_2, 512, 2, is_train, name='conv5_1')
        h_conv5_2 = _residual_block(h_conv5_1, is_train, name='conv5_2')
    with tf.variable_scope('Dense1'):
        avg_pool = tf.reduce_mean(h_conv5_2, [1, 2])
        in_channel = np.asarray(avg_pool.get_shape().as_list()[1])
        W_fcon1 = weight_variable([in_channel, class_num], np.sqrt(1.0/class_num))
        b_fcon1 = bias_variable([class_num])
        h_fcon1 = tf.nn.bias_add(tf.matmul(avg_pool, W_fcon1), b_fcon1)

    logit = h_fcon1
    probs = tf.nn.softmax(h_fcon1)

    return logit, probs
    

    