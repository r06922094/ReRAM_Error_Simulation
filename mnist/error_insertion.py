import numpy as np
import random
import tensorflow as tf
from math import *

ERR_LIST = []

def crossbar(x, w, IL, FL, WL, unit, error_list, computeType):
    x_bit = tf.py_func(decompose_bit, [x, IL, FL, WL], tf.float32)
    w_bit = tf.py_func(decompose_bit, [w, IL, FL, WL], tf.float32)
    w_bit_unit, count_cell = tf.py_func(decompose_unit, [w_bit, unit, computeType], [tf.float32, tf.float32])
    return compute_and_compose(x_bit, w_bit_unit, count_cell, IL, FL, WL, w.shape, unit, computeType, error_list)

def decompose_bit(x, IL, FL, WL):
    # input: high precision x
    # output: A list of deomposed x. (one bit)
    xshape = x.shape 
    x = [roundAndConvert(i, IL, FL, WL) for i in np.nditer(x)]
    x = np.array(x).T
    return np.float32(x).reshape(((-1,) + xshape))

def decompose_unit(w, unit, computeType):
    w = np.int8(w)

    if computeType == 0: # Conv2d(BHWIO)
        shape = w.shape 
        w = w.reshape(shape[0], -1, shape[4]) # BHWIO->BIO
    else: # Matmul(BIO)
        pass

    A = ceil(w.shape[1] / unit)
    
    Act = np.array([w for i in range(A)])
    Cnt = np.zeros((A,)+w.shape, dtype=np.int8)

    start = -unit
    end = 0

    for a in range(A): # A
        start += unit
        end = min(start+unit, w.shape[1])
        for b in range(w.shape[0]): # B
            Cnt[a][b][start:end] = 1

    Act &= Cnt

    if computeType == 0: 
        Act = Act.reshape(((-1,)+shape)) # ABHWIO
        Cnt = Cnt.reshape(((-1,)+shape)) # ABHWIO

    return np.float32(Act), np.float32(Cnt)

def compute_and_compose(x, w, count, IL, FL, WL, shape, unit, computeType, error_list):
    global ERR_LIST
    ERR_LIST = error_list
    
    if computeType == 0: # Conv2d
        iterate = ceil(int(shape[0]*shape[1]*shape[2]) / unit)
    else: # Matmul
        iterate = ceil(int(shape[0]) / unit)
    result = 0
    shift_x = IL - 1
    for b_x in range(WL):
        act_composed = 0
        for a in range(iterate):
            shift_w = IL - 1
            # sign bit:
            if computeType == 0: # Conv2d
                compute_result = tf.nn.conv2d(x[b_x], w[a][0], strides=[1,1,1,1], padding='SAME')
                count_result = tf.nn.conv2d(x[b_x], count[a][0], strides=[1,1,1,1], padding='SAME')
            else: # Matmul
                compute_result = tf.matmul(x[b_x], w[a][0])
                count_result = tf.matmul(x[b_x], count[a][0])
            count_result = tf.reshape(count_result, (-1,))
            result_with_error = tf.py_func(insert_error, [compute_result, count_result, unit], tf.float32)
            bit_composed = result_with_error * (2**shift_w) * (-1)
            for b_w in range(1,WL):
                shift_w -= 1
                if computeType == 0: # Conv2d
                    compute_result = tf.nn.conv2d(x[b_x], w[a][b_w], strides=[1,1,1,1], padding='SAME') 
                    count_result = tf.nn.conv2d(x[b_x], count[a][b_w], strides=[1,1,1,1], padding='SAME')
                else: # Matmul
                    compute_result = tf.matmul(x[b_x], w[a][b_w])
                    count_result = tf.matmul(x[b_x], count[a][b_w])
                count_result = tf.reshape(count_result, (-1,))
                result_with_error = tf.py_func(insert_error, [compute_result, count_result, unit], tf.float32)
                bit_composed += result_with_error * (2**shift_w)
            act_composed += bit_composed
        if b_x == 0:
            result -= act_composed * (2**shift_x) # sign bit
        else:
            result += act_composed * (2**shift_x)
        shift_x -= 1
    return result

def insert_error(arr, m, unit): #m: table_index
    global ERR_LIST
    i = 0
    for x in np.nditer(arr, op_flags=['readwrite']):
        j = int(x)
        k = int(m[i])-1
        if k < 0: continue
        x[...] = ERR_LIST[k][j][random.randint(0, 99)]   
        i += 1
    return arr

def isround(p):
    if random.uniform(0,1) > p:
        return 0
    else:
        return 1

def stochasticRounding(x, FL):
    power = 1 << FL
    tmp = floor(x*power)
    floor_data = tmp / power
    prob = (x - floor_data) * power
    return isround(prob) * (1/power) + floor_data

def convert(x, IL, FL):
    maximum = (1 << (IL-1)) - 1/float(1<<FL)
    minimum = -1 * (1<<(IL-1))
    if x >= maximum:
        return maximum
    elif x <= minimum:
        return minimum
    else:
        return stochasticRounding(x, FL)

def convert2TwosComplement(x, FL, WL):
    power = 1 << FL
    x = int(x * power)
    binstr = np.binary_repr(x, width=WL)
    arr = list(map(int, list(binstr)))
    return arr

def roundAndConvert(x, IL, FL, WL):
    x_round = convert(x, IL, FL)
    arr = convert2TwosComplement(x_round, FL, WL)
    return arr
