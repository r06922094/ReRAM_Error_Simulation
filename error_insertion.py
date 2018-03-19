import numpy as np
import random
import tensorflow as tf
import pickle as pk
from math import *

def testing_fun():
    print('testing')
    return

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

def decomposition(x, IL, FL, WL):
    # input: x with fixed point
    # output: A list of deomposed x. Each element in the list is one bit
    xshape = x.shape 
    x = [roundAndConvert(i, IL, FL, WL) for i in np.nditer(x)]
    x = np.array(x).T 
    return np.float32(x).reshape(((-1,) + xshape))

def activation_unit(W, unit, computeType):
    # must be bit-decomposed first 
    Act_list = []
    for i in range(W.shape[0]):
        if computeType == 0: # Conv2d
            S = W[i].reshape((-1, W[i].shape[3])) # straighten
        elif computeType == 1: # Matmul
            S = W[i]
        else:
            S = W[i]
            print(computeType)
        act_list = []
        index = 0
        for j in range(0, S.shape[0], unit):
            act_list.append(np.zeros(S.shape))
            if (j+unit) < S.shape[0]:
                en = j + unit
            else:
                en = S.shape[0]
            act_list[index][j:en] = S[j:en]
            index += 1
        Act_list.append(act_list)
    if computeType == 0:
        Act_list = np.array(Act_list).reshape(W.shape[0], -1, W.shape[1], W.shape[2], W.shape[3], W.shape[4])
    elif computeType == 1:
        Act_list = np.array(Act_list)
    return np.float32(Act_list)

def insert_error(ideal, err_list):
    for x in np.nditer(ideal, op_flags=['readwrite']):
        if x[...] != 0:  
            for u in range(11): # only consider 10 cells accumulation
                if random.uniform(0,1) < err_list[9][int(x[...])][u]:
                    x[...] = u
                    break
    return ideal

#def new_insert_error(arr, err): 



def composition(x_image, conv_list, IL, FL, WL, shape, Act_unit, computeType):
    # load error file
    error_list = pk.load(open('Err_file.p', 'rb'))
    arr = []
    result = []
    for i in range(WL):
        arr.append([])
        for j in range(WL):
            # Deal with activation unit
            if computeType == 'Conv2d':
                tmp_result = tf.nn.conv2d(x_image[i], conv_list[j][0], strides=[1,1,1,1], padding='SAME') 
                iterate = shape[0] * shape[1] * shape[2] // Act_unit + 1
            elif computeType == 'Matmul':
                tmp_result = tf.matmul(x_image[i], conv_list[j][0])
                iterate = shape[0] // Act_unit + 1
            for k in range(1, iterate, 1):
                if computeType == 'Conv2d':
                    con_result = tf.nn.conv2d(x_image[i], conv_list[j][k], strides=[1,1,1,1], padding='SAME')
                elif computeType == 'Matmul':
                    con_result = tf.matmul(x_image[i], conv_list[j][k])
                tmp_result += tf.py_func(insert_error, [con_result, error_list], tf.float32)
            arr[i].append(tmp_result)
        shift = IL - 1
        result.append(-arr[i][0]*(2**shift)) # sign bit
        for k in range(1,WL):
            shift -= 1
            result[i] += arr[i][k] * (2**shift)
    shift = IL - 1
    output = -result[0]*(2**shift) # sign bit
    for k in range(1,WL):
        shift -= 1
        output += result[k] * (2**shift)
    return output

