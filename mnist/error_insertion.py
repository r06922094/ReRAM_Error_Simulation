import numpy as np
import random
import tensorflow as tf
import pickle as pk
from math import *
'''
def new_composition(X, W, IL, FL, WL, shape, Act_unit, computeType):
    # load error file
    error_list = pk.load(open('Err_file.p', 'rb'))
    if computeType = 'Conv2d':
        # do something
        shiftAndAdd_result_x = []

    elif computeType = 'Matmul':
        for i in range(WL): # X
            for j in range(WL): # W
                compute_result = tf.matmul(X[i], W[j][0])
                iterate = shape[
                for k in range(1, 
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
                #tmp_result = tf.matmul(x_image[i], conv_list[j][0])
                iterate = shape[0] // Act_unit + 1
            for k in range(1, iterate, 1):
                if computeType == 'Conv2d':
                    con_result = tf.nn.conv2d(x_image[i], conv_list[j][k], strides=[1,1,1,1], padding='SAME')
                elif computeType == 'Matmul':
                    con_result = tf.matmul(x_image[i], conv_list[j][k])
                tmp_result += tf.py_func(insert_error, [con_result, error_list, 10], tf.float32)
                #tmp_result += new_insert_error(con_result, error_list, WL, iterate)
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
'''

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
    # input: high precision x
    # output: A list of deomposed x. (one bit)
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

def new_activation_unit(x, unit, computeType): 
    if computeType == 0: # Conv2d(BNHWC)
        shape = x.shape 
        x = x.reshape(shape[0], shape[1], -1) # BNHWC->BNL
    else: # Matmul(BNL)
        pass
    Act = []
    A = ceil(x.shape[2] / unit)
    for a in range(A): # A
        Act.append(np.zeros(x.shape))
        for b in range(x.shape[0]): # B
            for n in range(x.shape[1]): # N
                start = a * unit
                end = (a+1)*unit
                if end > x.shape[2]:
                    end = x.shape[2]
                Act[a][b][n][start:end] = x[b][n][start:end]
    Act = np.array(Act) # ABNL
    if computeType == 0:
        Act = Act.reshape(((-1,)+shape))
    return np.float32(Act) 

def insert_error(arr, err_list, num_of_cell):
    for x in np.nditer(arr, op_flags=['readwrite']):
        probability = random.uniform(0,1)
        for u in range(num_of_cell+1):
            if probability < err_list[num_of_cell-1][int(x[...])][u]:
                x[...] = u
                break
    return arr

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
                tmp_result += tf.py_func(insert_error, [con_result, error_list, 10], tf.float32)
                #tmp_result += new_insert_error(con_result, error_list, WL, iterate)
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

