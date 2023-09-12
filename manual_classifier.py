import numpy as np
import math 
import read_data
import torch
import torch.nn as nn
import torch.nn.functional as F

def linear(weight, bias, input_data):
    out = np.dot(weight, input_data) + bias
    return out

def relu(layer):
    return np.maximum(layer, 0)

def logsoftmax(layer):
    return np.log(np.exp(layer)/np.sum(np.exp(layer)))

def logsoftmaxbackward(layer):
    softmax = np.exp(layer)/np.sum(np.exp(layer))
    grad = - softmax
    return grad

def relubackward(layer):
    return np.where(layer > 0, np.ones(layer.shape), np.zeros(layer.shape))

def forward(input_data, parameters):
    z1 = linear(parameters["weights1"], parameters["bias1"], input_data)
    a1 = relu(z1)
    z2 = linear(parameters["weights2"], parameters["bias2"], a1)
    a2 = logsoftmax(z2)
    return z1, a1, z2, a2 

def initialize_parameters(network):
    np.random.seed(1)
    parameters = {}
    for i in range (1, len(network)):
        weights = np.random.randn(network[i], network[i-1])*0.01
        bias =  np.zeros((network[i], 1))
        
        # Store in dictionary
        parameters['weights' + str(i)] = weights
        parameters['bias' + str(i)] = bias
    return parameters

def backward(parameters):
    grads = {}

    

def update_parameters(parameters, gradients, learning_rate):
    # print(f"weights1: {parameters['weights1'].shape}")
    parameters['weights1'] = parameters['weights1'] - gradients['weights1'] * learning_rate
    # print(f"weights1: {parameters['weights1'].shape}")
    parameters['weights2'] = parameters['weights2'] - gradients['weights2'] * learning_rate
    print(f"param bias1 : {parameters['bias1'].shape}")
    print(f"grad bias1: {gradients['bias1'].shape}")
    parameters['bias1'] = parameters['bias1'] - gradients['bias1'].reshape(parameters['bias1'].shape) * learning_rate
    print(f"param bias1: {parameters['bias1'].shape}")
    parameters['bias2'] = parameters['bias2'] - gradients['bias2'].reshape(parameters['bias2'].shape) * learning_rate


def calculate_cost(output, label):
    loss = -output[label, range(len(label))]   
    return np.mean(loss)


def train(x_train, y_train):

    # Init stuff, can be put into a class later
    network = [784, 196, 10]
    # Init weights and bias
    parameters = initialize_parameters(network)

    # Training data
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)

    # Reshape data and transpose
    x_train = np.reshape(x_train, newshape=(x_train.shape[0], -1)).T
    # y_train = np.reshape(y_train, newshape=(y_train.shape[0], -1))

    #select batch from x_train and y_train
    x_train = x_train[:, 0:1]
    y_train = y_train[0:1]

    _, _, _, output = forward(x_train, parameters)
    
    epoch = 5

    for i in range(epoch):
        # Forward Pass
        z1, a1, z2, output = forward(x_train, parameters)
        cost = calculate_cost(output, y_train)
        print(f"cost: {cost}")
        #Update parameters

        def jostuff():

            dz2_lsm = 1 - np.exp(z2[y_train, range(len(y_train))])/np.sum(np.exp(z2[:, range(len(y_train))]))
            dw2_z2 = a1 #k
            da2_cost = -1
            db2_z2 = 1

            grad_w1 = np.zeros((196, 784))
            grad_b1 = np.zeros(196)
            
            for j in range(len(y_train)):
                for k in range(196):
                    for i in range(784):
                        da1_z2 = parameters['weights2'][(y_train[j]-1)][k] #k
                        dz1_da1 = np.where(np.maximum(z1[k][j], 0) > 0, 1, 0) #k
                        dw1_z1 = x_train[i,j] #i
                        db1_z1 = 1

                        print(da1_z2)
                        print(dz1_da1)
                        print(dw1_z1)
                        print(db1_z1)


                        grad_w1[k][i] = dz2_lsm * da2_cost * da1_z2 * dz1_da1 #* dw1_z1
                        grad_b1[k] = dz2_lsm * da2_cost * da1_z2 * dz1_da1 * db1_z1
                
            grad_w2 = dz2_lsm * dw2_z2 * da2_cost
            grad_b2 = dz2_lsm * db2_z2 * da2_cost
            
           
            return grad_w1, grad_b1
        
       
        def benstuff():
            grads = {}
            
            m = output.shape[1]
            d_output = - output/ m

            dZ = d_output - logsoftmaxbackward(z2)
            dW = 1/m * np.dot(dZ, a1.T)
            db = 1/m * np.sum(dZ, axis=1)

            # print(f"dZ: {dZ}")
            # print(f"dW: {dW}")
            print(f"db: {db}")

            grads['weights2'] = dW
            grads['bias2'] = db
            
            d_p = np.dot(parameters['weights2'].T, dZ)
            dZ = d_p * relubackward(z1)
            dW = 1/m * np.dot(dZ, x_train.T)
            db = 1/m * np.sum(dZ, axis=1)

            # print(f"dZ: {dZ.shape}")
            # print(f"dW: {dW.shape}")
            print(f"db: {db.shape}")

            grads['weights1'] = dW
            grads['bias1'] = db

            update_parameters(parameters, grads, 0.01)

          # dW = 1/m * np.dot(dZ, .T)

       
            # grads["dA"]

            # d_output =  - output/ len(y_train)
            # lsm = z2 - logsoftmax(z2)
            # d_lsm = d_output - np.exp(lsm)*d_output.sum(axis=1).reshape(-1,1)
            # d_z2 = a1.dot(d_lsm)
            # d_relu = d_z2.dot(z2)
            # d_z1 = (a1 > 0) * d_relu
            # d_final = x_train.dot(d_z1)

            # grads = {}
            # grads['weights1'] = d_z1
            # grads['weights2'] = d_z2
            # update_parameters(parameters, grads, 0.01)

        benstuff()
            

        
    


    
    print(f"cost: {cost}")
