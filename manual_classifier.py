import numpy as np
import math 
import read_data

def linear(weight, bias, input_data):
    out = np.dot(weight, input_data)
    out = np.add(out, bias)
    return out

def relu(layer):
    return np.maximum(layer, 0)
    # for i in range(len(layer)):
    #     if layer[i] < 0:
    #         layer[i] = 0
    # return layer

def logsoftmax(layer):
    # for i in range(len(layer)):
    #     try:
    #         layer[i] = math.exp(layer[i])
    #     except:
    #         print(layer[i])
    # sumlayer = layer.sum()
    # for i in range(len(layer)):
    #     layer[i] = np.log(layer[i]/sumlayer)
    return np.log(np.exp(layer)/np.sum(np.exp(layer)))


def forward(input_data, parameters):
    l1 = relu(linear(parameters["weights1"], parameters["bias1"], input_data))
    l2 = logsoftmax(linear(parameters["weights2"], parameters["bias2"], l1))
    return l2

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

def update_parameters(parameters, gradients, learning_rate):
    print("to be implemented")

def calculate_cost(output, label):
    # CONVERT LABEL INTO ONE HOT ENCODING
    #  -1 * out * label 
    # [0,0,0,0,1,0,0,0,0,0]
    # [-435,-45,45,56,56,546,2,6]
    return -1 * output * label 

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
    y_train = np.reshape(y_train, newshape=(y_train.shape[0], -1))

    print(y_train[0])
   
    # print(forward(x_train, parameters))
    # print(forward(x_train, parameters).shape)


  
