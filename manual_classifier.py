import numpy as np
import math 
import read_data

def linear(weight, bias, input_data):
    out = np.dot(weight, input_data)
    out = out + bias
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
        bias =  np.zeros((network[i]))

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
    gradient_w1 = np.zeros((network[1], network[0]))
    gradient_w2 = np.zeros((network[2], network[1]))
    gradient_b1 = np.zeros((network[1]))
    gradient_b2 = np.zeros((network[2]))

    # Init weights and bias
    parameters = initialize_parameters(network)

    # Training data
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)

    # Reshape data and transpose
    x_train = np.reshape(x_train, newshape=(x_train.shape[0], -1)).T
    y_train = np.reshape(y_train, newshape=(y_train.shape[0], -1))
    
    def gradient_weight2(a1, z2, i):
        grad_w = np.zeros(network[1])

        for i in range(network[1]):
            #y_train[0] needs to be changed to allow for more tests and the x_train above
            grad_w[i] = -1 * ((np.exp(z2[(y_train[i]-1)])/np.sum(np.exp(z2))) * a1[i]) - a1[i]
        #need a statement adding the grad and averaging it instead of just changing
        #gradientw2[(y_train[0]-1)] = grad_w
        return grad_w

    def gradient_bias2(z2, i):
        grad_b = -1 * (np.exp(z2[(y_train[i]-1)])/np.sum(np.exp(z2))) - 1
        
        #gradientb2[(y_train[0]-1)] = grad_b
        return grad_b

    def gradient_weight1(z1, z2, input_data, parameters, i):
        grad_w = np.zeros((network[1], network[0]))

        for k in range(network[1]):
            for i in range(network[0]):
                grad_w[k][i] = (np.exp(z2[(y_train[i]-1)])/np.sum(np.exp(z2))) * parameters["weights2"][(y_train[i]-1), k] * np.maximum(z1[k], 0) * input_data[i]
            
        #need a statement adding the grad and averaging it instead of just changing
        #gradientw1[:] = grad_w
        return grad_w

    def gradient_bias1(z1, z2, parameters, i):
        grad_w = np.zeros(network[1])

        for k in range(network[1]):
                grad_w[k] = (np.exp(z2[(y_train[i]-1)])/np.sum(np.exp(z2))) * parameters["weights2"][(y_train[i]-1), k] * np.maximum(z1[k], 0) * 1
            
        #need a statement adding the grad and averaging it instead of just changing
        #gradientb1[:] = grad_w
        return grad_w



    for i in range(10):
        # print(forward(x_train, parameters))
        z1, a1, z2, a2 = forward(x_train[:,i], parameters)
        # print(forward(x_train[:,0], parameters)[1].shape)
        # print(forward(x_train[:,0], parameters)[2].shape)
        # print(forward(x_train[:,0], parameters)[3].shape)
        
        if i == 1:

            gradient_b1[:] = (gradient_bias1(z1, z2, parameters, i))
            gradient_w1[:] = (gradient_weight1(z1, z2, x_train[:,i], parameters, i))
            gradient_b2[(y_train[i]-1)] = (gradient_bias2(z2, i))
            gradient_w2[(y_train[i]-1)] = (gradient_weight2(a1, z2, i))
        
        else:
            gradient_b1[:] = ((gradient_bias1(z1, z2, parameters, i)) + gradient_b1[:])/2
            gradient_w1[:] = ((gradient_weight1(z1, z2, x_train[:,i], parameters, i)) + gradient_w1[:])/2
            gradient_b2[(y_train[i]-1)] = ((gradient_bias2(z2, i)) + gradient_b2[(y_train[i]-1)])/2
            gradient_w2[(y_train[i]-1)] = ((gradient_weight2(a1, z2, i)) + gradient_w2[(y_train[i]-1)])/2
        
    print(gradient_w1)
    print(gradient_w2)
    print(gradient_b1)
    print(gradient_b2)
    a2 = list(a2)
    print((a2.index(max(a2)))+1)
    print(y_train)

  
