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
    try:
        return np.log(np.exp(layer)/np.sum(np.exp(layer)))
    except:
        print(layer)

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

    parameters['weights1'] = parameters['weights1'] + gradients['weights1'] * learning_rate
    parameters['weights2'] = parameters['weights2'] + gradients['weights2'] * learning_rate
    parameters['bias1'] = parameters['bias1'] + gradients['bias1'] * learning_rate
    parameters['bias2'] = parameters['bias2'] + gradients['bias2'] * learning_rate

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
    
    def gradient_weight2(a1, z2, i):
        grad_w = np.zeros(network[1])

        for k in range(network[1]):
            #y_train[0] needs to be changed to allow for more tests and the x_train above
            grad_w[k] = -1 * ((np.exp(z2[(y_train[i]-1)])/np.sum(np.exp(z2))) * a1[k]) - a1[k]
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
            for a in range(network[0]):
                grad_w[k][a] = (np.exp(z2[(y_train[i]-1)])/np.sum(np.exp(z2))) * parameters["weights2"][(y_train[i]-1), k] * np.where(np.maximum(z1[k], 0) > 0, 1, 0) * input_data[a]
            
        #need a statement adding the grad and averaging it instead of just changing
        #gradientw1[:] = grad_w
        return grad_w

    def gradient_bias1(z1, z2, parameters, i):
        grad_w = np.zeros(network[1])

        for k in range(network[1]):
                grad_w[k] = (np.exp(z2[(y_train[i]-1)])/np.sum(np.exp(z2))) * parameters["weights2"][(y_train[i]-1), k] * np.where(np.maximum(z1[k], 0) > 0, 1, 0) * 1
            
            
        #need a statement adding the grad and averaging it instead of just changing
        #gradientb1[:] = grad_w
        return grad_w

    for j in range(100):

        gradient_w1 = np.zeros((network[1], network[0]))
        gradient_w2 = np.zeros((network[2], network[1]))
        gradient_b1 = np.zeros((network[1]))
        gradient_b2 = np.zeros((network[2]))
        success = 0

        for i in range(50):
            
            # print(forward(x_train, parameters))
            z1, a1, z2, a2 = forward(x_train[:,(i+j*100)], parameters)
            # print(forward(x_train[:,0], parameters)[1].shape)
            # print(forward(x_train[:,0], parameters)[2].shape)
            # print(forward(x_train[:,0], parameters)[3].shape)
            
            if i == 1:

                gradient_b1[:] = (gradient_bias1(z1, z2, parameters, (i+j*100)))
                gradient_w1[:] = (gradient_weight1(z1, z2, x_train[:,(i+j*100)], parameters, (i+j*100)))
                gradient_b2[(y_train[(i+j*100)]-1)] = (gradient_bias2(z2, (i+j*100)))
                gradient_w2[(y_train[(i+j*100)]-1)] = (gradient_weight2(a1, z2, (i+j*100)))
            
            else:
                gradient_b1[:] = ((gradient_bias1(z1, z2, parameters, (i+j*100))) + gradient_b1[:])/2
                gradient_w1[:] = ((gradient_weight1(z1, z2, x_train[:,(i+j*100)], parameters, (i+j*100))) + gradient_w1[:])/2
                gradient_b2[(y_train[(i+j*100)]-1)] = ((gradient_bias2(z2, (i+j*100))) + gradient_b2[(y_train[(i+j*100)]-1)])/2
                gradient_w2[(y_train[(i+j*100)]-1)] = ((gradient_weight2(a1, z2, (i+j*100))) + gradient_w2[(y_train[(i+j*100)]-1)])/2

            a2 = list(a2)
            if ((a2.index(max(a2)))+1) == y_train[(i+j*100)]:
                success = success + 1

        success = (success/50)*100

        print(f'Success Rate {success} %')

        gradients = {}
        gradients['weights1'] = gradient_w1
        gradients['weights2'] = gradient_w2
        gradients['bias1'] = gradient_b1
        gradients['bias2'] = gradient_b2

        learning_rate = 0.1

        update_parameters(parameters, gradients, learning_rate)
    
    
