import numpy as np
import math 
import read_data

Neurons = [784, 196, 196, 10]

l0 = np.zeros(Neurons[0])
l1 = np.zeros(Neurons[1])
l2 = np.zeros(Neurons[2])
l3 = np.zeros(Neurons[3])

Weightslp1 = np.tile(0.01, (Neurons[1], Neurons[0]))
Weightslp2 = np.tile(0.01, (Neurons[3], Neurons[2]))

Biaslp1 = np.tile(0.01, Neurons[1])
Biaslp2 = np.tile(0.01, Neurons[3])

def linear(weight, bias, input):
    out = np.dot(weight, input)
    out = np.add(out, bias)
    return out

def relu(layer):
    for i in range(len(layer)):
        if layer[i] < 0:
            layer[i] = 0
    return layer

def logsoftmax(layer):
    for i in range(len(layer)):
        try:
            layer[i] = math.exp(layer[i])
        except:
            print(layer[i])

    sumlayer = layer.sum()
    for i in range(len(layer)):
        layer[i] = np.log(layer[i]/sumlayer)
    return layer

def forward(l0):
    l1 = relu(linear(Weightslp1, Biaslp1, l0))
    l2 = logsoftmax(linear(Weightslp2, Biaslp2, l1))

    return l2

(x_train, y_train),(x_test, y_test) = read_data.load_data()
x_train = np.array(x_train[3])

l0 = x_train.flatten()

print(forward(l0))
