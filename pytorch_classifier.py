import torch
import torch.nn as nn
import torch.nn.functional as F
import torchviz

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(28*28, 14*14)
        self.l2 = nn.Linear(14*14, 10)
        self.sm = nn.LogSoftmax()

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x) 
        x = self.sm(x)
        return x
    
def train(model, x_train, y_train):
    # model.train()
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0)
    batch_size = 128
    epochs = 1000

    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    
    losses, accuracies = [], []

    for i in range(epochs):
        samp = np.random.choice(len(x_train), size=batch_size)
        x_batch = torch.tensor(x_train[samp]).reshape(batch_size, 28 * 28)
        y_batch = torch.tensor(y_train[samp])
        optimizer.zero_grad()

        output = model(x_batch)    
        # print(output)    

        loss = loss_function(output, y_batch)
        loss.backward()
        optimizer.step()

        best = torch.argmax(output, dim=1)
        accuracy = (best == y_batch).float().mean()
        losses.append(loss.item())
        accuracies.append(accuracy.item())

    print(f"Epoch: {i}, Loss: {loss.item()}, Accuracy: {accuracy.item()}")

    #plot the loss and accuracy
    plt.ylim(-0.1, 1,1)
    plt.plot(losses)
    plt.plot(accuracies)
    # plt.show()
    plt.savefig("thing.png")

    #plot the gradients
    plt.ylim(1000)
    plt.imshow(model.l1.weight.grad)
    plt.show()
    plt.imshow(model.l2.weight.grad)
    plt.show()
  

def test(model, x_test, y_test):
    model.eval()
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).reshape(-1, 28 * 28)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)
    output = model(x_test_tensor)
    best = torch.argmax(output, dim=1)
    accuracy = (best == y_test_tensor).float().mean()
    print(f"Accuracy: {accuracy.item()}")

def show_predictions(model, x_test, y_test):
    batch_size = 20
    model.eval()
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)

    samp = np.random.choice(len(x_test), size=batch_size)

    x_test_tensor = torch.tensor(x_test[samp], dtype=torch.float32).reshape(-1, 28 * 28)
    y_test_tensor = torch.tensor(y_test[samp], dtype=torch.int64)
    output = model(x_test_tensor)
    best = torch.argmax(output, dim=1)
    for i in range(batch_size):
        print(f"Prediction: {best[i]}, Label: {y_test_tensor[i]}")
        plt.imshow(x_test_tensor[i].reshape(28, 28))
        plt.show()

def show_mistakes(model, x_test, y_test):
    batch_size = len(x_test)
    model.eval()
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)

    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).reshape(-1, 28 * 28)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)
    output = model(x_test_tensor)
    predicted = torch.argmax(output, dim=1)
    
    mistake_indices = (predicted != y_test_tensor).nonzero()
    
    for i in mistake_indices:
        index = i.item()
        print(f"Prediction: {predicted[index]}, Label: {y_test_tensor[index]}")
        plt.imshow(x_test_tensor[index].reshape(28, 28))
        plt.show()
    
     