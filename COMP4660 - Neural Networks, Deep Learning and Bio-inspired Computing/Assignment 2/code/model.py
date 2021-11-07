
# import libraries

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Neural network

class Net(nn.Module):
    def __init__(self, n_input, n_layer1, n_out=1):
        super(Net, self).__init__()
        self.n_input = n_input
        self.n_layer1 = n_layer1
        self.n_out = n_out

        self.layer_1 = nn.Linear(self.n_input, self.n_layer1)
        self.layer_out = nn.Linear(self.n_layer1, self.n_out)
        
        self.input_to_hidden = None

    def forward(self, inputs):
        mask_full = self.mask.repeat(len(inputs), axis = 0)
        mask_full = torch.Tensor(mask_full).float()
        inputs_masked = torch.mul(mask_full, inputs)

        z1 = torch.sigmoid(self.layer_1(inputs_masked)) 
        self.input_to_hidden = z1
        z2 = self.layer_out(z1)
        
        return z2
    
    def set_mask(self, mask):
        self.mask = mask

    def get_weights(self):
        return self.layer_1.weight.data, self.layer_out.weight.data
    
    def set_weights(self, layer1_weights, layer1_bias, layerout_weights, layerout_bias):
        with torch.no_grad():
            self.layer_1.weight.data = layer1_weights
            self.layer_1.bias.data = layer1_bias
            self.layer_out.weight.data = layerout_weights
            self.layer_out.bias.data = layerout_bias
    
    def get_activation_vectors(self):
        return self.input_to_hidden
    
    def set_activation_vectors(self, M):

        self.input_to_hidden = M


def train_model(X, Y, X_test, Y_test, model, optimiser, loss_func, num_epochs):
    test_accuracy_ls = []
    train_accuracy_ls = []
    
    model.train()
    for epoch in range(num_epochs):

        optimiser.zero_grad(set_to_none= True)  
        Y_pred = model(X)

        loss = loss_func(Y_pred, Y)

        _, predicted = torch.max(Y_pred, 1)
        # calculate and print accuracy
        total = predicted.size(0)
        correct = sum(predicted.data.numpy() == Y.data.numpy())
        train_accuracy = 100 * correct / total
        train_accuracy_ls.append(train_accuracy)
            
        with torch.no_grad():
             
            model.eval()
            Y_pred_test = model(X_test)

            # get prediction
            _, predicted_test = torch.max(Y_pred_test, 1)

            # calculate accuracy
            total_test = predicted_test.size(0)
            correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

            test_accuracy = 100 * correct_test / total_test
            test_accuracy_ls.append(test_accuracy)
                
            if epoch == num_epochs-1:
                # print confusion matrix 
                confusion_test = torch.zeros(model.n_out, model.n_out)

                for i in range(Y_test.shape[0]):
                    actual_class_ = Y_test.data[i]
                    predicted_class_ = predicted_test.data[i]

                    confusion_test[actual_class_][predicted_class_] += 1

        loss.backward()
        optimiser.step()
    
    return train_accuracy_ls, test_accuracy_ls, confusion_test

def simple_train_model(X, Y, X_test, Y_test, model, optimiser, loss_func, num_epochs):

    model.train()
    for epoch in range(num_epochs):

        optimiser.zero_grad(set_to_none= True)  
        Y_pred = model(X)

        loss = loss_func(Y_pred, Y)

        _, predicted = torch.max(Y_pred, 1)
        
        if epoch == num_epochs-1:
            
            with torch.no_grad():

                model.eval()
                Y_pred_test = model(X_test)

                # get prediction
                _, predicted_test = torch.max(Y_pred_test, 1)

                # calculate accuracy
                total_test = predicted_test.size(0)
                correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

                test_accuracy = 100 * correct_test / total_test
                return test_accuracy

        loss.backward()
        optimiser.step()

