#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from model import Net, train_model, simple_train_model
scaler = MinMaxScaler()


# In[ ]:


# load the dataset saved from classification 

dataset_by_id = np.load('dataset_by_id.npy', allow_pickle = True).item()
dataset_by_id_ga = np.load('dataset_by_id_ga.npy', allow_pickle = True).item()


# In[ ]:


def angle(a, b):
    num = np.dot(a, b)
    den = np.linalg.norm(a)*np.linalg.norm(b)

    res = np.arccos(np.clip(num/den, -1, 1))
    return np.degrees(res)


# In[ ]:


def unit_pruning(model, M, X_test, Y_test, k, lower_threshold, upper_threshold):
    M = M.detach().numpy()
    
    # normalise activation vectors to -0.5, 0.5
    M = scaler.fit_transform(M)
    M = M - 0.5
    
    to_prune = []
    nprune = 0
    
    for i in range(0, M.shape[1]-1):
        if i not in to_prune:
            col1 = M[:,i]
                
            for j in range(i+1, M.shape[1]-1):
                if j not in to_prune:
                    
                    col2 = M[:,j]
                    a = angle(col1, col2)
                    
                    if a <= lower_threshold:
                        
                        # adding weight
                        model.layer_1.weight.data[j,:] += model.layer_1.weight.data[i,:]
                        model.layer_out.weight.data[:,j] += model.layer_out.weight.data[:,i]
                            
                        # add index to to_prune list    
                        to_prune.append(i)
                        
                        nprune += 1
                        
                        # when number of to be pruned reached the desired sparsity
                        if nprune/M.shape[1] >= k:
                            return prune_and_accuracy(model, to_prune, 
                                                      X_test, Y_test, M.shape[1])
                        
                        break
                            
                    elif a >= upper_threshold:
                            # prune 
                        
                        to_prune.append(i)
                        nprune += 1
                        
                        if nprune/M.shape[1] >= k:
                            return prune_and_accuracy(model, to_prune, 
                                                      X_test, Y_test, M.shape[1])
                        
                        to_prune.append(j)
                        nprune += 1
                        
                        if nprune/M.shape[1] >= k:
                            return prune_and_accuracy(model, to_prune, 
                                                      X_test, Y_test, M.shape[1])

                        break
                        
    return prune_and_accuracy(model, to_prune, X_test, Y_test, M.shape[1])


# In[ ]:


def prune_and_accuracy(model, to_prune, X_test, Y_test, n_uint):
    og = np.arange(0, n_uint, 1)
    to_keep = []
    for n in og:
        if n not in to_prune:
            to_keep.append(n)
            
    iw, ow =  model.layer_1.weight.data.clone(), model.layer_out.weight.data.clone()
    bias = model.layer_1.bias.data.clone()

    # actual prune 
    model.layer_1.weight.data = iw[to_keep]
    model.layer_out.weight.data = ow[:, to_keep]
    model.layer_1.bias.data = bias[to_keep]
    return len(to_prune)


# In[ ]:


# list of prune proportions

prune_p = np.arange(0.05, 1, 0.05)
p = [0]
for each in prune_p:
    p.append(each)


# In[ ]:


# This function is used to set up models saved from classification

def set_up(n_input, n_layer1, n_out, model_path, lr, msk):
    model = Net(n_input, n_layer1, n_out)
    optimiser = torch.optim.Adam(model.parameters(), lr = lr)
        
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    model.set_activation_vectors(M)
    model.set_mask(msk)
    return model


# In[ ]:


learning_rate = 0.0013
n_input, n_layer1, n_out = 39, 100, 2
loss_func = torch.nn.CrossEntropyLoss()


# In[ ]:


lr_ls = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
epochs = np.arange(50, 451, 50)


# # Perform pruning on baseline ANN model

# In[ ]:


num_epoch = 450
big_confusion_ls = {}
for lr in lr_ls:
    conf = []
    big_P = np.zeros((len(p), len(epochs)))

    for pid in dataset_by_id:
        X, Y, X_test, Y_test, M, model_path, accuracy_before_prune, msk = dataset_by_id[pid]

        lower_threshold, upper_threshold = 15, 165
        P = np.zeros((len(p), len(epochs)))
        j = 0
        for k in prune_p:
            model = set_up(n_input, n_layer1, n_out, model_path, learning_rate, msk)

            nprune = unit_pruning(model, M, X_test, Y_test, k, lower_threshold, upper_threshold)
            while nprune/n_layer1 < k:

                model = set_up(n_input, n_layer1, n_out, model_path, learning_rate, msk)
                lower_threshold += 10
                nprune = unit_pruning(model, M, X_test, Y_test, k, lower_threshold, upper_threshold)

            w1, b1, wo, bo = model.state_dict()['layer_1.weight'], model.state_dict()['layer_1.bias'], model.state_dict()['layer_out.weight'], model.state_dict()['layer_out.bias']
            pruned_model = Net(n_input, wo.shape[1], n_out)
            pruned_model.set_weights(w1, b1, wo, bo)
            pruned_model.set_mask(msk)

            optimiser_ = torch.optim.Adam(pruned_model.parameters(), lr = lr)
                
            #retrain
            _, test_acc_ls, confusion = train_model(X, Y, X_test, Y_test, pruned_model, optimiser_, loss_func, num_epoch)
            pk = test_acc_ls[49:450:50]
            #print(pk)
            conf.append(confusion)

            P[j,:] = pk
            j += 1
            
        big_P += P
    
    big_confusion_ls[lr] = conf
    avg_P_for_lr = big_P/len(dataset_by_id)
    
    fig, ax = plt.subplots(figsize = (10,4), sharex = True)

    best_acc0, best_acc1 = np.max(avg_P_for_lr[:,0]), np.max(avg_P_for_lr[:,-1])
    ax.plot(p, avg_P_for_lr[:,0], label = '{}, {}'.format(epochs[0], np.round(best_acc0, 2)), c = 'blue')
    ax.plot(p, avg_P_for_lr[:,-1], label = '{}, {}'.format(epochs[-1], np.round(best_acc1, 2)), c = 'red')

    ax.grid(color = 'gray', linestyle = '-', alpha = 0.3)
    ax.set_title('Learning rate = {}'.format(lr))
    ax.set_xlabel('Prune proportion (out of 1)')
    ax.set_ylabel('Testing Accuracy (%)')
    ax.set_axisbelow(True)
    ax.set_xticks(p)
    plt.legend()
    plt.show()


# # Perform pruning on ANN + GA model

# In[ ]:


num_epoch = 450
confusion_ls = []
for lr in lr_ls:
    big_P = np.zeros((len(p), len(epochs)))

    for pid in dataset_by_id_ga:
        X, Y, X_test, Y_test, M, model_path, accuracy_before_prune, msk = dataset_by_id_ga[pid]

        lower_threshold, upper_threshold = 15, 165
        P = np.zeros((len(p), len(epochs)))
        j = 0
        for k in prune_p:
            model = set_up(n_input, n_layer1, n_out, model_path, learning_rate, msk)

            nprune = unit_pruning(model, M, X_test, Y_test, k, lower_threshold, upper_threshold)
            while nprune/n_layer1 < k:

                model = set_up(n_input, n_layer1, n_out, model_path, learning_rate, msk)
                lower_threshold += 10
                nprune = unit_pruning(model, M, X_test, Y_test, k, lower_threshold, upper_threshold)

            w1, b1, wo, bo = model.state_dict()['layer_1.weight'], model.state_dict()['layer_1.bias'], model.state_dict()['layer_out.weight'], model.state_dict()['layer_out.bias']
            pruned_model = Net(n_input, wo.shape[1], n_out)
            pruned_model.set_weights(w1, b1, wo, bo)
            pruned_model.set_mask(msk)

            optimiser_ = torch.optim.Adam(pruned_model.parameters(), lr = lr)
                
            #retrain
            _, test_acc_ls, confusion = train_model(X, Y, X_test, Y_test, pruned_model, optimiser_, loss_func, num_epoch)
            pk = test_acc_ls[49:450:50]
            #print(pk)
            confusion_ls.append(confusion)

            P[j,:] = pk
            j += 1
            
        big_P += P
        
    avg_P_for_lr = big_P/len(dataset_by_id)
    
    
    fig, ax = plt.subplots(figsize = (10,4), sharex = True)

    best_acc0, best_acc1 = np.max(avg_P_for_lr[:,0]), np.max(avg_P_for_lr[:,-1])
    ax.plot(p, avg_P_for_lr[:,0], label = '{}, {}'.format(epochs[0], np.round(best_acc0, 2)), c = 'blue')
    ax.plot(p, avg_P_for_lr[:,-1], label = '{}, {}'.format(epochs[-1], np.round(best_acc1, 2)), c = 'red')

    ax.grid(color = 'gray', linestyle = '-', alpha = 0.3)
    ax.set_title('Learning rate = {}'.format(lr))
    ax.set_xlabel('Prune proportion (out of 1)')
    ax.set_ylabel('Testing Accuracy (%)')
    ax.set_axisbelow(True)
    ax.set_xticks(p)
    plt.legend()
    plt.show()


# In[ ]:




