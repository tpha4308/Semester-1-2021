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

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from model import Net, train_model

scaler = MinMaxScaler()
path = './og-model'


# In[ ]:


chosen_mask = np.load('subsetGA.npy')


# In[ ]:


# Reading files 

filepath = 'subjective_belief_observers_features_labels.csv'

data = pd.read_csv(filepath)

# Extract features relating to Pupil Dilation 

data.drop(data.columns[1:81], axis=1, inplace=True)

# try shuffle data
data = data.sample(frac=1).reset_index(drop=True)


# In[ ]:


n_input, n_layer1, n_out = 39, 100, 2
loss_func = torch.nn.CrossEntropyLoss()


# In[ ]:


id_list = []
for each in data[data.columns[0]].values:
    pid, vid = each.split('_')
    if pid not in id_list:
        id_list.append(pid)


# # Baseline ANN classification

# In[ ]:


test_accuracy_lists = []
train_accuracy_lists = []
confusion_ls = []
dataset_by_id = {}

correct_trust, correct_doubt = [], []

for pid in id_list:
    
    # Getting all data belong to one participant to be the holdout set
    
    holdout = data[data['pid_vid'].str.match(pid)].copy()
    holdout.drop(holdout.columns[0], axis=1, inplace=True)
    n_features = holdout.shape[1] - 1
    
    test_input = holdout.iloc[:, :n_features]
    test_input = scaler.fit_transform(test_input)
    test_target = holdout.iloc[:, n_features]
    
    # Training set
    
    rest = data[~data['pid_vid'].str.match(pid)].copy()
    rest.drop(rest.columns[0], axis=1, inplace=True)
    train_input = rest.iloc[:, :n_features]
    train_input = scaler.fit_transform(train_input)
    train_target = rest.iloc[:, n_features]

    X_test = torch.Tensor(test_input).float()
    Y_test = torch.Tensor(test_target.values).long()

    X = torch.Tensor(train_input).float()
    Y = torch.Tensor(train_target.values).long()
    
    # Get the initial weights
    
    model_ = Net(n_input, n_layer1, n_out)
    optimiser = torch.optim.Adam(model_.parameters(), lr = 0.0013)
    checkpoint = torch.load(path)
    model_.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    msk = np.ones(n_input).reshape(1,-1)
    model_.set_mask(msk)
    
    num_epochs = 700
    
    train_ls, test_ls, confusion = train_model(X, Y, X_test, Y_test, model_, 
                                               optimiser, loss_func, num_epochs)
    test_accuracy_lists.append(test_ls)
    train_accuracy_lists.append(train_ls)
    confusion_ls.append(confusion)
    
    # Save the trained model for pruning 
    
    path_ = 'save-model/ann-{}'.format(pid)
    torch.save({'model_state_dict': model_.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'activation_vectors': model_.input_to_hidden}, path_)
        
    M = model_.get_activation_vectors()
    dataset_by_id[pid] = [X, Y, X_test, Y_test, M, path_, test_ls[-1], msk]


# In[ ]:


train_accuracy_means = np.mean(train_accuracy_lists, axis = 0)
test_accuracy_means = np.mean(test_accuracy_lists, axis = 0)

fig, axs = plt.subplots(1, 2, figsize = (15,4))
axs[0].plot(train_accuracy_means, color = 'blue')
axs[0].set_title('Training Accuracy')
axs[1].plot(test_accuracy_means, color = 'blue')
axs[1].set_title('Testing Accuracy')
for ax in axs:
    ax.grid(color = 'gray', linestyle = '-', alpha = 0.3)
    ax.set_axisbelow(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')

plt.show()


# In[ ]:


test_accuracy_lists = np.array(test_accuracy_lists)

print('Averaged accuracy:', np.mean(test_accuracy_lists[:,-1]))


# In[ ]:


correct_trust = []
correct_doubt = []

for confusion_test in confusion_ls:
    if torch.sum(confusion_test[:,1]) > 0:
        percent_trust_correctly = confusion_test[1,1]/torch.sum(confusion_test[:,1])
        correct_trust.append(percent_trust_correctly.item())

    if torch.sum(confusion_test[:,0]) > 0:
        percent_doubt_correctly = confusion_test[0,0]/ torch.sum(confusion_test[:,0])
        correct_doubt.append(percent_doubt_correctly.item())
      


# In[ ]:


print('Correct trust:', np.mean(correct_trust) * 100)
print('Correct doubt:', np.mean(correct_doubt) * 100)


# # Classification for ANN + GA model

# In[ ]:


# load the mask found by genetic algo

chosen_msk = np.load('subsetGA.npy')


# In[ ]:


test_accuracy_lists2 = []
train_accuracy_lists2 = []
confusion_ls2 = []
dataset_by_id_ga = {}

for pid in id_list:
    
    # Getting all data belong to one participant to be the holdout set
    
    holdout = data[data['pid_vid'].str.match(pid)].copy()
    holdout.drop(holdout.columns[0], axis=1, inplace=True)
    n_features = holdout.shape[1] - 1
    
    test_input = holdout.iloc[:, :n_features]
    test_input = scaler.fit_transform(test_input)
    test_target = holdout.iloc[:, n_features]
    
    # Training set
    
    rest = data[~data['pid_vid'].str.match(pid)].copy()
    rest.drop(rest.columns[0], axis=1, inplace=True)
    train_input = rest.iloc[:, :n_features]
    train_input = scaler.fit_transform(train_input)
    train_target = rest.iloc[:, n_features]

    X_test = torch.Tensor(test_input).float()
    Y_test = torch.Tensor(test_target.values).long()

    X = torch.Tensor(train_input).float()
    Y = torch.Tensor(train_target.values).long()
    
    # Get the initial weights
    
    model2_ = Net(n_input, n_layer1, n_out)
    optimiser = torch.optim.Adam(model2_.parameters(), lr = 0.0013)
    checkpoint = torch.load(path)
    model2_.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    #chosen_msk = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0,0, 0, 0])
    msk2 = chosen_msk.reshape(1,-1)
    model2_.set_mask(msk2)
    
    num_epochs = 637
    
    train_ls2, test_ls2, confusion2 = train_model(X, Y, X_test, Y_test, model2_, 
                                                  optimiser, loss_func, num_epochs)
    test_accuracy_lists2.append(test_ls2)
    train_accuracy_lists2.append(train_ls2)
    confusion_ls2.append(confusion2)
    
    # Save the trained model for pruning 
    
    path__ = 'save-model/ann-ga-{}'.format(pid)
    torch.save({'model_state_dict': model2_.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'activation_vectors': model2_.input_to_hidden}, path__)
        
    M2 = model2_.get_activation_vectors()
    dataset_by_id_ga[pid] = [X, Y, X_test, Y_test, M2, path__, test_ls2[-1], msk2]


# In[ ]:


train_accuracy_lists2 = np.array(train_accuracy_lists2)
train_accuracy_means2 = np.mean(train_accuracy_lists2, axis = 0)

test_accuracy_lists2 = np.array(test_accuracy_lists2)
test_accuracy_means2 = np.mean(test_accuracy_lists2, axis = 0)
fig, axs = plt.subplots(1, 2, figsize = (15,4))
axs[0].plot(train_accuracy_means2, color = 'blue')
axs[0].set_title('Training Accuracy')
axs[1].plot(test_accuracy_means2, color = 'blue')
axs[1].set_title('Testing Accuracy')
for ax in axs:
    ax.grid(color = 'gray', linestyle = '-', alpha = 0.3)
    ax.set_axisbelow(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')

plt.show()


# In[ ]:


print('Average accuracy for ANN + GA model:', test_accuracy_means2[-1])


# In[ ]:


correct_trust = []
correct_doubt = []

for confusion_test in confusion_ls2:
    if torch.sum(confusion_test[:,1]) > 0:
        percent_trust_correctly = confusion_test[1,1]/torch.sum(confusion_test[:,1])
        correct_trust.append(percent_trust_correctly.item())

    if torch.sum(confusion_test[:,0]) > 0:
        percent_doubt_correctly = confusion_test[0,0]/ torch.sum(confusion_test[:,0])
        correct_doubt.append(percent_doubt_correctly.item())
      


# In[ ]:


print('Correct trust:', np.mean(correct_trust) * 100)
print('Correct doubt:', np.mean(correct_doubt) * 100)


# In[ ]:


# Saving models' parameters for pruning
np.save('dataset_by_id.npy', dataset_by_id)
np.save('dataset_by_id_ga.npy', dataset_by_id_ga)

