#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import libraries

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from model import Net, simple_train_model


# In[ ]:


# Reading files 

filepath = 'subjective_belief_observers_features_labels.csv'

data = pd.read_csv(filepath)

# Extract features relating to Pupil Dilation 

data.drop(data.columns[1:81], axis=1, inplace=True)

# try shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Remove the first column 
data.drop(data.columns[0], axis=1, inplace=True)


# In[ ]:


n_features = data.shape[1]-1

# Separate features and target 
X = data.iloc[:, :n_features]
y = data.iloc[:, n_features]

# Normalise data to be within 0-1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Prepare data 
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y.values,test_size = 0.2)

X_test = torch.Tensor(X_test).float()
Y_test = torch.Tensor(y_test).long()

X = torch.Tensor(X_train).float()
Y = torch.Tensor(y_train).long()


# In[ ]:


# Initial configuration 

path = './og-model'
n_input, n_layer1, n_out = 39, 100, 2

lr = np.arange(0.0001, 0.011, 0.0003)
epoch = np.arange(50, 1001, 25)

loss_func = torch.nn.CrossEntropyLoss()
msk = np.ones(n_input).reshape(1,-1)


# In[ ]:


# Inspect param_dic to find the best parameters

param_dic = {}

for learning_rate in lr:
    for num_epoch in epoch:
        model = Net(n_input, n_layer1, n_out)
        optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.set_mask(msk)
        test_accuracy = simple_train_model(X, Y, X_test, Y_test, model, optimiser, loss_func, num_epoch)
        param_dic[(learning_rate, num_epoch)] = test_accuracy

