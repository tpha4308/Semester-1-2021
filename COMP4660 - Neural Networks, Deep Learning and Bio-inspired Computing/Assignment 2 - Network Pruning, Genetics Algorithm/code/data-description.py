#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# Reading files 

filepath = 'subjective_belief_observers_features_labels.csv'

data = pd.read_csv(filepath)

# Extract features relating to Pupil Dilation 

data.drop(data.columns[1:81], axis=1, inplace=True)

# try shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Remove the first column 
data.drop(data.columns[0], axis=1, inplace=True)


# In[3]:


# Separate features and target 

n_features = data.shape[1]-1

X = data.iloc[:, :n_features]
y = data.iloc[:, n_features]


# In[7]:


# Inspect the features by plotting their mean values

X = X.to_numpy()
X_mean = np.mean(X, axis = 0)


# In[24]:


fig, ax = plt.subplots(figsize = (12, 4))
ax.plot(X_mean, color = 'blue')
ax.set_xlabel('Feature')
ax.set_ylabel('Mean Value')
ax.set_axisbelow(True)
ax.grid(color = 'gray', linestyle = '-', alpha = 0.3)


# In[4]:


# Inspect class distribution 

y = y.to_numpy()

print(np.sum(y == 0)/y.shape[0])
print(np.sum(y == 1)/y.shape[0])

