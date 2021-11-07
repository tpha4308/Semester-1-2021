#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import libraries

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Net, train_model


# In[3]:


n_input, n_layer1, n_out = 39, 100, 2
model = Net(n_input, n_layer1, n_out)
num_epochs = 525
optimiser = torch.optim.Adam(model.parameters(), lr = 0.0071)
loss_func = torch.nn.CrossEntropyLoss()

path = './og-model'
torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict()}, path)


# In[ ]:




