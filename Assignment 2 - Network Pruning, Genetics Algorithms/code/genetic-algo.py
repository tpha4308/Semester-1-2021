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
import scipy.stats as ss

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[ ]:


from model import Net, train_model, simple_train_model
path = './og-model'


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


# Initial configurations for GA

DNA_size = n_features
population_size = 10*n_features
mutation_rate = 1/n_features


# In[ ]:


def selection(s_name, chosen, population_size, selection_prob):
    # Selection operator: Continuously choosing each individual based on their selection probabilty 
    # Final chosen list will be of size N/2
    
    while len(chosen) < population_size/2:
        x = np.random.choice(s_name, 1, p = selection_prob)
        while x in chosen:
            x = np.random.choice(s_name, 1, p= selection_prob)
        chosen.append(x[0])
    return chosen


# In[ ]:


def crossover(mating_pool, mutation_rate, trained_models):
    # Crossover operator: Choosing 2 parents at random and let them recombine + produce 2 children
    # Mutation operator is also included in this function
    next_gen = []

    parents_index = np.arange(0, len(mating_pool))
    np.random.shuffle(parents_index)
    
    for i in range(0,len(parents_index)-1,2):
        parent1 = mating_pool[i]
        parent2 = mating_pool[i+1]

        for nbabies in range(2):
            new_baby = []
            for n in range(len(parent1)):
                feature_to_choose = np.random.random_sample()
                
                # Selecting which parent does the n feature comes from
                if np.random.random_sample() < 0.5:
                    new_baby.append(parent1[n])
                else:
                    new_baby.append(parent2[n])
                    
            # Mutation, if a random number is less than the mutation rate then flip the gene
            if np.random.random_sample() < mutation_rate:
                if new_baby[n] == '1':
                    new_baby[n] = '0'
                else:
                    new_baby[n] = '1'
            new_baby_dna = ''.join(new_baby)
            
            # If the new model is already explored before, then discard the model
            if new_baby_dna not in trained_models: 
                next_gen.append(new_baby_dna)
            
    return next_gen


# In[ ]:


num_epochs = 700
loss_func = torch.nn.CrossEntropyLoss()
n_input, n_layer1, n_out = 39, 100, 2


# In[ ]:


trained_models = {}

selection_pressure = np.arange(1, 2.1, 0.1)


# In[ ]:


# This code chunk both check for the best selection pressure constant k by performing GA 
# for each k, the k that gives the model with best accuracy will be chosen
# that model will also be chosen for ANN + GA

best_acc = float('-inf')
best_model = None

for k in selection_pressure:
    print('k = {}'.format(k))
    chosen_models = {}
    population_size = n_features*10
    x = np.random.randint(0,2,(population_size,39))
    x[0,:] = np.ones(39)
    
    while len(x) > 1:
        model_with_accuracies = []
        
        # train the population 
        for i in range(len(x)):
            m = [str(elem) for elem in x[i]]
            m_name = ''.join(m)
            if m_name in trained_models:
                model_with_accuracies.append((trained_models[m_name], m_name))

            else:
                model_ = Net(n_input, n_layer1, n_out)
                optimiser = torch.optim.Adam(model_.parameters(), lr = 0.0013)
                checkpoint = torch.load(path)
                model_.load_state_dict(checkpoint['model_state_dict'])
                optimiser.load_state_dict(checkpoint['optimizer_state_dict'])

                msk = x[i].reshape(1,-1)
                model_.set_mask(msk)
                test_accuracy = simple_train_model(X, Y, X_test, Y_test, model_, 
                                                     optimiser, loss_func, num_epochs)

                model_with_accuracies.append((test_accuracy, m_name))
                trained_models[m_name] = test_accuracy
                chosen_models[m_name] = test_accuracy

        # rank 
        s = sorted(model_with_accuracies)
        s_accuracies = [v[0] for v in s]
        s_name = [v[1] for v in s]
        ranking_s = ss.rankdata(s_accuracies)

        # give a fitness value
        fitness_values = [i*k for i in ranking_s]

        # probability to be chosen for next gen
        selection_prob = [elem/np.sum(fitness_values) for elem in fitness_values]

        # elitism 
        top1percent = s_name[int(len(s)*0.99):]
        top2to1percent = s_name[int(len(s)*0.98):int(len(s)*0.99)]

        # selection
        chosen = [v for v in top1percent]
        for each in top2to1percent:
            chosen.append(each)

        chosen = selection(s_name, chosen, population_size, selection_prob)

        # cross-over
        mating_pool = chosen[len(top1percent):]

        next_gen = crossover(mating_pool, mutation_rate, chosen_models)

        for each in top1percent:
            next_gen.append(each)

        # finish one loop, set the new gen to be the current gen
        x_ = [[int(val) for val in next_gen[i]] for i in range(len(next_gen))]
        x = np.array(x_)
        population_size = len(next_gen)
    
    
    print('done')
    print(x)
    chosen_model = Net(n_input, n_layer1, n_out)
    optimiser = torch.optim.Adam(chosen_model.parameters(), lr = 0.0071)
    checkpoint = torch.load(path)
    chosen_model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    chosen_model.set_mask(x)

    test_acc  = simple_train_model(X, Y, X_test, Y_test, chosen_model, optimiser, loss_func, num_epochs)
    print(test_acc)
    print()
    
    if test_acc > best_acc:
        best_acc = test_acc
        best_model = x


# In[ ]:


# saving the best model to be used in ANN + GA

np.save('subsetGA.npy', best_model)


# In[ ]:




