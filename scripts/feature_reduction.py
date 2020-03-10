#!/usr/bin/env python
# coding: utf-8

# # Feature reduction analysis

# In[1]:


"""
Imports
"""

import matplotlib.pyplot as plt

import sys,os
sys.path.insert(0,'ml_tools/')

from ml_tools.descriptors import RawSoapInternal
from ml_tools.models.KRR import KRR,TrainerCholesky,KRRFastCV
from ml_tools.kernels import KernelPower,KernelSum
from ml_tools.utils import get_mae,get_rmse,get_sup,get_spearman,get_score,load_pck,tqdm_cs
from ml_tools.split import KFold,LCSplit,ShuffleSplit
from ml_tools.compressor import FPSFilter
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

import numpy as np
from ase.io import read,write
from ase.visualize import view

from glob import glob
from load_data import *
from math_tools import *
from active_learning import *
from plots import *


# In[2]:


number_of_molecules = 4360294
 #61760 for first 2 dataset #22057374 for all dataset #864898 for first 4 dataset, 4360294 for 6 databases
number_of_molecules_COMP = 83670
number_of_data = 10000
number_of_data_COMP = 10000

# Prepare Active Learning dataset

with open('../ANI-1_release/frames.json', 'r') as fp:
    frames_different_molecule = json.load(fp)
random_indices = np.random.RandomState(seed=882).permutation(np.array(frames_different_molecule['frames']))
print(random_indices.shape)
random_indices = random_indices[:number_of_data]
#frames, Y, mol_indices = load_new_version(random_indices)
frames, Y, mol_indices = load_(random_indices, '../ANI-1_release/ani_gdb_s*.h5')
print(Y.shape, mol_indices.shape)

### Feature Selection with FPS ###
ind_FPS,_,X = FPS_reduction(frames)
print(X.shape)
X_FPS = X[:,ind_FPS[:500]]
print(X_FPS.shape)


# In[3]:


from sklearn.feature_selection import SelectKBest, f_regression

F_test = SelectKBest(f_regression, k = 4000).fit(X,np.ravel(Y))
ind_F_test = F_test.get_support(indices = True)


# In[4]:



random_indices = np.random.RandomState(seed = 2020).permutation(np.arange(number_of_molecules_COMP))   
random_indices = random_indices[:number_of_data_COMP]

frames_benchmark, Y_benchmark, mol_indices_comp = load_COMP(random_indices)

X_benchmark = compute_soap_matrix(frames_benchmark)
weights = ridge_regression(Y, X, np.ones_like(X[0,:].T))
full_errorMAE, full_errorMSE = compute_loss(X_benchmark, weights, Y_benchmark)


# In[5]:


from sklearn.decomposition import KernelPCA

pca = KernelPCA(n_components=4000,kernel='precomputed')
pca.fit(np.dot(X.T,X))
XPCA = pca.transform(X)
X_benchmarkPCA = pca.transform(X_benchmark)
XPCA.shape


# In[25]:


methods = ['F','FPS','PCA']
indices ={
    'F': ind_F_test,
    'FPS':ind_FPS,
}
numbers_steps = 2*np.logspace(0,3,7).astype(int)
vecMAE_AL = np.zeros([len(numbers_steps),len(methods)])
vecMSE_AL = np.zeros([len(numbers_steps),len(methods)])
    

for i,number_of_feature in enumerate(numbers_steps):
    for in_method, method in enumerate(methods):
        if method in indices.keys():
            X_method = X[:,indices[method][:number_of_feature]]
            print(X_method.shape)
            weights = ridge_regression(Y, X_method, np.ones([number_of_feature,1]))
            vecMAE_AL[i, in_method], vecMSE_AL[i, in_method] = compute_loss(X_benchmark[:,indices[method][:number_of_feature]], weights, Y_benchmark)
        elif method == 'PCA':
            X_method = XPCA[:,:number_of_feature]
            X_benchmarkPCA_reduced = X_benchmarkPCA[:,:number_of_feature]
            weights = ridge_regression(Y, X_method, np.ones([number_of_feature,1]))
            vecMAE_AL[i, in_method], vecMSE_AL[i, in_method] = compute_loss(X_benchmarkPCA_reduced, weights, Y_benchmark)


# In[27]:


for in_method, method in enumerate(methods):
    MAE_AL_plot = plt.plot(numbers_steps,vecMSE_AL[:,in_method], linestyle = '--', marker = 'x', label = method + ' MAE')
plt.plot(numbers_steps, full_errorMSE*np.ones_like(numbers_steps), linestyle = '-', color = 'black', label = 'full')
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize = 13)
plt.xlabel('Number of features', fontsize = 13)
plt.ylabel('MSE of prediction on COMP', fontsize = 13)
plt.savefig('../plot/feature_reduction.eps', fontsize = 13)


# In[ ]:




